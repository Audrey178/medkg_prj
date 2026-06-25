# SPEC: Nhánh A bổ sung (gold_pmid mode) + Nhánh B — VectorStore Retrieval Layer

## PHẦN 0 — Nhánh A: thêm luồng BioASQ end-to-end (Agent 1 → Agent 2)

### Bối cảnh: vì sao cần sửa cả Agent 1, không chỉ Agent 2

`DiseaseProfiler.run()` hiện tại nhận `{disease_id, disease_name}` và thực hiện 6 bước, đa số là **gọi API ngoài để dò thông tin một disease đã biết tên** (OMIM lookup, GeneReviews check qua NCBI esearch, PubMed-count estimation, sinh `recommended_pubmed_queries` để Agent 2 đi search). BioASQ sample KHÔNG cung cấp `disease_name` sạch — nó cung cấp `body` (câu hỏi), `documents` (PMID có sẵn), `snippets` (text+offset có sẵn), `concepts` (URI Disease Ontology/MeSH), `ideal_answer`, `type`, `id`. Toàn bộ việc "dò" của Agent 1 hiện tại là vô nghĩa ở đây — nguồn đã có sẵn, không cần dò.

→ Quyết định: **Agent 1 vẫn chạy** (giữ đúng vị trí trong pipeline, đúng pattern `BaseAgent`/`AgentResult` để đồng bộ logging), nhưng khi `input_data["mode"] == "bioasq_data"`, nó **bỏ qua toàn bộ 6 bước gọi API cũ**, chỉ đọc trực tiếp file BioASQ và build ra một class profile mới — **`BioASQProfile`** — hoàn toàn độc lập với `DiseaseProfile` (không kế thừa, không dùng chung field, vì bản chất dữ liệu khác hẳn: `DiseaseProfile` là kết quả suy ra/dò từ nhiều API, `BioASQProfile` là copy 1-1 từ JSON đã có sẵn).

### Bước 0.1 — Thêm `BioASQProfile` vào `models.py`

Vị trí: đặt cạnh `DiseaseProfile` trong khối "Agent communication models" (`models.py`, sau dòng ~400).

```python
@dataclass
class BioASQProfile:
    """
    Profile cho 1 sample BioASQ Task B, build bởi Agent 1 khi mode="bioasq_data".
    KHÔNG kế thừa DiseaseProfile — bản chất khác: đây là copy 1-1 từ JSON
    BioASQ đã có sẵn (PMID, snippet, concept), không phải kết quả dò/suy ra
    từ API ngoài như DiseaseProfile. Giữ nguyên cấu trúc gốc, không rút gọn,
    để không mất thông tin cho các bước downstream có thể cần sau (ví dụ
    ideal_answer dùng để so sánh/đánh giá benchmark, concepts dùng cho
    Tier 3 ontology work sau này).
    """
    bioasq_id: str                        # item["id"]
    question_body: str                    # item["body"]
    question_type: str                    # item["type"] — "summary" | "yesno" | "factoid" | "list"
    pmids: list[str] = field(default_factory=list)              # parse từ item["documents"] (URL -> số PMID cuối)
    document_urls: list[str] = field(default_factory=list)       # item["documents"] nguyên bản, giữ để trace ngược
    snippets: list[dict] = field(default_factory=list)           # item["snippets"] nguyên bản (text, offset, beginSection, document)
    concepts: list[str] = field(default_factory=list)            # item["concepts"] nguyên bản (URI DOID/MeSH)
    ideal_answer: list[str] = field(default_factory=list)        # item["ideal_answer"] nguyên bản

    # Field phụ trợ — build trong Agent 1, dùng cho logging/downstream, không phải từ JSON gốc
    pmids_with_snippet: list[str] = field(default_factory=list)  # subset của pmids có >=1 snippet khớp
    pmids_missing_snippet: list[str] = field(default_factory=list)  # subset của pmids KHÔNG có snippet nào khớp — Agent 2 sẽ bỏ qua các PMID này theo spec đã chốt ở Phần 0.2

    def to_dict(self) -> dict:
        return {
            "bioasq_id": self.bioasq_id,
            "question_body": self.question_body,
            "question_type": self.question_type,
            "pmids": self.pmids,
            "document_urls": self.document_urls,
            "snippets": self.snippets,
            "concepts": self.concepts,
            "ideal_answer": self.ideal_answer,
            "pmids_with_snippet": self.pmids_with_snippet,
            "pmids_missing_snippet": self.pmids_missing_snippet,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BioASQProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

Lưu ý field `pmids_with_snippet`/`pmids_missing_snippet`: tính sẵn ở Agent 1 (so khớp `snippet["document"]` với từng PMID trong `document_urls`) để Agent 2 không phải làm lại việc này, và để `AgentResult.metrics` của Agent 1 có thể báo cáo ngay `skipped_no_snippet_match` (xem Bước 0.3) mà không cần chạy Agent 2 trước.

### Bước 0.2 — Sửa `disease_profiler.py`: thêm mode `bioasq_data`

Format file input đã chốt: JSON thường, dạng `{"questions": [ {...item...}, {...item...}, ... ]}` (đúng format BioASQ Task B gốc, ví dụ đã có trong dữ liệu mẫu phiên trước).

Sửa `DiseaseProfiler.run()`:

```python
async def run(self, input_data: dict) -> AgentResult:
    if input_data.get("mode") == "bioasq_data":
        return await self._run_bioasq_mode(input_data)
    # ... toàn bộ logic disease-driven hiện tại giữ nguyên, không đổi ...
```

Thêm method mới `_run_bioasq_mode(self, input_data: dict) -> AgentResult`:

- `input_data["file_path"]`: đường dẫn tới file JSON BioASQ.
- Đọc file, lấy `data["questions"]` (list các item).
- Với mỗi item:
  - Parse `pmids = [url.rsplit("/", 1)[-1] for url in item["documents"]]`.
  - Tính `pmids_with_snippet`/`pmids_missing_snippet` bằng cách so khớp `snippet["document"].rsplit("/", 1)[-1]` với từng pmid trong `pmids`.
  - Build `BioASQProfile(bioasq_id=item["id"], question_body=item["body"], question_type=item["type"], pmids=pmids, document_urls=item["documents"], snippets=item.get("snippets", []), concepts=item.get("concepts", []), ideal_answer=item.get("ideal_answer", []), pmids_with_snippet=..., pmids_missing_snippet=...)`.
  - Nếu item thiếu field bắt buộc (`id`, `body`, `documents`) → bỏ qua item đó, ghi log, tính vào số lỗi (không raise exception làm hỏng cả batch).
- KHÔNG gọi `_resolve_identity`, `_query_omim`, `_check_tier1_sources`, `_check_genereviews`, `_profile_primekg`, `_estimate_literature`, `_generate_strategy`, `_identify_differentials`, `_save_config` — toàn bộ các method này thuộc luồng disease-driven cũ, không đụng tới trong mode `bioasq_data`.
- Trả về:
  ```python
  return AgentResult(
      agent_name="DiseaseProfiler",
      disease_id="BIOASQ_BATCH",
      status="success" if error_count == 0 else ("partial" if profiles else "failed"),
      data={"profiles": [p.to_dict() for p in profiles]},
      metrics={
          "total_samples": total_items_in_file,
          "valid_profiles": len(profiles),
          "skipped_invalid_item": error_count,
          "skipped_no_snippet_match": sum(len(p.pmids_missing_snippet) for p in profiles),
      },
      timestamp=datetime.utcnow(),
  )
  ```
  (`status` = `"success"` nếu không có item lỗi nào, `"partial"` nếu có ít nhất 1 profile hợp lệ nhưng cũng có item lỗi, `"failed"` nếu không build được profile nào — theo đúng 3 giá trị `status` đã dùng trong `BaseAgent`/`AgentResult` hiện có, không tạo giá trị status mới.)

### Bước 0.3 — Sửa `evidence_harvester.py`: `_harvest_from_bioasq_gold()` nhận `BioASQProfile`, không nhận raw dict

(Phần này thay thế hoàn toàn bản trước — trước đây spec ghi nhận `bioasq_item: dict` trực tiếp, giờ Agent 1 đã chuẩn hóa thành `BioASQProfile`, Agent 2 nhận đúng object đó.)

#### Quyết định đã chốt (giữ từ phiên trước, không đổi)

- **Không gọi `esearch`** — danh sách PMID đã biết trước.
- **Vẫn gọi `EFetch` một lần** cho đúng `profile.pmids_with_snippet` (KHÔNG fetch cho `pmids_missing_snippet` — đã biết trước là sẽ bị bỏ qua vì không có snippet, fetch metadata cho chúng là vô ích) để lấy `journal`/`publication_date`/`pub_types` thật, phục vụ `credibility_score` đúng.
- `study_type` mặc định khi không lấy được `pub_types` là `StudyType.OTHER` — không tự nâng cấp.
- **CẢNH BÁO giữ nguyên từ phiên trước**: không tái dùng thẳng `fetch_abstracts()`/`_parse_article()` nguyên bản — 2 filter ngầm (`MIN_ABSTRACT_WORDS`, `EXCLUDED_PUB_TYPES`) sẽ xóa mất metadata của PMID gold. Cần viết parser metadata-only riêng (`_parse_article_metadata_only()` hoặc thêm `skip_quality_filter: bool = False` vào `_parse_article()` hiện có).

#### Thay đổi cụ thể trong `evidence_harvester.py`

1. Đổi signature: `_harvest_from_bioasq_gold(self, profile: BioASQProfile, collection: EvidenceCollection) -> None` (nhận `BioASQProfile`, import từ `core.models`, không nhận `dict` thô nữa).
2. Logic bên trong:
   - Gọi 1 lần EFetch metadata-only cho `profile.pmids_with_snippet` (EPost+EFetch batch như cơ chế hiện có, dùng parser mới không filter) → `dict[pmid, {"journal":..., "publication_date":..., "pub_types":...}]`. Lỗi/timeout → log warning, tiếp tục với metadata rỗng, không chặn pipeline.
   - Với mỗi `pmid` trong `profile.pmids_with_snippet` (đã biết chắc có snippet, không cần check lại):
     - Lấy các `snippet` trong `profile.snippets` có `snippet["document"].rsplit("/", 1)[-1] == pmid`.
     - Nối các đoạn `snippet["text"]`, giữ phân biệt theo `beginSection` (ghép `"## {beginSection}\n{text}"`, nối bằng `\n\n`) → `SourceDocument.text`.
     - Map metadata EFetch được vào `SourceDocument.journal`, `.publication_date`; `pub_types` đưa vào `self._classify_study_type()` có sẵn.
     - `source_id = f"PMID:{pmid}"`, `source_type = "bioasq_snippet"`, `tier = EvidenceTier.TIER_2`.
     - `credibility_score` dùng `self.scorer.compute(journal_name=..., publication_date=..., citation_count=None, study_type=..., is_retracted=False)` — giữ nguyên hàm có sẵn. `citation_count=None` là nhất quán với luồng Tier-2-search-thường hiện tại (đã luôn `None`, xem dòng ~736).
     - Append vào `collection.tier2_documents`.
   - `profile.pmids_missing_snippet` đã được Agent 1 xác định trước — không xử lý gì thêm ở đây ngoài log (ví dụ `self.logger.info("Bỏ qua %d PMID không có snippet: %s", len(profile.pmids_missing_snippet), profile.pmids_missing_snippet)`).
3. `AgentResult.data["mode"] = "gold_pmid"` để pipeline downstream biết nhánh nào tạo ra collection này.
4. Method `run()` của `EvidenceHarvester`: nếu `input_data` chứa `bioasq_profile` (1 `BioASQProfile`, lấy từ `list[BioASQProfile]` mà Agent 1 trả ra rồi orchestrator lặp qua từng profile gọi Agent 2 1 lần/profile) → gọi `_harvest_from_bioasq_gold()` thay vì `_harvest_tier2()`. Tier 1 (OMIM/GeneReviews/Orphanet) vẫn chạy như cũ cho luồng disease-driven — không tắt, không đụng.

### Bàn giao Phần 0

- `models.py`: thêm `BioASQProfile` dataclass (Bước 0.1).
- `disease_profiler.py`: thêm `_run_bioasq_mode()`, nhánh điều kiện trong `run()` (Bước 0.2). **Không sửa 8 method cũ** (`_resolve_identity`, `_query_omim`, `_check_tier1_sources`, `_check_genereviews`, `_profile_primekg`, `_estimate_literature`, `_generate_strategy`, `_identify_differentials`, `_save_config`).
- `evidence_harvester.py`: đổi signature `_harvest_from_bioasq_gold()` nhận `BioASQProfile`, thêm parser metadata-only, sửa nhánh điều kiện trong `run()` (Bước 0.3).
- Verify: chạy `DiseaseProfiler` với mode `bioasq_data` trên file mẫu chứa item Hirschsprung disease (9 PMID, có snippets) → xác nhận `AgentResult.data["profiles"]` có đúng 1 `BioASQProfile` với `pmids_with_snippet` đủ 9 (hoặc log rõ nếu thiếu). Sau đó chạy `EvidenceHarvester._harvest_from_bioasq_gold()` với profile đó → xác nhận `EvidenceCollection.tier2_documents` có đủ document, `credibility_score` khác `0.0` mặc định bất hợp lý nếu EFetch thành công.
- **Không sửa `KnowledgeExtractor` (Agent 3)** — input chỉ còn 9-15 `SourceDocument`/sample, cost tự giảm.

---

# Nhánh B — VectorStore Retrieval Layer (Textbook + BioASQ snippets)

## Bối cảnh tổng thể (đọc trước khi code bất kỳ phần nào)

MEDKG đang chuyển sang kiến trúc hybrid gồm 2 nhánh độc lập:

- **Nhánh A (Knowledge Graph)** — build từ BioASQ qua 4-agent pipeline hiện tại, với Agent 1 (`DiseaseProfiler`, mode `bioasq_data` → `BioASQProfile`) và Agent 2 (`EvidenceHarvester`, `_harvest_from_bioasq_gold()` + EFetch metadata-only) đã được mở rộng ở Phần 0 phía trên. Agent 3 (`KnowledgeExtractor`) và Agent 4 (`QualityController`) không đổi. Phần 0 đã xử lý xong phạm vi sửa đổi cần thiết trong Nhánh A cho spec này — **các phần còn lại của 4 agent (ngoài đúng thay đổi ở Phần 0) không thuộc phạm vi, không sửa thêm.**
- **Nhánh B (VectorStore retrieval)** — phạm vi chính của phần này. Nạp 2 loại nguồn vào một vector store dùng FAISS:
  1. 4 file textbook (`Anatomy_Gray.txt`, `Biochemistry_Lippincott.txt`, `Cell_Biology_Alberts.txt`, `First_Aid_Step1.txt`, ở `/mnt/user-data/uploads/`)
  2. BioASQ snippets (text verbatim + offset đã có sẵn trong dataset BioASQ, KHÔNG fetch lại từ NCBI cho nhánh này — lưu ý đây là bản COPY riêng cho VectorDB, độc lập với việc Phần 0 có EFetch metadata-only cho Nhánh A; hai việc không loại trừ nhau vì mục đích khác nhau: Phần 0 cần metadata cho credibility_score của KG, Phần B chỉ cần text để embed)

Nhánh B **không tạo `RawTriple`**, không qua `QualityController`, không cần `EvidenceTier`/`credibility_score` — đây là corpus retrieval thuần, không phải knowledge-graph claim cần verify từng cái.

Mục tiêu cuối: khi có một câu hỏi (MedQA hoặc bất kỳ), hệ thống **query song song cả VectorStore và MEDKG (KG)**, gộp context từ cả hai, không phân biệt nguồn nào chính/phụ — đây đã là quyết định chốt, không cần hỏi lại.

---

## PHẦN 1 — Chunking textbook (rà soát trước, không code mù)

### Vấn đề đã phát hiện ở khảo sát sơ bộ

4 file đều là `.txt` thuần (kết quả OCR/convert từ PDF, không có markdown). Khảo sát sơ bộ (không phải kết luận cuối — xem Bước 0 dưới) cho thấy:

| File | Dòng non-empty | Độ dài dòng TB | Độ dài dòng max | Heading rõ ràng? |
|---|---|---|---|---|
| `Biochemistry_Lippincott.txt` | 8588 | ngắn | — | **CÓ** — `^[IVXLCDM]+\.\s+[A-Z]` match sạch: `I. OVERVIEW`, `II. STRUCTURE`, `III. ACIDIC AND BASIC PROPERTIES`... |
| `Anatomy_Gray.txt` | 9972 | 226.8 | 2563 | **KHÔNG** rõ — gần như không có heading dòng riêng dạng số/La Mã đáng tin; layout gốc (cột, hình, caption) có vẻ đã bị làm phẳng thành văn bản liên tục |
| `Cell_Biology_Alberts.txt` | 10627 | 456.1 | 7671 (!) | **KHÔNG** — có dòng `Chapter N: ...` nhưng không có sub-heading tách dòng nhất quán; nhiều dòng dài bất thường gợi ý đoạn văn + citation/figure caption bị nối lẫn vào nhau |
| `First_Aid_Step1.txt` | 4245 | 154.7 | 1744 | **KHÔNG** — không có heading in hoa tách dòng; nội dung là đoạn fact-dense ngắn, lẫn cả caption hình minh họa vào câu văn |

→ **Không dùng 1 regex heading chung cho cả 4 file.** Ép cùng cách chunk sẽ tạo chunk rác cho 3/4 file.

### Bước 0 — Tự verify lại, không tin bảng trên là đủ

Trước khi viết chunker, đọc trực tiếp 20-30 đoạn ngẫu nhiên mỗi file (đầu, 25%, 50%, 75%, cuối file — không chỉ đầu file) bằng script Python, in ra quan sát bằng mắt. Xác nhận:
1. Pattern heading thật của từng file (có thể bảng trên bỏ sót — ví dụ Gray's Anatomy có thể có heading kiểu tên xương/cơ quan in hoa, hoặc số mục kiểu `8.1`).
2. Rác lẫn vào (figure caption, reference list, page number, running header/footer lặp lại mỗi trang).
3. Đơn vị ngữ nghĩa tự nhiên thực tế của từng file (câu hoàn chỉnh? fact ngắn kiểu First Aid? đoạn mô tả dài kiểu Alberts?).

### Bước 1 — Chiến lược chunk RIÊNG cho từng file

Dựa trên Bước 0, chọn 1 trong các chiến lược sau cho MỖI file (không bắt buộc giống nhau):

- **Heading-based**: dùng khi file có heading rõ, nhất quán (khả năng cao: Lippincott). `section_heading` = heading text thật.
- **Sliding window theo câu**: tách câu bằng `. `/`? `/`! ` có aware viết tắt y khoa (`e.g.`, `i.e.`, `Fig.`, `vs.`, `Dr.`), ghép thành chunk ~500-800 token, overlap ~10-15%. Dùng khi không có heading đáng tin (khả năng cao: Gray, Alberts, First Aid).
- **Hybrid**: heading lớn (ví dụ `Chapter N:`) chia block lớn, rồi sliding-window bên trong block (cần cho Alberts vì có dòng dài 7671 ký tự không thể giữ nguyên làm 1 chunk).

Ghi rõ trong code comment lý do chọn chiến lược, dựa trên bằng chứng cụ thể từ Bước 0 — không suy luận từ bảng khảo sát sơ bộ ở trên.

### Bước 2 — Lọc rác trước khi chunk

Loại hoặc tách riêng:
- Citation/reference block giữa câu (ví dụ `M. Baron, D.G. Norman..., Trends Biochem. Sci. 16:13–17, 1991, with permission from Elsevier`).
- Running header/footer lặp lại đều đặn (dấu hiệu: cùng 1 dòng ngắn xuất hiện > N lần cách đều trong file).
- Page number đơn lẻ trên 1 dòng.

Không cần sạch 100% — mục tiêu giảm nhiễu rõ rệt.

### Bước 3 — Output: `VectorChunk` dataclass

```python
@dataclass
class VectorChunk:
    chunk_id: str          # "TEXTBOOK:First_Aid_Step1:0042" | "BIOASQ:PMID:15829955:snip_02"
    source_type: str       # "textbook" | "bioasq_snippet"
    source_name: str       # "First_Aid_Step1" | "PMID:15829955"
    section_heading: str   # heading thật nếu có, "" nếu không
    text: str              # nội dung sau khi lọc rác (Bước 2)
    char_start: int        # offset trong file/snippet gốc — bắt buộc, đây là traceability
    char_end: int
    embedding: Optional[np.ndarray] = None  # gán ở bước embed, không gán lúc chunk
```

`chunk_id` phải duy nhất, trace ngược được vị trí gốc qua `char_start`/`char_end` — tương đương tinh thần "PMID-traceable" của Nhánh A, áp cho nguồn không có PMID.

### Bước 4 — QA bắt buộc sau chunk (trước khi coi là xong)

In ra:
1. Số chunk mỗi file, độ dài trung vị/min/max (ký tự hoặc token).
2. 5 chunk ngẫu nhiên mỗi file để đọc thủ công — không cắt cụt vô lý giữa câu, không còn rác citation/header rõ rệt.
3. Nếu phát hiện vấn đề — quay lại Bước 1/2, KHÔNG tiến sang Phần 2 với chunk lỗi.

### Bàn giao Phần 1

File `retrieval/textbook_chunker.py`:
- `detect_chunking_strategy(file_path: str) -> dict` — trả về chiến lược + bằng chứng quan sát được (để review lại).
- `chunk_textbook(file_path: str, strategy: dict) -> list[VectorChunk]`.
- `if __name__ == "__main__":` chạy thử cả 4 file, in kết quả Bước 4.

File `retrieval/bioasq_snippet_loader.py`:
- `load_bioasq_snippets(json_path: str) -> list[VectorChunk]` — đọc file JSON BioASQ (format `{"questions": [...]}`, cùng file có thể dùng cho cả Agent 1 ở Phần 0 và loader này — đây là **2 đường đọc độc lập có chủ đích**, không cần hợp nhất: Agent 1 build `BioASQProfile` để nuôi Nhánh A/KG qua `EvidenceHarvester`, loader này chỉ cần text+offset để nhúng vào VectorStore, không cần đi qua `BioASQProfile`/`AgentResult`/pattern agent gì cả). Với mỗi item trong `data["questions"]`, với mỗi `snippet` trong `item["snippets"]` tạo 1 `VectorChunk`:
  - `chunk_id = f"BIOASQ:PMID:{pmid}:snip_{i:02d}"` (lấy `pmid` từ `snippet["document"]`, parse phần số cuối URL `.../pubmed/{pmid}`)
  - `source_type = "bioasq_snippet"`
  - `source_name = f"PMID:{pmid}"`
  - `section_heading = snippet["beginSection"]` (thường là `"title"` hoặc `"abstract"`)
  - `text = snippet["text"]`
  - `char_start = snippet["offsetInBeginSection"]`, `char_end = snippet["offsetInEndSection"]`
  - Không cần lọc rác (BioASQ snippet đã sạch, là verbatim quote chính thức).

---

## PHẦN 2 — VectorStore (FAISS + embedding y khoa tái dùng)

### Quyết định đã chốt (không cần hỏi lại)

- **Embedding model**: tái dùng đúng model trong `entity_normalizer.py::EmbeddingLinker` — `FremyCompany/BioLORD-2023-C` với fallback `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` nếu BioLORD load lỗi. Dùng `sentence_transformers.SentenceTransformer`, lazy-load giống `EmbeddingLinker._ensure_loaded()`.
- **Lưu ý normalize**: trong `EmbeddingLinker`, lúc `build_index()` KHÔNG gọi `normalize_embeddings=True`, nhưng lúc `link()` thì TỰ normalize cả query và index bằng tay (`/ np.linalg.norm(...)`) trước khi tính dot product. Để dùng FAISS `IndexFlatIP` (cosine similarity qua inner product) đúng cách, **phải normalize vector NGAY LÚC ADD vào index** (không làm thủ công sau như `EmbeddingLinker`) — vì FAISS không tự normalize. Cụ thể: gọi `model.encode(texts, normalize_embeddings=True)` cả lúc build index và lúc query, rồi dùng `faiss.IndexFlatIP`. Đây là điểm khác biệt có chủ đích so với `EmbeddingLinker`, ghi rõ trong code comment để người đọc sau không nhầm là bug.
- **Backend**: FAISS, lưu local file (không cần server).
- **Batch size encode**: theo pattern cũ, `batch_size=256`.

### Interface bắt buộc — `retrieval/vector_store.py`

```python
@dataclass
class RetrievedChunk:
    chunk: VectorChunk
    score: float  # cosine similarity, 0-1

class VectorStore:
    def __init__(self, model_name: str = "FremyCompany/BioLORD-2023-C", index_path: str | None = None):
        ...

    def add_chunks(self, chunks: list[VectorChunk]) -> None:
        """Encode chunks.text theo batch, normalize, add vào FAISS IndexFlatIP.
        Lưu song song một list metadata (chunk_id, source_type, source_name,
        section_heading, char_start, char_end, text) theo đúng thứ tự index
        FAISS — FAISS chỉ lưu vector, không lưu metadata, nên cần map riêng
        (ví dụ list[VectorChunk] song song với index, hoặc dict id->chunk)."""
        ...

    def query(self, text: str, top_k: int = 5, filter_source_type: str | None = None) -> list[RetrievedChunk]:
        """Encode query (normalize), FAISS search, map kết quả về VectorChunk + score.
        filter_source_type: nếu set, lọc theo 'textbook' hoặc 'bioasq_snippet'
        SAU KHI search FAISS (FAISS không filter native) — search top_k * 3,
        rồi filter, rồi cắt lại top_k, để tránh thiếu kết quả khi filter."""
        ...

    def persist(self, path: str) -> None:
        """Lưu faiss index (faiss.write_index) + metadata list (pickle/json) ra path."""
        ...

    def load(self, path: str) -> None:
        """Load lại faiss index + metadata, khôi phục đúng mapping index<->chunk."""
        ...

    def stats(self) -> dict:
        """Trả về {'total_chunks': N, 'by_source_type': {'textbook': N1, 'bioasq_snippet': N2}}"""
        ...
```

### Script ingest — `retrieval/build_vector_store.py`

Chạy 1 lần để build toàn bộ Nhánh B:
1. Gọi `chunk_textbook()` cho cả 4 file textbook → list `VectorChunk`.
2. Gọi `load_bioasq_snippets()` cho file BioASQ JSON (`{"questions": [...]}`) → list `VectorChunk`.
3. Gộp lại, `VectorStore().add_chunks(all_chunks)`.
4. `persist()` ra đường dẫn cố định (ví dụ `data/vector_store/medkg_vectorstore.faiss` + `.meta.json`).
5. In `stats()` để xác nhận số lượng chunk mỗi nguồn trước khi coi là xong.

---

## PHẦN 3 — Flow truy vấn lúc trả lời câu hỏi (Hybrid Retrieval)

### Quyết định đã chốt

Khi có câu hỏi (MedQA hoặc bất kỳ): **luôn query song song cả VectorStore và MEDKG (KG)**, gộp context, không phân biệt nguồn chính/phụ.

### Interface — `retrieval/hybrid_retriever.py`

```python
@dataclass
class HybridContext:
    question: str
    vector_results: list[RetrievedChunk]      # từ VectorStore.query()
    kg_results: list[dict]                     # từ MEDKG lookup (entity match -> subgraph/edges liên quan)
    combined_context_text: str                 # ghép cả 2 thành 1 block text để đưa vào prompt LLM

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, kg_lookup_fn: Callable[[str], list[dict]]):
        """kg_lookup_fn: hàm nhận question (hoặc entity đã linking), trả về
        list edge/node liên quan từ MEDKG. KHÔNG implement kg_lookup_fn trong
        spec này — chỉ định nghĩa interface; việc nối với MEDKG thật (qua
        entity_normalizer.py để link entity trong câu hỏi -> PrimeKG/MEDKG
        node) sẽ làm ở một spec riêng SAU KHI Nhánh A build xong. Tạm thời
        cho phép truyền một stub function trả về [] để code chạy được."""
        ...

    def retrieve(self, question: str, top_k_vector: int = 5) -> HybridContext:
        """
        1. Query VectorStore song song (không filter source_type — lấy cả
           textbook và bioasq_snippet).
        2. Gọi kg_lookup_fn(question) — nếu trả về [] (chưa nối KG thật) thì
           bỏ qua, không lỗi.
        3. Ghép combined_context_text: với mỗi vector_result, format
           '[Nguồn: {source_name}{" - " + section_heading if section_heading
           else ""}] {text}'; với mỗi kg_result, format theo cấu trúc edge
           có sẵn (subject - relation - object [PMID nếu có]).
        4. KHÔNG rerank, KHÔNG dedupe phức tạp ở bản đầu — chỉ nối tuần tự
           vector_results trước, kg_results sau. Để lại comment TODO nếu
           sau này cần rerank theo relevance score thống nhất giữa 2 nguồn
           (vector cosine score và KG không có score cùng thang đo, không
           tự ý quy đổi mà chưa hỏi).
        """
        ...
```

### Việc KHÔNG làm ở Phần 3 (để spec sau xử lý riêng)

- Không implement `kg_lookup_fn` thật (cần Nhánh A build xong + entity linking vào MEDKG node — đây là quyết định kiến trúc riêng, chưa chốt).
- Không làm reranking/dedup giữa 2 nguồn.
- Không tích hợp vào prompt LLM cuối cùng để trả lời MedQA (đó là bước downstream khác).

---

## Ràng buộc chung — KHÔNG làm trong spec này

- `models.py`: CHỈ thêm `BioASQProfile` dataclass (Bước 0.1) — không sửa/xóa field nào của `DiseaseProfile` hay các dataclass khác đã có.
- `disease_profiler.py`: CHỈ thêm `_run_bioasq_mode()` + nhánh điều kiện đầu `run()` (Bước 0.2) — không sửa 8 method cũ (`_resolve_identity`, `_query_omim`, `_check_tier1_sources`, `_check_genereviews`, `_profile_primekg`, `_estimate_literature`, `_generate_strategy`, `_identify_differentials`, `_save_config`).
- `evidence_harvester.py`: CHỈ sửa đúng phạm vi Bước 0.3 (đổi signature `_harvest_from_bioasq_gold()` + parser metadata-only + nhánh điều kiện trong `run()`) — không sửa `_harvest_tier1()`, `_harvest_tier2()`, hay logic Tier-1-search hiện có.
- Không sửa `knowledge_extractor.py`, `quality_controller.py`, `schema_alignment.py`, `temporal_reasoner.py`, `credibility_scorer.py`, `orchestrator.py` — ngoài phạm vi spec này.
- Không áp `EvidenceTier`/`credibility_score` cho textbook hoặc bioasq snippet chunk trong VectorStore (Nhánh B) — đây là corpus retrieval thuần, khác với `SourceDocument` của Nhánh A (Phần 0) là vẫn cần `credibility_score` vì nó đi vào KG thật qua `QualityController`.
- Không cần đảm bảo `sentence-transformers`/`faiss` đã cài — nếu chưa có, dùng `pip install sentence-transformers faiss-cpu --break-system-packages` và ghi rõ trong code nếu import lỗi (theo đúng pattern try/except của `EmbeddingLinker._ensure_loaded()`).

## Bàn giao tổng (checklist)

**Nhánh A:**
- [ ] `models.py` — thêm `BioASQProfile` dataclass (Bước 0.1)
- [ ] `disease_profiler.py` — `_run_bioasq_mode()` + nhánh điều kiện trong `run()` (Bước 0.2), verify với file BioASQ mẫu chứa item Hirschsprung disease
- [ ] `evidence_harvester.py` — `_harvest_from_bioasq_gold(profile: BioASQProfile, ...)` + parser metadata-only (không filter quality) + nhánh điều kiện trong `run()` (Bước 0.3), verify nối tiếp từ `BioASQProfile` ở bước trên

**Nhánh B:**
- [ ] `retrieval/textbook_chunker.py` — chunk 4 file textbook, đã verify chất lượng (Bước 4 Phần 1)
- [ ] `retrieval/bioasq_snippet_loader.py` — load snippet BioASQ thành `VectorChunk`
- [ ] `retrieval/vector_store.py` — `VectorStore` class với FAISS + BioLORD/SapBERT
- [ ] `retrieval/build_vector_store.py` — script ingest end-to-end, in `stats()`
- [ ] `retrieval/hybrid_retriever.py` — `HybridRetriever` với `kg_lookup_fn` stub
- [ ] Chạy thử toàn bộ, xác nhận `VectorStore.query()` trả kết quả hợp lý cho vài câu hỏi mẫu (ví dụ lấy thẳng câu hỏi MedQA mẫu đã có: "A 23-year-old pregnant woman... Which of the following is the best treatment?" → kiểm tra top-5 chunk trả về có liên quan UTI/nitrofurantoin/thai kỳ không)
