from openai import OpenAI
from google import genai 
import os
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer

_PRIMARY_MODEL = "FremyCompany/BioLORD-2023-C"
_FALLBACK_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

load_dotenv()

class Embeddings:
    def __init__(self, type: str, model_name: str = _PRIMARY_MODEL) -> None:
        self._model_name = model_name
        self._type = type
        self._model = None
        
        if self._type == "openai":
            self._model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self._type == "genai":
            self._model = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            try:
                self._model = SentenceTransformer(self._model_name, device="cpu")
            except ImportError:
                logging.error("sentence-transformers not installed. "
                              "Run: pip install sentence-transformers")
            except Exception as exc:
                logging.warning("Primary model failed (%s), trying SapBERT fallback", exc)
                try:
                    self._model = SentenceTransformer(_FALLBACK_MODEL, device="cpu")
                    self._model_name = _FALLBACK_MODEL
                    logging.info("SapBERT fallback loaded (device=cpu)")
                except Exception as exc2:
                    logging.error("No embedding model available: %s", exc2)
            

    def encode(self, doc):
        if self._type == "openai":
            response = self._model.embeddings.create(input=doc, model=self._model_name)
            return response.data[0].embedding
        elif self._type == "gemini":
            response = self._model.models.embed_content(contents=doc, model=self._model_name)
            return response.embeddings
        else:
            return self._model.encode(doc)
    
    def encode_batch(self, docs):
        return [self.encode(doc) for doc in docs]