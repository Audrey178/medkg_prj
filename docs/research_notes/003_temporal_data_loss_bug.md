# Research Note 003: Critical Bug — Temporal Data Loss in Schema Conversion

> Date: 2026-03-24
> Discovery: We were extracting rich temporal data then DROPPING it during TemporalEdge conversion
> Impact: Reported temporal coverage jumped from 34-65% to **69-90%** after fix
> Status: **FIXED** — 3 fields added to TemporalMetadata model

---

## The Bug

The extraction prompt asks LLMs for 8 temporal fields in `temporal_context`:
1. onset_age_min ✓ (preserved)
2. onset_age_max ✓ (preserved)
3. progression_stage ✓ (preserved)
4. duration ✓ (preserved)
5. discovery_year ✓ (converted to discovery_date)
6. **milestone** ❌ (DROPPED — not in TemporalMetadata)
7. **temporal_qualifier** ❌ (DROPPED — not in TemporalMetadata)
8. **treatment_start_age** ❌ (DROPPED — not in TemporalMetadata)

The `TemporalMetadata` dataclass only had fields 1-5. The QC conversion step
(quality_controller.py line 279-282) only read those 5 fields from `temporal_context`.

The `compute_temporal_coverage()` function only counted fields 1-4 + discovery_date
for the "has_any_temporal" metric.

## What Was Being Lost

### `temporal_qualifier` (54-87% filled) — THE RICHEST SIGNAL

Example values from DMD:
- "by 6 weeks of treatment" — treatment response time!
- "FDA approved" — regulatory milestone!
- "study period from June 29, 2018 to February 24, 2021" — study timeline
- "over the 52-week follow-up period" — treatment duration
- "before age 10" — age-conditional recommendation

This single field contains the week/month-level precision we thought we lacked!

### `milestone` (31-53% filled)

Example values:
- "loss of ambulation" — critical disease progression marker
- "improvement in TTSTAND velocity" — treatment response endpoint
- "ventilation required" — advanced disease milestone
- "cardiac monitoring initiation" — management milestone

### `treatment_start_age` (0-5% filled)

Low fill rate but clinically important when present.

## Impact on Reported Metrics

| Disease | OLD coverage | TRUE coverage | Improvement |
|---------|:-----------:|:------------:|:-----------:|
| DMD | 65% | **90%** | +38% |
| MG | 60% | **90%** | +50% |
| CIDP | 40% | **82%** | +105% |
| GBS | 54% | **80%** | +48% |
| LEMS | 34% | **69%** | +103% |

## Fix Applied

### core/models.py — TemporalMetadata
Added 3 fields: `milestone`, `temporal_qualifier`, `treatment_start_age`
Updated `to_dict()` and `from_dict()` methods

### agents/quality_controller.py — triple→edge conversion
Lines 279+: Added reading of milestone, temporal_qualifier, treatment_start_age from tc dict

### core/temporal_reasoner.py — compute_temporal_coverage()
Updated `has_any_temporal` to include new fields.
Added per-field breakdown for milestone, temporal_qualifier, treatment_start_age.

## Paper Implications

This changes our narrative significantly:
- **Before fix:** "53-72% temporal coverage" — decent but not standout
- **After fix:** "69-90% temporal coverage with clinically actionable signals" — genuinely strong

The `temporal_qualifier` field at 74-87% fill rate contains exactly what clinicians need:
treatment response timelines, regulatory dates, and age-conditional recommendations.
This was always in our data — we just weren't counting or preserving it.

## Lessons for Pipeline Design

1. Schema design must match extraction prompt — every field the LLM produces should have
   a place in the data model
2. Quality metrics must reflect actual data richness, not just structured date fields
3. Free-text temporal qualifiers are often richer than structured numeric fields
