"""ChronoMedKG: temporally-grounded biomedical knowledge graph and benchmark.

Quickstart:

    >>> import chronomedkg as cmkg
    >>> for t in cmkg.load_triples():       # streams from Zenodo
    ...     print(t["source_name"], "--[", t["relation"], "]-->", t["target_name"])
    ...
    >>> qa = cmkg.load_benchmark()          # ChronoTQA, 3,341 questions
    >>> cases = cmkg.load_pmc_cases()       # 31 PMC diagnostic-odyssey cases

Files are streamed from the Zenodo deposit
(https://doi.org/10.5281/zenodo.19697543) on first call and cached locally
in ``~/.cache/chronomedkg/`` (override with the ``CHRONOMEDKG_CACHE``
environment variable).
"""

from chronomedkg.loader import (
    cache_dir,
    load_benchmark,
    load_consensus,
    load_pmc_cases,
    load_triples,
)

__version__ = "0.0.1"

__all__ = [
    "cache_dir",
    "load_benchmark",
    "load_consensus",
    "load_pmc_cases",
    "load_triples",
]
