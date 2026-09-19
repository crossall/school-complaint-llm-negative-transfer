from __future__ import annotations

from .analyze_experiment1 import run as run_experiment1
from .analyze_multiseed import run as run_multiseed
from .analyze_rag import run as run_rag
from .analyze_retrieval import run as run_retrieval
from .common import RESULTS, write_json


def main() -> None:
    results = {
        "experiment1": run_experiment1(),
        "public_multiseed": run_multiseed(),
        "rag": run_rag(),
        "retrieval": run_retrieval(),
    }
    write_json(RESULTS / "all_results.json", results)
    print("Reproduced all analyses in results/.")


if __name__ == "__main__":
    main()
