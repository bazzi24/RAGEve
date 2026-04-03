# RAGEve Evaluation Suite

# Re-export run_all so run.py can call it via importlib.util.
from test.benchmark.evaluation.suite import run_full_evaluation as run_all

__all__ = ["run_all"]
