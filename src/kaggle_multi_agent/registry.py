from pathlib import Path

from kaggle_multi_agent.contracts import BenchmarkResult


class ExperimentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: BenchmarkResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(result.model_dump_json())
            file.write("\n")

    def load(self) -> list[BenchmarkResult]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            return [BenchmarkResult.model_validate_json(line) for line in file if line.strip()]

    def best(self, metric: str = "holdout_rmse") -> BenchmarkResult | None:
        results = self.load()
        if not results:
            return None
        return min(results, key=lambda item: getattr(item, metric))
