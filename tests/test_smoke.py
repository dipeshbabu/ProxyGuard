import unittest
from pathlib import Path

from risk_models.configs import clone_experiment_config, get_benchmark_model_configs, get_dataset_config, get_default_experiment_config
from risk_models.cv_runner import run_benchmark


class SmokeTest(unittest.TestCase):
    def test_debug_style_smoke(self) -> None:
        output_root = Path("outputs") / "test_smoke"
        exp_cfg = clone_experiment_config(
            get_default_experiment_config(),
            n_repeats=1,
            output_root=str(output_root),
            save_reliability=False,
            save_shap=False,
        )
        result = run_benchmark(
            get_dataset_config("german_credit"),
            get_benchmark_model_configs()[:2],
            exp_cfg,
            mode="test_smoke",
        )
        self.assertFalse(result["aggregate_metrics"].empty)
        self.assertTrue({"Model", "AUC"}.issubset(result["aggregate_metrics"].columns))


if __name__ == "__main__":
    unittest.main()
