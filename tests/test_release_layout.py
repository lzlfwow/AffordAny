from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTest(unittest.TestCase):
    def test_canonical_components_exist(self) -> None:
        expected = (
            ROOT / "research" / "pipeline" / "module_real_lvis_runner",
            ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5" / "src",
            ROOT / "models" / "affordance_decoder_selftraining_v1" / "src",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_dir())

    def test_experiment_artifacts_are_excluded(self) -> None:
        forbidden = {"__pycache__", "artifacts", "checkpoints", "outputs"}
        violations = [
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if path.is_dir() and path.name in forbidden
        ]
        self.assertEqual(violations, [])

    def test_launch_scripts_have_no_machine_paths(self) -> None:
        violations: list[Path] = []
        machine_prefixes = ("/" + "mnt/", "/" + "disk/")
        for path in ROOT.rglob("*.sh"):
            content = path.read_text(encoding="utf-8")
            if any(prefix in content for prefix in machine_prefixes):
                violations.append(path.relative_to(ROOT))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
