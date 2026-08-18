from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", "dist", "node_modules", "playwright-report", "test-results"}


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)


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
        forbidden = {"artifacts", "checkpoints", "outputs"}
        violations = [
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if not is_ignored(path) and path.is_dir() and path.name in forbidden
        ]
        self.assertEqual(violations, [])

    def test_launch_scripts_have_no_machine_paths(self) -> None:
        violations: list[Path] = []
        machine_prefixes = ("/" + "mnt/", "/" + "disk/")
        for path in ROOT.rglob("*.sh"):
            if is_ignored(path):
                continue
            content = path.read_text(encoding="utf-8")
            if any(prefix in content for prefix in machine_prefixes):
                violations.append(path.relative_to(ROOT))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
