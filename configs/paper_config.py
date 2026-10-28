"""Paper configuration overrides for the three-module SRD model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from config import default_config  # noqa: E402


CFG = default_config()
CFG["image_size"] = 1024
CFG["stage_a_epochs"] = 30
CFG["stage_b_epochs"] = 20
CFG["reasoning"]["spectral_enabled"] = True
CFG["reasoning"]["memory_enabled"] = True
CFG["reasoning"]["mrf_enabled"] = True

CFG["reasoning"]["quality"]["enabled"] = False

CFG["manifest_path"] = str(ROOT / "configs" / "manifest.example.json")
CFG["output_dir"] = str(ROOT / "srd_results")
