import sys
import json
import hashlib
import copy
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config import load_config, require

# --- CONFIGURATION ---
# Ensure these match your actual file/folder names
CFG_FILE = "config_rev2.yaml"      
RESULTS_DIR = "results_rev3"  
# ---------------------

cfg_path = project_root / CFG_FILE
if not cfg_path.exists():
    print(f"❌ Error: {cfg_path} not found!")
    sys.exit(1)

cfg = load_config(cfg_path)
effective = copy.deepcopy(cfg)

def _path_from_root(project_root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()

results = _path_from_root(project_root, RESULTS_DIR)
run_tag = results.name

# Reconstruct the exact paths dictionary the pipeline expects
for key in ("raw_dir", "processed_dir", "models_dir", "configs_dir", "logs_dir", "locks_dir"):
    effective["paths"][key] = str(_path_from_root(project_root, require(cfg, f"paths.{key}")))

if "nazario" in effective:
    for key in ("main_dir", "external_dir"):
        effective["nazario"][key] = str(_path_from_root(project_root, require(cfg, f"nazario.{key}")))

effective["paths"]["results_dir"] = str(results)
for key in ("processed_dir", "models_dir"):
    effective["paths"][key] = str(Path(effective["paths"][key]) / run_tag)

# Build the exact owner dictionary
owner = {
    "results_dir": str(results),
    "run_tag": run_tag,
    "config_digest": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
    "paths": effective["paths"]
}

# The 3 locations where run_context.json must exist
owner_paths = [
    results / "run_context.json",
    Path(effective["paths"]["processed_dir"]) / "run_context.json",
    Path(effective["paths"]["models_dir"]) / "run_context.json"
]

# Forcefully write the context files, ignoring existing data
for path in owner_paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(owner, handle, indent=2)
    print(f"✅ Wrote {path}")

print("\n✅ Success! All run_context.json files have been forcefully regenerated.")