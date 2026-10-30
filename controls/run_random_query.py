"""Random-query control at the 10% label budget (see splits/RANDOM_QUERY_CONTROL_PREREGISTRATION.md).

R1 = random query selection, no propagation (supervision-only control for M1).
R2 = random query selection, basic SAM2 propagation (control for M3).

The two methods are added to the pipeline's method table in memory only.
Files under training/, selection/, evaluation/ and propagation/ are untouched,
so the provenance hashes recorded by the pipeline are identical to the reused
10% column. The runner hash and the runtime extension are recorded in
control_manifest.json next to the results.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import pipeline  # noqa: E402

CONTROL_METHODS = {'R1': ('random', None), 'R2': ('random', 'basic')}
CONFIG = ROOT / 'configs' / 'validated_pilot.json'
OUTPUT = ROOT / 'results' / 'random_query_control'
SEEDS = [17, 42, 73]


def main():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    pipeline.METHODS.update(CONTROL_METHODS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'purpose': 'random-query control at the 10% label budget',
        'methods': {name: list(parts) for name, parts in CONTROL_METHODS.items()},
        'selector': 'random',
        'selector_seed': 42,
        'training_seeds': SEEDS,
        'config_path': str(CONFIG),
        'runner': str(Path(__file__).resolve()),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note': 'Methods added to training.pipeline.METHODS at runtime; '
                'pipeline source files are unmodified.',
    }
    (OUTPUT / 'control_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    result = pipeline.run_experiment(config, str(OUTPUT), 'pilot',
                                     list(CONTROL_METHODS), SEEDS)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
