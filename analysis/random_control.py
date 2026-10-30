"""Random-query control analysis (frozen plan, splits/RANDOM_QUERY_CONTROL_PREREGISTRATION.md).

Compares the control column (R1/R2, random query at 10%) with the reused
10% column (M1/M3, time-coverage query) over seeds 17/42/73. Descriptive only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / 'results' / 'random_query_control'
TEN_PCT_DIRS = [ROOT / 'results' / 'pilot_validated_20261025_v3',
                ROOT / 'results' / 'pilot_validated_20261026_v4']
SEEDS = [17, 42, 73]


def ap_table(path):
    runs = json.loads(path.read_text(encoding='utf-8'))['runs']
    return {(r['method'], r['seed']): r['student']['best_validation_AP'] for r in runs}


def comparison(table, a, b):
    diffs = [(table[(a, s)] - table[(b, s)]) * 100 for s in SEEDS]
    mean = sum(diffs) / len(diffs)
    sd = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    return {'comparison': f'{a}-{b}', 'per_seed_pp': [round(d, 3) for d in diffs],
            'mean_pp': round(mean, 3), 'sd_pp': round(sd, 3),
            'positive': sum(1 for d in diffs if d > 0), 'n': len(diffs)}


def reading(comp):
    if abs(comp['mean_pp']) <= 0.5:
        return 'no clear separation (|mean| <= 0.5 pp)'
    if comp['mean_pp'] > 1.0 and comp['positive'] == comp['n']:
        return 'first method above second by > 1.0 pp with consistent sign'
    if comp['mean_pp'] < -1.0 and comp['positive'] == 0:
        return 'second method above first by > 1.0 pp with consistent sign'
    return 'no clear separation under the pre-registered rule'


def main():
    prov = json.loads((CONTROL_DIR / 'provenance.json').read_text(encoding='utf-8'))
    ref = json.loads((TEN_PCT_DIRS[0] / 'provenance.json').read_text(encoding='utf-8'))
    checks = {
        'source_sha256_identical': prov['source_sha256'] == ref['source_sha256'],
        'manifest_sha256_identical': prov['manifest_sha256'] == ref['manifest_sha256'],
        'split_sha256_identical': prov['split_sha256'] == ref['split_sha256'],
        'audit_sha256_identical': prov['config'].get('audit_sha256') == ref['config'].get('audit_sha256'),
        'budget_ratio': prov['config']['budget_ratio'],
        'control_manifest': json.loads((CONTROL_DIR / 'control_manifest.json').read_text(encoding='utf-8')),
    }

    table = {}
    for d in TEN_PCT_DIRS:
        table.update(ap_table(d / 'results_final.json'))
    table.update(ap_table(CONTROL_DIR / 'results_final.json'))

    methods = {m: {'per_seed_AP': {str(s): round(table[(m, s)] * 100, 3) for s in SEEDS},
                   'mean_AP': round(sum(table[(m, s)] for s in SEEDS) / len(SEEDS) * 100, 3)}
               for m in ('M1', 'M3', 'R1', 'R2')}

    primary = [comparison(table, 'M3', 'R2'), comparison(table, 'M1', 'R1')]
    secondary = [comparison(table, 'M3', 'M1'), comparison(table, 'R2', 'R1')]
    for comp in primary:
        comp['reading'] = reading(comp)

    ten_pct_by_seed = {17: TEN_PCT_DIRS[0], 42: TEN_PCT_DIRS[0], 73: TEN_PCT_DIRS[1]}
    overlap = {}
    for seed in SEEDS:
        m3 = json.loads((ten_pct_by_seed[seed] / f'seed_{seed}' / 'time_coverage' /
                         'query.json').read_text(encoding='utf-8'))['selected_keys']
        r2 = json.loads((CONTROL_DIR / f'seed_{seed}' / 'random' /
                         'query.json').read_text(encoding='utf-8'))['selected_keys']
        a = {tuple(k) for k in m3}
        b = {tuple(k) for k in r2}
        overlap[f'seed_{seed}'] = {'shared': len(a & b), 'M3_only': len(a - b),
                                   'random_only': len(b - a)}

    result = {'provenance_checks': checks, 'methods': methods,
              'primary_comparisons': primary, 'secondary_comparisons': secondary,
              'query_overlap': overlap,
              'note': 'Descriptive control; reading rules fixed in the pre-registration.'}
    out = CONTROL_DIR / 'random_control.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
