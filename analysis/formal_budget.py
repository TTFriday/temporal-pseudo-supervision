"""Formal S2 budget-sweep analysis (frozen analysis plan, splits/FORMAL_S2_PREREGISTRATION.md).

Reads the three formal result files plus the reused 10% column (v3+v4),
checks that every column shares source/manifest/split hashes, and reports the
paired comparisons and budget curve. Descriptive only: no significance claims,
no method or threshold changes after seeing the results.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'results'
COLUMNS = {
    0.05: FORMAL / 'pilot_budget05' / 'results_final.json',
    0.20: FORMAL / 'pilot_budget20' / 'results_final.json',
    1.00: FORMAL / 'pilot_full100' / 'results_final.json',
}
TEN_PCT_DIRS = [
    ROOT / 'results' / 'pilot_validated_20261025_v3',
    ROOT / 'results' / 'pilot_validated_20261026_v4',
]
SEEDS3 = [17, 42, 73]
SEEDS6 = [17, 42, 73, 101, 202, 303]
METHODS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
PRIMARY = ('M3', 'M1')
COMPARISONS = [('M3', 'M1'), ('M4', 'M3'), ('M4', 'M1'), ('M2', 'M1')]


def ap_by_column(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return {(r['method'], r['seed']): r['student']['best_validation_AP']
            for r in data['runs']}


def provenance_checks():
    paths = [p.parent / 'provenance.json' for p in COLUMNS.values()] + \
            [d / 'provenance.json' for d in TEN_PCT_DIRS]
    provs = {str(p.parent): json.loads(p.read_text(encoding='utf-8')) for p in paths}
    first = next(iter(provs.values()))
    checks = {
        'source_sha256_identical': all(p['source_sha256'] == first['source_sha256'] for p in provs.values()),
        'manifest_sha256_identical': all(p['manifest_sha256'] == first['manifest_sha256'] for p in provs.values()),
        'split_sha256_identical': all(p['split_sha256'] == first['split_sha256'] for p in provs.values()),
        'audit_sha256_identical': all(p['config'].get('audit_sha256') == first['config'].get('audit_sha256')
                                      for p in provs.values()),
        'budget_ratios': {Path(k).name: p['config']['budget_ratio'] for k, p in provs.items()},
    }
    return checks


def summary(values):
    n = len(values)
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {'mean': mean, 'sd': sd, 'min': min(values), 'max': max(values),
            'positive': sum(1 for v in values if v > 0), 'n': n}


def paired(table, method_a, method_b, seeds):
    diffs = []
    for s in seeds:
        a = table.get((method_a, s))
        b = table.get((method_b, s))
        if a is None or b is None:
            raise KeyError(f'missing {method_a}/{method_b} seed {s}')
        diffs.append({'seed': s, 'a': a, 'b': b, 'diff_pp': (a - b) * 100})
    return {'comparison': f'{method_a}-{method_b}',
            'per_seed_pp': [round(d['diff_pp'], 3) for d in diffs],
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary(
                [d['diff_pp'] for d in diffs]).items()}}


def main():
    ten = {}
    for d in TEN_PCT_DIRS:
        ten.update(ap_by_column(d / 'results_final.json'))
    tables = {0.10: ten}
    for ratio, path in COLUMNS.items():
        tables[ratio] = ap_by_column(path)

    result = {'provenance_checks': provenance_checks(), 'budgets': {}}
    for ratio in sorted(tables):
        table = tables[ratio]
        method_means = {}
        for m in METHODS:
            vals = [table[(m, s)] for s in SEEDS6 if (m, s) in table]
            vals3 = [table[(m, s)] for s in SEEDS3 if (m, s) in table]
            if vals3:
                method_means[m] = {
                    'seeds3_mean': round(sum(vals3) / len(vals3) * 100, 3),
                    'seeds3_per_seed': {str(s): round(table[(m, s)] * 100, 3)
                                        for s in SEEDS3 if (m, s) in table},
                }
                if len(vals) > len(vals3):
                    method_means[m]['seeds6_mean'] = round(sum(vals) / len(vals) * 100, 3)
        has_all = all((m, s) in table for m in ('M1', 'M2', 'M3', 'M4') for s in SEEDS3)
        comparisons = [paired(table, a, b, SEEDS3) for a, b in COMPARISONS] if has_all else None
        result['budgets'][f'{ratio:.2f}'] = {
            'methods': method_means,
            'comparisons_seeds3': comparisons,
        }

    curve = []
    for ratio in (0.05, 0.10, 0.20):
        table = tables[ratio]
        column_dir = COLUMNS[ratio].parent if ratio in COLUMNS else TEN_PCT_DIRS[0]
        labels = json.loads((column_dir / 'protocol.json').read_text(encoding='utf-8'))['budget']
        diffs = [(table[PRIMARY[0], s] - table[PRIMARY[1], s]) * 100 for s in SEEDS3]
        curve.append({'budget_ratio': ratio,
                      'budget_labels': labels,
                      'M1_mean': round(sum(table[('M1', s)] for s in SEEDS3) / 3 * 100, 3),
                      'M3_mean': round(sum(table[('M3', s)] for s in SEEDS3) / 3 * 100, 3),
                      'm3_minus_m1_mean_pp': round(sum(diffs) / 3, 3),
                      'm3_minus_m1_per_seed_pp': [round(d, 3) for d in diffs],
                      'm3_minus_m1_positive': sum(1 for d in diffs if d > 0)})
    result['budget_curve_m3_minus_m1'] = curve

    full = tables[1.0]
    result['full_label_reference'] = {
        'per_seed_AP': {str(s): round(full[('M1', s)] * 100, 3) for s in SEEDS3},
        'mean_AP': round(sum(full[('M1', s)] for s in SEEDS3) / 3 * 100, 3),
        'note': 'Context only; not a theoretical upper bound of the low-label methods',
    }

    out = FORMAL / 'formal_budget.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
