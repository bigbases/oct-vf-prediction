"""
B 실험 + baseline(CNN repro, XGB) 집계 및 비교표 출력.

사용법:
  python scripts/summarize_b_experiments.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44, 45, 46]


def load_run(path: Path):
    if not path.is_file():
        return None
    return json.load(open(path, encoding='utf-8'))


def agg(name, pattern_fn):
    rmses, maes, n = [], [], 0
    for s in SEEDS:
        r = load_run(ROOT / pattern_fn(s))
        if r is None:
            continue
        rm = r.get('best_holdout_test_rmse')
        ma = r.get('best_holdout_test_mae')
        if rm is not None:
            rmses.append(rm)
            maes.append(ma)
            n += 1
    if not rmses:
        return {'name': name, 'n': 0}
    return {
        'name': name,
        'n': n,
        'rmse_mean': statistics.mean(rmses),
        'rmse_std': statistics.pstdev(rmses) if len(rmses) > 1 else 0.0,
        'mae_mean': statistics.mean(maes),
        'mae_std': statistics.pstdev(maes) if len(maes) > 1 else 0.0,
    }


def main():
    rows = [
        agg('B0 CNN baseline (Inception, thickness)',
            lambda s: f'runs/repro_90d_imgonly_flip_papertrack_lr1e4_p100_s{s}/results.json'),
        agg('B0 CNN baseline (bbcmp inception_v3)',
            lambda s: f'runs/bbcmp_90d_inception_v3_imgonly_s{s}/results.json'),
        agg('B1 freeze backbone',
            lambda s: f'runs/bexp_90d_b1_freeze_s{s}/results.json'),
        agg('B3 thickness+deviation',
            lambda s: f'runs/bexp_90d_b3_deviation_s{s}/results.json'),
    ]
    xgb = load_run(ROOT / 'baseline_xgb_results.json')
    if xgb and '90d_excl_empty_flip' in xgb:
        d = xgb['90d_excl_empty_flip']
        rows.append({
            'name': 'XGB tabular (5-fold CV mean)',
            'n': 5,
            'rmse_mean': None,
            'rmse_std': None,
            'mae_mean': d['cv_mean'],
            'mae_std': d['cv_std'],
            'note': f"test MAE {d['test_mae']:.3f} (fixed cv_fold test, split differs from repro)",
        })

    print('\n=== 90d 성능 비교 (holdout test, 5-seed where available) ===')
    print(f"{'설정':<42s} {'n':>2s}  {'test RMSE':>18s}  {'test MAE':>18s}")
    print('-' * 85)
    for r in rows:
        if r['n'] == 0:
            print(f"{r['name']:<42s}  --  (결과 없음)")
            continue
        rm = (f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}"
              if r.get('rmse_mean') is not None else '  (CV only)     ')
        ma = f"{r['mae_mean']:.3f} ± {r['mae_std']:.3f}"
        print(f"{r['name']:<42s} {r['n']:>2d}  {rm:>18s}  {ma:>18s}")
        if r.get('note'):
            print(f"    ※ {r['note']}")

    out = ROOT / 'runs/bexp_90d_comparison.json'
    json.dump(rows, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
