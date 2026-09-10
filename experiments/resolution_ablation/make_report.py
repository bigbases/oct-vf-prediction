#!/usr/bin/env python3
"""해상도 절제 파일럿 결과 집계 → pilot_fold0.json + pilot_fold0.md

각 n 의 run 디렉터리 results.json 에서 fold 0 의 best-epoch val pooled RMSE/MAE 를 읽고,
val_preds npz 로 같은 값을 독립 재계산해 교차검증한다.
n=161 은 정본 runs/phasec_b0_inception_resnet_v2_5fold/results.json 의 fold 0 과 대조한다.
env: hvf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / 'experiments' / 'resolution_ablation'
CANON = REPO / 'runs' / 'phasec_b0_inception_resnet_v2_5fold' / 'results.json'
NS = [161, 32, 16, 8, 4, 2]
TOL = 1e-9   # 비트 재현 기대치


def pooled_from_npz(p: Path):
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    pred, lab, m = z['pred'], z['labels'], z['mask'].astype(bool)
    d = (pred - lab)[m]
    return {'rmse': float(np.sqrt((d ** 2).mean())), 'mae': float(np.abs(d).mean()),
            'n_points': int(m.sum()), 'n_eyes': int(pred.shape[0])}


def main():
    canon = json.load(open(CANON, encoding='utf-8'))
    c0 = next(r for r in canon['results'] if r['fold'] == 0)

    rows = []
    for n in NS:
        rp = HERE / 'runs' / f'n{n}' / 'results.json'
        if not rp.exists():
            rows.append({'blocks_per_side': n, 'status': 'not_run'})
            continue
        r = json.load(open(rp, encoding='utf-8'))
        f0 = next(x for x in r['results'] if x['fold'] == 0)
        rec = {
            'blocks_per_side': n,
            'blocks_per_panel': n * n,
            'best_epoch': f0['best_epoch'],
            'val_pooled_rmse': f0['best_val_rmse'],
            'val_pooled_mae': f0['best_val_mae'],
            'epochs_run': len(f0['history']),
            'stopped_by': 'epoch_cap' if len(f0['history']) >= r['args']['epochs'] else 'patience',
            'n_train': f0['n_train'], 'n_val': f0['n_val'],
            'recomputed_from_npz': pooled_from_npz(HERE / 'runs' / f'n{n}' / 'val_preds_fold0.npz'),
            'status': 'ok',
        }
        rows.append(rec)

    gate = {
        'canonical_source': str(CANON.relative_to(REPO)),
        'canonical_val_pooled_rmse': c0['best_val_rmse'],
        'canonical_val_pooled_mae': c0['best_val_mae'],
        'canonical_best_epoch': c0['best_epoch'],
    }
    n161 = next((r for r in rows if r['blocks_per_side'] == 161), None)
    if n161 and n161['status'] == 'ok':
        d_rmse = n161['val_pooled_rmse'] - c0['best_val_rmse']
        d_mae = n161['val_pooled_mae'] - c0['best_val_mae']
        gate.update({
            'replicate_val_pooled_rmse': n161['val_pooled_rmse'],
            'replicate_val_pooled_mae': n161['val_pooled_mae'],
            'replicate_best_epoch': n161['best_epoch'],
            'delta_rmse': d_rmse, 'delta_mae': d_mae,
            'passed': bool(abs(d_rmse) <= TOL and abs(d_mae) <= TOL
                           and n161['best_epoch'] == c0['best_epoch']),
        })
    else:
        gate['passed'] = None

    base = next((r for r in rows if r['blocks_per_side'] == 161 and r['status'] == 'ok'), None)
    for r in rows:
        if r['status'] == 'ok' and base:
            r['delta_rmse_vs_original'] = r['val_pooled_rmse'] - base['val_pooled_rmse']
            r['delta_mae_vs_original'] = r['val_pooled_mae'] - base['val_pooled_mae']

    out = {
        'what': 'spatial resolution ablation pilot — 렌더된 두께맵을 패널 내 n x n 블록 평균 '
                '계단 함수로 거칠게 만든 뒤 같은 CNN 에 넣는다',
        'design': {
            'input': '161x322 (GCA 161x161 | RNFL 161x161) thickness concat, use_deviation=False',
            'block_boundaries': 'round(linspace(0,161,n+1)) — 패널 내부에서만, 두 패널 미혼합',
            'normalization_order': 'Resize -> (train: ColorJitter) -> ToTensor -> BlockAverage -> ImageNet Normalize',
            'backbone': 'inception_resnet_v2 (ImageNet pretrained)',
            'fixed': {'seed': 42, 'fold': 0, 'epochs': 300, 'optimizer': 'rmsprop',
                      'lr': 1e-4, 'weight_decay': 1e-5, 'momentum': 0.9, 'batch_size': 64,
                      'patience': 100, 'early_stop_metric': 'mae',
                      'csv': 'ml_final_90d_excl_empty_flip.csv'},
            'metric': 'fold 0 검증 폴드, best epoch 의 point-pooled masked RMSE / MAE (dB)',
            'note_colorjitter': 'train transform 의 ColorJitter(brightness 0.10, contrast 0.10) 는 '
                                '정본 5백본 학습과 동일하게 유지했다 (제거하면 정본 재현 불가).',
        },
        'gate_n161_reproduces_canonical': gate,
        'curve': rows,
    }
    (HERE / 'pilot_fold0.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    L = []
    L.append('# 공간 해상도 절제 파일럿 (fold 0)\n')
    L.append('렌더된 두께맵을 패널 안에서 `n x n` 블록 평균 계단 함수로 거칠게 만든 뒤 같은 CNN 에 넣는다.')
    L.append('모델족·백본·하이퍼파라미터·폴드가 전부 고정되므로 변하는 것은 입력의 공간 표현뿐이다.\n')
    L.append('- 입력: `161x322` (GCA `161x161` | RNFL `161x161`) thickness concat, `use_deviation=False`')
    L.append('- 블록 경계: `round(linspace(0,161,n+1))` — 패널 내부에서만 잡고 두 패널을 섞지 않는다')
    L.append('- 순서: `Resize -> (train: ColorJitter) -> ToTensor -> BlockAverage -> ImageNet Normalize`')
    L.append('- 고정: inception_resnet_v2, seed 42, fold 0, 300 epoch, RMSprop lr 1e-4, wd 1e-5, '
             'momentum 0.9, batch 64, patience 100, early stop = val MAE')
    L.append('- 지표: fold 0 검증 폴드, best epoch 의 point-pooled masked RMSE / MAE (dB)\n')
    L.append('## 게이트: n=161(원본) 이 정본 fold 0 을 재현하는가\n')
    if gate.get('passed') is None:
        L.append('아직 실행되지 않음.\n')
    else:
        L.append(f"| | best epoch | pooled RMSE | pooled MAE |")
        L.append('|---|---|---|---|')
        L.append(f"| 정본 `runs/phasec_b0_inception_resnet_v2_5fold` fold 0 | "
                 f"{gate['canonical_best_epoch']} | {gate['canonical_val_pooled_rmse']:.6f} | "
                 f"{gate['canonical_val_pooled_mae']:.6f} |")
        L.append(f"| 본 파일럿 n=161 | {gate['replicate_best_epoch']} | "
                 f"{gate['replicate_val_pooled_rmse']:.6f} | {gate['replicate_val_pooled_mae']:.6f} |")
        L.append(f"| 차이 | — | {gate['delta_rmse']:+.2e} | {gate['delta_mae']:+.2e} |\n")
        L.append(f"**게이트 {'통과' if gate['passed'] else '실패'}**"
                 + ('' if gate['passed'] else ' — 어긋났으므로 나머지 n 은 해석하지 말 것.') + '\n')
    L.append('## 곡선 (n 오름차순)\n')
    L.append('| n (한 변당 블록) | 패널당 블록 | best epoch | 종료 사유 | pooled RMSE (dB) | pooled MAE (dB) | ΔRMSE vs 원본 | ΔMAE vs 원본 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in sorted([x for x in rows if x['status'] == 'ok'], key=lambda x: x['blocks_per_side']):
        n = r['blocks_per_side']
        tag = ' (원본)' if n == 161 else (' ≈ 인쇄 파라미터 조밀도' if n == 4 else '')
        cap = '300 epoch 상한' if r['stopped_by'] == 'epoch_cap' else f"patience ({r['epochs_run']}ep)"
        L.append(f"| {n}{tag} | {r['blocks_per_panel']} | {r['best_epoch']} | {cap} | "
                 f"{r['val_pooled_rmse']:.3f} | {r['val_pooled_mae']:.3f} | "
                 f"{r.get('delta_rmse_vs_original', float('nan')):+.3f} | "
                 f"{r.get('delta_mae_vs_original', float('nan')):+.3f} |")
    missing = [x['blocks_per_side'] for x in rows if x['status'] != 'ok']
    if missing:
        L.append(f"\n미실행: n = {missing}")
    capped = [x['blocks_per_side'] for x in rows if x.get('stopped_by') == 'epoch_cap']
    if capped:
        L.append(f"\n**주의:** n = {capped} 은 patience 가 아니라 300 epoch 상한에 걸려 끝났다 "
                 "(best epoch 이 상한 가까이). 정본과 같은 고정 예산 비교이지만, 거친 입력일수록 "
                 "수렴이 느려 예산에 잘리는 쪽으로 불리하게 편향될 수 있다 — 해당 n 의 값은 "
                 "'이 예산 안에서의 성능'으로 읽어야 한다.")
    L.append('\n산출물: `experiments/resolution_ablation/runs/n<N>/` (results.json, train.log, '
             'best_fold0.pt, val_preds_fold0.npz). 정본 `runs/` 와 원고는 건드리지 않았다.')
    (HERE / 'pilot_fold0.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"\nwrote {HERE/'pilot_fold0.json'}\nwrote {HERE/'pilot_fold0.md'}")


if __name__ == '__main__':
    main()
