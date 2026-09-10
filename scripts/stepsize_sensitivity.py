#!/usr/bin/env python3
"""타깃 축 정렬 민감도 분석 — Hasan 2025 'TS normalisation' 축으로 재계산.

Hasan 2025는 14 dB 미만 타깃을 14로 클램프한 좌표계("normalised TS")와
raw TS 두 축을 모두 보고한다. 본 연구는 raw 축으로만 보고해 왔으므로,
동일 클램프 하에서 우리 MAE가 어디까지 내려가는지 실측한다.

■ 클램프를 '어디에' 거느냐에 따라 숫자가 갈린다. 세 축을 구분한다:

  (1) clamp_both     예측·라벨 모두 사후 클램프. 학습 후 예측을 눌러주는 조작.
  (2) clamp_label    라벨만 클램프, 예측은 원본 그대로. 모델을 손대지 않음.
  (3) clamp_retrain  14로 정규화한 타깃으로 재학습한 뒤 그 공간에서 집계.
                     ← Hasan 프로토콜과 유일하게 동등한 축.

  Hasan은 정규화된 타깃으로 '학습'했으므로 (3)만이 진짜 대응이다.
  (1)을 단일 대표값으로 쓰면 리뷰어가 정확히 그 지점을 찌른다.

  ⚠️ (1)을 '낙관 상한', (2)를 '보수 하한'으로 부르지 말 것 — 2026-08-08 실측 반증.
     XGB OOF 14 dB 축에서 both 4.363 / label 5.177 / retrain 4.180 으로
     재학습이 both 보다도 좋았다. 즉 (3)은 (1)(2) 구간 안에 있지 않으므로
     '구간으로 보고' 전략은 성립하지 않는다.
     이유: 재학습 모델은 예측의 99.3%를 14 이상으로 내보내(raw 83.9%)
     저감도 20.5%를 사실상 포기하는 대신 정상 79.5% 구간 정확도를 크게 얻는다.
     (라벨<14 MAE both 3.539 → retrain 6.217, 라벨≥14 MAE 4.575 → 3.656)
     → 세 축은 서로 다른 '조작'이지 하나의 값을 감싸는 구간이 아니다.
        각 축을 그 조작이 무엇인지와 함께 개별 보고할 것.

주의:
  - raw 축이 정본이다. 클램프 축은 §5.3 민감도 행으로만 쓴다.
    (라벨의 20% 이상이 상수로 눌리는 좌표계를 헤드라인에 올리지 않는다.)
  - CLAMPS 격자 중 14 외의 값은 분포 민감도용이며 특정 문헌 기준으로
    명명하지 않는다 (원문 미확인 상태에서 라벨링 금지).
  - 어느 축에서도 fusion > XGB / fusion > CNN 이 유지되는지 자동 점검한다.
    이 프로젝트의 핵심 공헌은 축에 의존하지 않아야 한다.

env: hvf. 사용:
  HVF_ROOT="$(pwd)" python scripts/stepsize_sensitivity.py
  # 재학습 축까지:
  HVF_ROOT="$(pwd)" python scripts/stepsize_sensitivity.py \\
      --retrain_cnn_dir runs/phasec_b0_inception_resnet_v2_5fold_clamp14 \\
      --retrain_xgb_tag 90d_clamp14 --retrain_floor 14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz  # noqa: E402

W_FIXED = 0.47
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
XGB_TAG = '90d'
CLAMPS = [None, 10.0, 14.0, 15.0, 19.0]  # None=raw, 14=Hasan, 나머지는 분포 민감도용


def gather_oof(cnn_dir: Path, xgb_tag: str):
    """CNN·XGB val fold npz를 행 키로 정렬해 OOF 240안으로 합친다."""
    cnn, xgb, lab, msk = [], [], [], []
    for k in range(5):
        c = load_oof_npz(cnn_dir / f'val_preds_fold{k}.npz')
        x = load_oof_npz(ROOT / f'runs/oof/xgb_{xgb_tag}_fold{k}_val.npz')
        xi = {kk: i for i, kk in enumerate(x['keys'])}
        for i, key in enumerate(c['keys']):
            if key not in xi:
                continue
            j = xi[key]
            cnn.append(c['pred'][i])
            xgb.append(x['pred'][j])
            lab.append(c['labels'][i])
            msk.append(c['mask'][i] & x['mask'][j])
    return (np.stack(cnn), np.stack(xgb), np.stack(lab),
            np.stack(msk).astype(bool))


def gather_test(cnn_dir: Path, xgb_tag: str):
    """test 37안. 정본 집계 = 5-fold '예측 평균' (fold별 지표 평균이 아님).

    이 프로젝트의 기준 수치(fusion 8.401)가 이 방식으로 산출됐다.
    fold별 지표를 평균하면 8.608이 나오는 다른 값이 되므로 섞지 말 것.
    """
    cs = [load_oof_npz(cnn_dir / f'test_preds_fold{k}.npz') for k in range(5)]
    xs = [load_oof_npz(ROOT / f'runs/oof/xgb_{xgb_tag}_fold{k}_test.npz')
          for k in range(5)]
    base = cs[0]['keys']

    def stack_mean(ds):
        out = []
        for d in ds:
            ii = {kk: i for i, kk in enumerate(d['keys'])}
            out.append(np.stack([d['pred'][ii[k]] for k in base]))
        return np.mean(out, axis=0)

    cnn = stack_mean(cs)
    xgb = stack_mean(xs)
    lab = cs[0]['labels']
    msk = cs[0]['mask'].astype(bool)
    for d in xs:
        ii = {kk: i for i, kk in enumerate(d['keys'])}
        msk = msk & np.stack([d['mask'][ii[k]] for k in base]).astype(bool)
    return cnn, xgb, lab, msk


def metrics(pred, lab, msk, floor, mode='both'):
    """floor 클램프 후 집계.

    mode='both'  : 예측·라벨 모두 클램프 (낙관 상한)
    mode='label' : 라벨만 클램프, 예측은 원본 (보수 하한)

    재학습 축에는 mode='label'을 쓴다. 이미 정규화 타깃으로 학습돼
    라벨이 클램프돼 있으므로 np.maximum은 무연산(idempotent)이고,
    예측은 그 공간에서 자연히 나온 값이라 손대지 않는 것이 맞다.
    """
    m = msk.astype(bool)
    p = pred[m].astype(np.float64)
    y = lab[m].astype(np.float64)
    if floor is not None:
        y = np.maximum(y, floor)
        if mode == 'both':
            p = np.maximum(p, floor)
    d = p - y
    return {
        'rmse': float(np.sqrt(np.mean(d ** 2))),
        'mae': float(np.mean(np.abs(d))),
        'n_points': int(d.size),
    }


def best_w(xgb, cnn, lab, msk, floor, mode):
    """해당 축에서 융합 가중치를 다시 최적화 (w가 축에 의존하는지 확인용)."""
    best = (None, float('inf'))
    for w in np.linspace(0.0, 1.0, 101):
        r = metrics(w * xgb + (1 - w) * cnn, lab, msk, floor, mode)['rmse']
        if r < best[1]:
            best = (float(w), r)
    return {'w': best[0], 'rmse': best[1]}


def stratified(pred, lab, msk, floor, mode, raw_lab):
    """저감도(raw<floor) / 정상(raw>=floor) 구간을 나눠 MAE를 본다.

    재학습 축이 총합에서 이기는 것이 '저감도를 포기하고 정상 구간을 얻는' 거래인지
    확인하기 위한 분해. 녹내장에서 임상적으로 중요한 쪽은 저감도 구간이므로,
    총합 MAE만 보고하면 그 거래가 숨는다.
    """
    m = msk.astype(bool)
    p = pred[m].astype(np.float64)
    y = lab[m].astype(np.float64)
    ry = raw_lab[m].astype(np.float64)
    if floor is not None:
        y = np.maximum(y, floor)
        if mode == 'both':
            p = np.maximum(p, floor)
    lo = ry < floor if floor is not None else np.zeros(ry.shape, bool)
    out = {}
    for tag, sel in (('low', lo), ('normal', ~lo)):
        if sel.sum() == 0:
            out[tag] = None
            continue
        out[tag] = {'mae': float(np.mean(np.abs(p[sel] - y[sel]))),
                    'n_points': int(sel.sum())}
    out['frac_pred_below_floor'] = (
        float(np.mean(p < floor)) if floor is not None else None)
    return out


def eye_rmse(pred, lab, msk, floor, mode):
    """안 단위 RMSE 벡터. paired 검정의 표본 단위 = 안(eye)."""
    out = []
    for i in range(pred.shape[0]):
        m = msk[i].astype(bool)
        if not m.any():
            out.append(np.nan)
            continue
        p = pred[i][m].astype(np.float64)
        y = lab[i][m].astype(np.float64)
        if floor is not None:
            y = np.maximum(y, floor)
            if mode == 'both':
                p = np.maximum(p, floor)
        out.append(float(np.sqrt(np.mean((p - y) ** 2))))
    return np.asarray(out, dtype=np.float64)


def paired_test(a, b):
    """a=fusion, b=solo 의 안 단위 RMSE. a<b 면 fusion 우세.

    2026-08-08 추가. 종전에는 pooled RMSE 크기만 비교해 부호만 봤으므로
    '유의하게'라는 표현을 쓸 수 없었다. 이 함수로 p값을 붙인다.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0 or np.allclose(a - b, 0):
        return {'wilcoxon_p': float('nan'), 'wins': 0, 'n': int(a.size),
                'median_delta': 0.0}
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = float('nan')
    return {'wilcoxon_p': float(p),
            'wins': int(np.sum(a < b)),
            'n': int(a.size),
            'median_delta': float(np.median(a - b))}


def axis_block(xgb, cnn, lab, msk, floor, mode, raw_lab=None):
    fus = W_FIXED * xgb + (1 - W_FIXED) * cnn
    blk = {
        'xgb': metrics(xgb, lab, msk, floor, mode),
        'cnn': metrics(cnn, lab, msk, floor, mode),
        'fusion_w047': metrics(fus, lab, msk, floor, mode),
        'fusion_refit_w': best_w(xgb, cnn, lab, msk, floor, mode),
    }
    # 저감도/정상 구간 분해 — 총합 MAE 뒤에 숨는 '거래'를 드러낸다.
    if floor is not None:
        rl = raw_lab if raw_lab is not None else lab
        blk['stratified_fusion'] = stratified(fus, lab, msk, floor, mode, rl)
    # 핵심 공헌 불변 점검: 어느 축에서도 fusion이 두 단일 모델보다 나아야 한다.
    blk['contribution_holds'] = {
        'fusion_beats_xgb': bool(blk['fusion_w047']['rmse'] < blk['xgb']['rmse']),
        'fusion_beats_cnn': bool(blk['fusion_w047']['rmse'] < blk['cnn']['rmse']),
    }
    # 안 단위 paired Wilcoxon (2026-08-08 추가). 위 부호 비교에 유의성을 붙인다.
    ef = eye_rmse(fus, lab, msk, floor, mode)
    blk['paired'] = {
        'fusion_vs_xgb': paired_test(ef, eye_rmse(xgb, lab, msk, floor, mode)),
        'fusion_vs_cnn': paired_test(ef, eye_rmse(cnn, lab, msk, floor, mode)),
    }
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain_cnn_dir', default=None,
                    help='정규화 타깃으로 재학습한 CNN 런 디렉토리 (축 3)')
    ap.add_argument('--retrain_xgb_tag', default=None,
                    help='정규화 타깃으로 재적합한 XGB npz 태그 (축 3)')
    ap.add_argument('--retrain_floor', type=float, default=14.0)
    ap.add_argument('--split', choices=['oof', 'test'], default='oof',
                    help='oof=CV 240안(기본) / test=held-out 37안. '
                         'test는 5-fold 예측 평균으로 집계')
    ap.add_argument('--out', default=None,
                    help='기본: runs/stepsize_sensitivity[_test].json')
    args = ap.parse_args()

    gather = gather_oof if args.split == 'oof' else gather_test
    if args.out is None:
        args.out = ('runs/stepsize_sensitivity.json' if args.split == 'oof'
                    else 'runs/stepsize_sensitivity_test.json')

    cnn, xgb, lab, msk = gather(CNN_DIR, XGB_TAG)
    m = msk.astype(bool)
    y = lab[m]

    dist = {
        'n_points': int(y.size),
        'frac_below_10dB': float(np.mean(y < 10)),
        'frac_below_14dB': float(np.mean(y < 14)),
        'frac_below_15dB': float(np.mean(y < 15)),
        'frac_below_19dB': float(np.mean(y < 19)),
        'frac_eq_0dB': float(np.mean(y <= 0)),
        'label_mean_raw': float(y.mean()),
        'label_mean_clamp14': float(np.maximum(y, 14).mean()),
    }

    out = {
        'split': args.split,
        'protocol': (
            f'split={args.split} ('
            + ('OOF, CV fold별 val 예측 연결' if args.split == 'oof'
               else 'held-out test, 5-fold 예측 평균 = 정본 집계')
            + '), IR-v2 + XGB. 축 1(clamp_both)·2(clamp_label)는 재학습 없는 '
            '사후 재집계, 축 3(clamp_retrain)은 정규화 타깃으로 재학습한 별도 런. '
            'Hasan 2025 normalised TS = floor 14 dB. raw 축이 본 연구의 정본.'),
        'axis_semantics': {
            'clamp_both': '예측·라벨 동시 사후 클램프 (학습 후 예측을 눌러줌)',
            'clamp_label': '라벨만 사후 클램프, 예측은 원본 (모델 무수정)',
            'clamp_retrain': '정규화 타깃으로 재학습 — Hasan과 유일하게 동등',
            'WARNING': ('세 축은 하나의 값을 감싸는 구간이 아니다. 2026-08-08 XGB 실측에서 '
                        'retrain(4.180) < both(4.363) < label(5.177) 로 재학습이 both보다 '
                        '좋아 both를 상한이라 부를 수 없다. 각 축을 조작 내용과 함께 개별 '
                        '보고할 것.'),
        },
        'label_distribution': dist,
        'clamp_both': {},
        'clamp_label': {},
        'clamp_retrain': None,
    }

    for c in CLAMPS:
        tag = 'raw' if c is None else f'clamp{int(c)}'
        out['clamp_both'][tag] = axis_block(xgb, cnn, lab, msk, c, 'both')
        out['clamp_label'][tag] = axis_block(xgb, cnn, lab, msk, c, 'label')

    if args.retrain_cnn_dir and args.retrain_xgb_tag:
        # gather_oof는 (cnn, xgb, lab, msk) 순서로 돌려주고
        # axis_block은 (xgb, cnn, ...) 순서로 받는다. 순서 주의.
        r_cnn, r_xgb, r_lab, r_msk = gather(
            ROOT / args.retrain_cnn_dir, args.retrain_xgb_tag)
        floor = args.retrain_floor
        # 재학습 축의 라벨은 이미 clamp돼 있어 저감도 구간을 식별할 수 없다.
        # 기준 런의 raw 라벨로 구간을 나눈다 (행 순서가 같은지 먼저 확인).
        if r_lab.shape != lab.shape or not np.array_equal(r_msk, msk):
            raise SystemExit('재학습 축과 기준 축의 행 정렬이 다릅니다 — 분해 중단')
        if not np.allclose(r_lab[msk], np.maximum(lab[msk], floor)):
            raise SystemExit('재학습 라벨이 기준 라벨의 clamp와 불일치 — 분해 중단')
        blk = axis_block(r_xgb, r_cnn, r_lab, r_msk, floor, 'label', raw_lab=lab)
        blk['floor'] = floor
        blk['cnn_dir'] = args.retrain_cnn_dir
        blk['xgb_tag'] = args.retrain_xgb_tag
        out['clamp_retrain'] = blk

    (ROOT / args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    n_eyes = lab.shape[0]
    print(f'=== split={args.split} | 라벨 분포 ({n_eyes}안, 유효 지점) ===')
    print(f"  n_points          : {dist['n_points']}")
    print(f"  <10 dB            : {dist['frac_below_10dB']*100:.1f}%")
    print(f"  <14 dB (Hasan 축) : {dist['frac_below_14dB']*100:.1f}%")
    print(f"  <15 dB            : {dist['frac_below_15dB']*100:.1f}%")
    print(f"  <19 dB            : {dist['frac_below_19dB']*100:.1f}%")
    print(f"  =0 dB             : {dist['frac_eq_0dB']*100:.1f}%")

    for mode in ('clamp_both', 'clamp_label'):
        label = ('예측·라벨 동시 사후 클램프' if mode == 'clamp_both'
                 else '라벨만 사후 클램프 (모델 무수정)')
        print(f'\n=== {mode} — {label} (MAE / RMSE, dB) ===')
        print(f"{'축':<10} {'XGB MAE':>9} {'CNN MAE':>9} {'fus MAE':>9} "
              f"{'fus RMSE':>10} {'w*':>6} {'fus>XGB':>8}")
        for tag, v in out[mode].items():
            ok = 'OK' if v['contribution_holds']['fusion_beats_xgb'] else 'FAIL'
            print(f"{tag:<10} {v['xgb']['mae']:>9.3f} {v['cnn']['mae']:>9.3f} "
                  f"{v['fusion_w047']['mae']:>9.3f} {v['fusion_w047']['rmse']:>10.3f} "
                  f"{v['fusion_refit_w']['w']:>6.2f} {ok:>8}")

    b14 = out['clamp_both']['clamp14']['fusion_w047']['mae']
    l14 = out['clamp_label']['clamp14']['fusion_w047']['mae']
    print(f'\n=== 14 dB 축 요약 (fusion MAE) ===')
    print(f'  clamp_both        : {b14:.3f}')
    print(f'  clamp_label       : {l14:.3f}')
    if out['clamp_retrain']:
        rb = out['clamp_retrain']
        print(f"  재학습 (Hasan 동등): {rb['fusion_w047']['mae']:.3f}")
        print('  ※ 세 값은 구간이 아니라 서로 다른 조작의 결과다.')
        print('\n=== 저감도(raw<14) vs 정상(raw>=14) 분해 — fusion MAE ===')
        for name, blk in (('clamp_both', out['clamp_both']['clamp14']),
                          ('clamp_label', out['clamp_label']['clamp14']),
                          ('clamp_retrain', rb)):
            s = blk.get('stratified_fusion')
            if not s:
                continue
            lo = s['low']['mae'] if s['low'] else float('nan')
            no = s['normal']['mae'] if s['normal'] else float('nan')
            pb = s['frac_pred_below_floor']
            print(f"  {name:<14} 저감도 {lo:>6.3f}  정상 {no:>6.3f}  "
                  f"예측<14 비율 {pb*100:>5.1f}%")
        print('  → 재학습이 총합에서 이긴다면 저감도를 포기한 대가인지 위 표로 확인할 것.')
    else:
        print('  재학습 축         : 미수행')

    # 두 공헌을 따로 집계한다. fusion>XGB 만 보고 '유지'라고 출력하면
    # fusion>CNN 실패가 JSON에만 남고 화면에서 사라진다 (2026-08-08 실제로 그랬음).
    def collect(key):
        f = [(mo, t) for mo in ('clamp_both', 'clamp_label')
             for t, v in out[mo].items()
             if not v['contribution_holds'][key]]
        if out['clamp_retrain'] and not out['clamp_retrain']['contribution_holds'][key]:
            f.append(('clamp_retrain', f"clamp{int(args.retrain_floor)}"))
        return f

    print()
    for key, name in (('fusion_beats_xgb', 'fusion > XGB  (주공헌)'),
                      ('fusion_beats_cnn', 'fusion > CNN  (약한 주장)')):
        f = collect(key)
        if not f:
            print(f'{name}: 모든 축에서 유지')
        else:
            print(f'{name}: ⚠ {len(f)}개 축에서 유지 실패')
            for mo, t in f:
                print(f'    - {mo}/{t}')

    # 안 단위 paired Wilcoxon — 부호 비교에 유의성을 붙인다 (2026-08-08 추가).
    def blocks():
        for mo in ('clamp_both', 'clamp_label'):
            for t, v in out[mo].items():
                yield f'{mo}/{t}', v
        if out['clamp_retrain']:
            yield (f"clamp_retrain/clamp{int(args.retrain_floor)}",
                   out['clamp_retrain'])

    print(f'\n=== 안 단위 paired Wilcoxon (n={n_eyes}안) ===')
    print(f"{'축':<26} {'fus>XGB p':>11} {'승':>8} {'fus>CNN p':>11} {'승':>8}")
    sig = {'fusion_vs_xgb': 0, 'fusion_vs_cnn': 0}
    tot = 0
    for name, v in blocks():
        pr = v['paired']
        tot += 1
        for k in sig:
            if pr[k]['wilcoxon_p'] < 0.05 and pr[k]['median_delta'] < 0:
                sig[k] += 1
        print(f"{name:<26} {pr['fusion_vs_xgb']['wilcoxon_p']:>11.2e} "
              f"{pr['fusion_vs_xgb']['wins']:>4}/{pr['fusion_vs_xgb']['n']:<3} "
              f"{pr['fusion_vs_cnn']['wilcoxon_p']:>11.2e} "
              f"{pr['fusion_vs_cnn']['wins']:>4}/{pr['fusion_vs_cnn']['n']:<3}")
    print(f"  → fusion>XGB 유의(p<.05 & fusion 우세): {sig['fusion_vs_xgb']}/{tot} 축")
    print(f"  → fusion>CNN 유의(p<.05 & fusion 우세): {sig['fusion_vs_cnn']}/{tot} 축")
    print(f"저장: {args.out}")


if __name__ == '__main__':
    main()
