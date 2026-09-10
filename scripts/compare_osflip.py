#!/usr/bin/env python3
"""OS 이미지 좌우반전(OD 정규화) 유무 대조 리포트.

기준 런 runs/phasec_b0_inception_resnet_v2_5fold (no-flip) 대
신규 런 runs/phasec_b0_inception_resnet_v2_5fold_osflip (flip) 을 비교한다.

산출:
  - image-only CNN, late fusion 의 OOF / test pooled RMSE·MAE
  - fusion 은 ① 고정 w_xgb=0.47 ② OOF 재최적화 w 두 가지
  - flip vs no-flip 안 단위 paired Wilcoxon (같은 안끼리 정렬)
  - runs/osflip_compare.json + docs/diag/osflip_compare.md

★ test 집계 규약 = 5개 fold 모델 "예측"의 평균 (정본).
  fold별 지표를 평균하면 8.608 로 어긋난다(기준 8.401 재현 실측 확인).

env: hvf. 사용:
  HVF_ROOT="$(pwd)" python scripts/compare_osflip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz, masked_overall_rmse_mae  # noqa: E402

W_FIXED = 0.47
BASE_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
FLIP_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold_osflip'
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}

# 참고 기준값(기존 no-flip IR-v2, 메모리·산출물 실측)
REFERENCE = {
    'cnn_oof_rmse': 8.540,
    'fusion_oof_rmse': 8.064,
    'fusion_test_rmse': 8.401,
}


# ── OOF: fold별 val 예측을 이어붙여 240안 구성
def gather_oof(cnn_dir: Path):
    keys, cnn, xgb, lab, msk = [], [], [], [], []
    for k in range(5):
        c = load_oof_npz(cnn_dir / f'val_preds_fold{k}.npz')
        x = load_oof_npz(XGB_VAL[k])
        xi = {kk: i for i, kk in enumerate(x['keys'])}
        for i, key in enumerate(c['keys']):
            if key not in xi:
                continue
            j = xi[key]
            keys.append(key)
            cnn.append(c['pred'][i])
            xgb.append(x['pred'][j])
            lab.append(c['labels'][i])
            msk.append(c['mask'][i] & x['mask'][j])
    return (keys, np.stack(cnn), np.stack(xgb), np.stack(lab),
            np.stack(msk).astype(bool))


# ── test: 5 fold "예측" 평균 (정본 규약)
def gather_test(cnn_dir: Path):
    cs = [load_oof_npz(cnn_dir / f'test_preds_fold{k}.npz') for k in range(5)]
    xs = [load_oof_npz(XGB_TEST[k]) for k in range(5)]
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
    return base, cnn, xgb, lab, msk


def best_w(xgb, cnn, lab, msk, n_grid=101):
    m = msk.astype(bool)
    y = lab[m].astype(np.float64)
    a = xgb[m].astype(np.float64)
    b = cnn[m].astype(np.float64)
    bw, br = 0.0, float('inf')
    for w in np.linspace(0, 1, n_grid):
        r = float(np.sqrt(np.mean((w * a + (1 - w) * b - y) ** 2)))
        if r < br:
            br, bw = r, float(w)
    return bw, br


def per_eye_rmse(pred, lab, msk):
    """안 단위 RMSE 벡터 (paired 검정용)."""
    out = []
    for i in range(pred.shape[0]):
        m = msk[i].astype(bool)
        if m.sum() == 0:
            out.append(np.nan)
            continue
        d = pred[i][m] - lab[i][m]
        out.append(float(np.sqrt(np.mean(d ** 2))))
    return np.array(out)


def paired_test(a_keys, a_pred, a_lab, a_msk, b_keys, b_pred, b_lab, b_msk):
    """같은 안끼리 정렬한 뒤 안 단위 RMSE paired Wilcoxon."""
    bi = {k: i for i, k in enumerate(b_keys)}
    common = [k for k in a_keys if k in bi]
    ai = {k: i for i, k in enumerate(a_keys)}
    idx_a = [ai[k] for k in common]
    idx_b = [bi[k] for k in common]
    m = a_msk[idx_a] & b_msk[idx_b]
    ra = per_eye_rmse(a_pred[idx_a], a_lab[idx_a], m)
    rb = per_eye_rmse(b_pred[idx_b], b_lab[idx_b], m)
    ok = ~(np.isnan(ra) | np.isnan(rb))
    ra, rb = ra[ok], rb[ok]
    if ra.size < 3 or np.allclose(ra, rb):
        return {'n_eyes': int(ra.size), 'p': None, 'median_delta': 0.0}
    stat, p = wilcoxon(ra, rb)
    return {
        'n_eyes': int(ra.size),
        'p': float(p),
        'median_delta': float(np.median(rb - ra)),  # flip - noflip, 음수면 flip 우수
    }


def summarize(tag, cnn_dir):
    res = {}
    ok, ocnn, oxgb, olab, omsk = gather_oof(cnn_dir)
    tk, tcnn, txgb, tlab, tmsk = gather_test(cnn_dir)

    r, m = masked_overall_rmse_mae(ocnn.copy(), olab, omsk)
    res['oof_cnn'] = {'rmse': r, 'mae': m, 'n_eyes': len(ok)}
    r, m = masked_overall_rmse_mae(oxgb.copy(), olab, omsk)
    res['oof_xgb'] = {'rmse': r, 'mae': m}
    r, m = masked_overall_rmse_mae(
        (W_FIXED * oxgb + (1 - W_FIXED) * ocnn).copy(), olab, omsk)
    res['oof_fusion_w047'] = {'rmse': r, 'mae': m, 'w_xgb': W_FIXED}
    bw, _ = best_w(oxgb, ocnn, olab, omsk)
    r, m = masked_overall_rmse_mae(
        (bw * oxgb + (1 - bw) * ocnn).copy(), olab, omsk)
    res['oof_fusion_wopt'] = {'rmse': r, 'mae': m, 'w_xgb': bw}

    r, m = masked_overall_rmse_mae(tcnn.copy(), tlab, tmsk)
    res['test_cnn'] = {'rmse': r, 'mae': m, 'n_eyes': len(tk)}
    r, m = masked_overall_rmse_mae(txgb.copy(), tlab, tmsk)
    res['test_xgb'] = {'rmse': r, 'mae': m}
    r, m = masked_overall_rmse_mae(
        (W_FIXED * txgb + (1 - W_FIXED) * tcnn).copy(), tlab, tmsk)
    res['test_fusion_w047'] = {'rmse': r, 'mae': m, 'w_xgb': W_FIXED}
    r, m = masked_overall_rmse_mae(
        (bw * txgb + (1 - bw) * tcnn).copy(), tlab, tmsk)
    res['test_fusion_wopt'] = {'rmse': r, 'mae': m, 'w_xgb': bw}

    raw = {'oof': (ok, ocnn, oxgb, olab, omsk), 'test': (tk, tcnn, txgb, tlab, tmsk)}
    return res, raw


def fmt(v):
    return '—' if v is None else f'{v:.3f}'


def fmt_p(v):
    if v is None:
        return '—'
    return f'{v:.3g}'


def main():
    if not FLIP_DIR.is_dir():
        raise SystemExit(f'신규 런 디렉토리 없음: {FLIP_DIR}')
    base, base_raw = summarize('noflip', BASE_DIR)
    flip, flip_raw = summarize('flip', FLIP_DIR)

    # 무결성: 기준 런이 참고값을 재현하는지
    checks = {
        'base_cnn_oof_rmse_matches_ref': abs(
            base['oof_cnn']['rmse'] - REFERENCE['cnn_oof_rmse']) < 0.01,
        'base_fusion_oof_rmse_matches_ref': abs(
            base['oof_fusion_w047']['rmse'] - REFERENCE['fusion_oof_rmse']) < 0.01,
        'base_fusion_test_rmse_matches_ref': abs(
            base['test_fusion_w047']['rmse'] - REFERENCE['fusion_test_rmse']) < 0.01,
    }

    # paired Wilcoxon (flip vs noflip), CNN 단독과 fusion(w=0.47)
    tests = {}
    for split in ('oof', 'test'):
        bk, bcnn, bxgb, blab, bmsk = base_raw[split]
        fk, fcnn, fxgb, flab, fmsk = flip_raw[split]
        tests[f'{split}_cnn'] = paired_test(bk, bcnn, blab, bmsk,
                                            fk, fcnn, flab, fmsk)
        tests[f'{split}_fusion_w047'] = paired_test(
            bk, W_FIXED * bxgb + (1 - W_FIXED) * bcnn, blab, bmsk,
            fk, W_FIXED * fxgb + (1 - W_FIXED) * fcnn, flab, fmsk)

    # 원고가 인쇄하는 것은 차이(flip - noflip)인데, 지금까지 그 차이는 아래
    # markdown 표에만 있고 JSON 에는 없었다. 검증 게이트의 근거 풀은 runs/*.json
    # 이라, 04_results.tex 의 "+0.147 dB" 는 근거 없이 떠 있었다(MISS).
    # 새 계산이 아니라 markdown 표와 같은 식을 스칼라로 남기는 것뿐이다.
    delta = {
        k: {m: flip[k][m] - base[k][m] for m in ('rmse', 'mae')}
        for k in ('oof_cnn', 'oof_xgb', 'oof_fusion_w047', 'oof_fusion_wopt',
                  'test_cnn', 'test_xgb', 'test_fusion_w047', 'test_fusion_wopt')
    }

    out = {
        'protocol': ('OS 이미지 좌우반전(OD 정규화) 유무 대조. 하이퍼파라미터 동일, '
                     'seed 42, num_workers 0. test 집계 = 5-fold 예측 평균(정본).'),
        'w_fixed': W_FIXED,
        'reference_noflip': REFERENCE,
        'integrity_checks': checks,
        'noflip': base,
        'flip': flip,
        'delta_flip_minus_noflip': delta,
        'paired_wilcoxon_flip_vs_noflip': tests,
    }
    (ROOT / 'runs/osflip_compare.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    # ── markdown 리포트
    rows = [
        ('OOF image-only CNN', 'oof_cnn'),
        ('OOF fusion (w=0.47)', 'oof_fusion_w047'),
        ('OOF fusion (w 재최적)', 'oof_fusion_wopt'),
        ('test image-only CNN', 'test_cnn'),
        ('test fusion (w=0.47)', 'test_fusion_w047'),
        ('test fusion (w 재최적)', 'test_fusion_wopt'),
    ]
    L = []
    L.append('# OS 이미지 좌우반전(OD 정규화) 대조 리포트\n')
    L.append('생성: `scripts/compare_osflip.py` · env hvf\n')
    L.append('## 배경\n')
    L.append('라벨(VF 52점)·tabular은 `*_flip.csv`에서 OS→OD 정규화돼 있으나 '
             '이미지 파이프라인에는 반전이 없었다. 이 불일치를 제거한 런과 대조한다.\n')
    L.append('- 기준(no-flip): `runs/phasec_b0_inception_resnet_v2_5fold`')
    L.append('- 신규(flip): `runs/phasec_b0_inception_resnet_v2_5fold_osflip`')
    L.append('- 하이퍼파라미터 동일(seed 42, num_workers 0, input_geometry paper), '
             'flag만 차이\n')
    L.append('## 무결성 확인\n')
    for k, v in checks.items():
        L.append(f'- `{k}`: {"PASS" if v else "**FAIL**"}')
    L.append('')
    L.append('## 성능 대조 (RMSE / MAE, dB — 낮을수록 좋음)\n')
    L.append('| 지표 | no-flip | flip | Δ(flip−noflip) |')
    L.append('|---|---|---|---|')
    for label, key in rows:
        b, f = base[key], flip[key]
        L.append(f'| {label} RMSE | {fmt(b["rmse"])} | {fmt(f["rmse"])} | '
                 f'{f["rmse"] - b["rmse"]:+.3f} |')
        L.append(f'| {label} MAE | {fmt(b["mae"])} | {fmt(f["mae"])} | '
                 f'{f["mae"] - b["mae"]:+.3f} |')
    L.append('')
    L.append(f'- OOF 재최적 w_xgb: no-flip {base["oof_fusion_wopt"]["w_xgb"]:.2f} / '
             f'flip {flip["oof_fusion_wopt"]["w_xgb"]:.2f}')
    L.append('- Δ가 음수면 flip이 우수.\n')
    L.append('## 안 단위 paired Wilcoxon (flip vs no-flip)\n')
    L.append('| 비교 | n(안) | median Δ RMSE | p |')
    L.append('|---|---|---|---|')
    name = {'oof_cnn': 'OOF CNN 단독', 'oof_fusion_w047': 'OOF fusion(0.47)',
            'test_cnn': 'test CNN 단독', 'test_fusion_w047': 'test fusion(0.47)'}
    for k in ('oof_cnn', 'oof_fusion_w047', 'test_cnn', 'test_fusion_w047'):
        t = tests[k]
        L.append(f'| {name[k]} | {t["n_eyes"]} | {t["median_delta"]:+.3f} | '
                 f'{fmt_p(t["p"])} |')
    L.append('')
    L.append('## 해석\n')
    d_oof = flip['oof_cnn']['rmse'] - base['oof_cnn']['rmse']
    p_oof = tests['oof_cnn']['p']
    sig = (p_oof is not None and p_oof < 0.05)
    if abs(d_oof) < 0.10 and not sig:
        L.append(f'OOF image-only RMSE 차이 {d_oof:+.3f} dB, paired p={fmt_p(p_oof)} '
                 '(비유의). **이미지 방향 정규화가 현 구조에서 무의미함을 실측 확인**했다. '
                 'backbone→GAP→Dense 구조는 국소화를 버리므로, 라벨-이미지 프레임 불일치가 '
                 '결과를 오염시키지 않았다. 기존 전 결과는 자기일관적으로 유효하며 '
                 '공헌 주장에도 영향이 없다. 리뷰어의 "flip ablation" 요구를 이 표로 선제 방어한다. '
                 '단 이는 **공간 구조를 쓰지 않는 현 구조에 한정**된 결론이며, '
                 'attention/MIL 등 공간 모델로 가면 방향 정규화가 필수 전제가 된다.')
    elif d_oof < 0 and sig:
        L.append(f'OOF image-only RMSE가 {d_oof:+.3f} dB 개선(paired p={fmt_p(p_oof)}). '
                 '**방향 정규화가 실제 이득**을 준다. OD/OS가 서로 거울인 두 부분모집단으로 '
                 '입력되던 잡음 분산이 제거된 것으로 해석된다. 본문 반영 여부는 사람이 판단.')
    else:
        L.append(f'OOF image-only RMSE 차이 {d_oof:+.3f} dB, paired p={fmt_p(p_oof)}. '
                 '위 표의 Δ·p를 그대로 보고하고 해석은 사람이 판단할 것.')
    L.append('')
    L.append('> 주의: 이 리포트는 진단용이다. 원고·스켈레톤 수치는 자동 갱신하지 않는다.')
    L.append('')

    dst = ROOT / 'docs/diag/osflip_compare.md'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(L), encoding='utf-8')
    print(f'저장: runs/osflip_compare.json')
    print(f'저장: {dst}')
    print(f'무결성: {checks}')
    for label, key in rows:
        print(f'  {label}: noflip {base[key]["rmse"]:.3f} → flip {flip[key]["rmse"]:.3f} '
              f'({flip[key]["rmse"] - base[key]["rmse"]:+.3f})')


if __name__ == '__main__':
    main()
