#!/usr/bin/env python3
"""vert_cd ablation: 26 vs 25(=vert_cd 제거) feature XGB를 동일 코드로 재적합하고
동일 CNN(IR-v2 seed42)과 late fusion을 재계산 → fusion이 여전히 영상단독을 이기는지 검증.

공헌 프레이밍("두께맵에 없는 유두지표 vert_cd가 이득의 원천이 아니다") 확인용.
재학습=XGB만(CPU). CNN 예측은 기존 npz 재사용. 출력: runs/ablation_vert_cd.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import row_key, load_oof_npz, align_oof  # noqa: E402

CSV = ROOT / 'ml_final_90d_excl_empty_flip.csv'
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'
PT = [f'p{i+1:02d}' for i in range(54) if f'p{i+1:02d}' not in ('p26', 'p35')]
N_PT = len(PT)
RNFL = ['rnfl_q_s','rnfl_q_t','rnfl_q_i','rnfl_q_n'] + [f'rnfl_h{i:02d}' for i in range(1,13)]

def feats(eye, drop_vertcd):
    s = 'od' if eye == 'OD' else 'os'
    base = [f'avg_gcl_{s}', f'min_gcl_{s}',
            f'{s}_s_sup', f'{s}_s_sup_t', f'{s}_s_inf_t', f'{s}_s_inf', f'{s}_s_inf_n', f'{s}_s_sup_n',
            f'{s}_avg_rnfl']
    if not drop_vertcd:
        base.append(f'{s}_vert_cd')
    return base + RNFL

XGB_KW = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, n_jobs=4,
              random_state=42, verbosity=0)

def row_to_xy(row, drop_vertcd):
    fx = [float(row[f]) if row.get(f, '') not in ('', None) else np.nan for f in feats(row['eye'], drop_vertcd)]
    y, m = [], []
    for p in PT:
        v = row.get(p, '')
        if v in ('', None, '-1'):
            y.append(0.0); m.append(False)
        else:
            y.append(float(v)); m.append(True)
    return np.array(fx, np.float32), np.array(y, np.float32), np.array(m, bool)

def fit_predict(rows_tr, rows_va, drop_vertcd):
    Xtr = np.stack([row_to_xy(r, drop_vertcd)[0] for r in rows_tr])
    Ytr = np.stack([row_to_xy(r, drop_vertcd)[1] for r in rows_tr])
    Mtr = np.stack([row_to_xy(r, drop_vertcd)[2] for r in rows_tr])
    Xva = np.stack([row_to_xy(r, drop_vertcd)[0] for r in rows_va])
    Yva = np.stack([row_to_xy(r, drop_vertcd)[1] for r in rows_va])
    Mva = np.stack([row_to_xy(r, drop_vertcd)[2] for r in rows_va])
    cm = np.nanmean(Xtr, axis=0); cm = np.where(np.isnan(cm), 0.0, cm)
    Xtr = np.where(np.isnan(Xtr), cm, Xtr); Xva = np.where(np.isnan(Xva), cm, Xva)
    pred = np.full((len(rows_va), N_PT), np.nan, np.float32)
    for pi in range(N_PT):
        valid = Mtr[:, pi]
        if valid.sum() < 10:
            continue
        mdl = XGBRegressor(**XGB_KW)
        mdl.fit(Xtr[valid], Ytr[valid, pi])
        pred[:, pi] = mdl.predict(Xva)
    return pred, Yva, Mva

def xgb_oof(rows, drop_vertcd):
    """returns per-fold list of dict(keys,pred,labels,mask) for val and test."""
    val_folds, test_folds = [], []
    for k in range(5):
        vf = str(k)
        rows_tr = [r for r in rows if r['cv_fold'] not in (vf, 'test')]
        rows_va = [r for r in rows if r['cv_fold'] == vf]
        rows_te = [r for r in rows if r['cv_fold'] == 'test']
        pv, yv, mv = fit_predict(rows_tr, rows_va, drop_vertcd)
        pt, yt, mt = fit_predict(rows_tr, rows_te, drop_vertcd)
        val_folds.append({'keys':[row_key(r) for r in rows_va],'pred':pv,'labels':yv,'mask':mv,'meta':{}})
        test_folds.append({'keys':[row_key(r) for r in rows_te],'pred':pt,'labels':yt,'mask':mt,'meta':{}})
    return val_folds, test_folds

def pooled(pred, labels, mask):
    m = mask.astype(bool); d = (pred - labels)[m]
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))

def per_eye_rmse(pred, labels, mask):
    out = []
    for i in range(pred.shape[0]):
        m = mask[i].astype(bool)
        out.append(np.sqrt(np.mean((pred[i][m]-labels[i][m])**2)) if m.sum() else np.nan)
    return np.array(out)

def best_w(xp, cp, lb, mk):
    m = mk.astype(bool); yx = xp[m]; yc = cp[m]; y = lb[m]
    bw, br = 0.0, 1e9
    for w in np.linspace(0, 1, 101):
        r = np.sqrt(np.mean((w*yx + (1-w)*yc - y)**2))
        if r < br: br, bw = r, float(w)
    return bw

def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    try: _, wp = stats.wilcoxon(a, b)
    except ValueError: wp = float('nan')
    return float(wp), int(np.sum(a < b)), len(a)

def build_oof(xgb_val, cnn_name):
    XS, CS, LS, MS = [], [], [], []
    for k in range(5):
        cz = load_oof_npz(CNN_DIR / f'{cnn_name}_fold{k}.npz')
        ax, bx, _ = align_oof(xgb_val[k], cz)
        m = ax['mask'] & bx['mask']
        XS.append(ax['pred']); CS.append(bx['pred']); LS.append(ax['labels']); MS.append(m)
    return (np.concatenate(XS), np.concatenate(CS), np.concatenate(LS), np.concatenate(MS))

def build_test(xgb_test):
    xk, ck, LB, MK = [], [], None, None
    for k in range(5):
        cz = load_oof_npz(CNN_DIR / f'test_preds_fold{k}.npz')
        ax, bx, _ = align_oof(xgb_test[k], cz)
        if LB is None: LB, MK = ax['labels'], ax['mask'] & bx['mask']
        xk.append(ax['pred']); ck.append(bx['pred'])
    return np.mean(np.stack(xk),0), np.mean(np.stack(ck),0), LB, MK

def evaluate(rows, drop_vertcd, tag):
    vf, tf = xgb_oof(rows, drop_vertcd)
    XP, CP, LB, MK = build_oof(vf, 'val_preds')
    xr, xm = pooled(XP, LB, MK); cr, cm = pooled(CP, LB, MK)
    res = {'tag': tag, 'n_features': len(feats('OD', drop_vertcd)),
           'oof': {'xgb_rmse': xr, 'xgb_mae': xm, 'cnn_rmse': cr, 'cnn_mae': cm}}
    for wlabel, w in [('w0.47', 0.47), ('w_refit', best_w(XP, CP, LB, MK))]:
        F = w*XP + (1-w)*CP
        fr, fm = pooled(F, LB, MK)
        ef, ex, ec = per_eye_rmse(F,LB,MK), per_eye_rmse(XP,LB,MK), per_eye_rmse(CP,LB,MK)
        p_c, win_c, n = paired(ef, ec); p_x, win_x, _ = paired(ef, ex)
        res['oof'][wlabel] = {'w_xgb': round(w,3), 'fusion_rmse': fr, 'fusion_mae': fm,
            'fus_vs_cnn_p': p_c, 'fus_vs_cnn_win': f'{win_c}/{n}',
            'fus_vs_xgb_p': p_x, 'fus_vs_xgb_win': f'{win_x}/{n}'}
    # test (ensemble, fixed w0.47)
    XPt, CPt, LBt, MKt = build_test(tf)
    Ft = 0.47*XPt + 0.53*CPt
    tr, tm = pooled(Ft, LBt, MKt); txr,_ = pooled(XPt,LBt,MKt)
    ef, ex = per_eye_rmse(Ft,LBt,MKt), per_eye_rmse(XPt,LBt,MKt)
    p_x, win_x, n = paired(ef, ex)
    res['test'] = {'fusion_rmse': tr, 'xgb_rmse': txr, 'fus_vs_xgb_p': p_x, 'fus_vs_xgb_win': f'{win_x}/{n}'}
    return res

def main():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8-sig')))
    out = {}
    for drop, tag in [(False, '26_feat_baseline'), (True, '25_feat_no_vertcd')]:
        print(f'\n=== {tag} ({len(feats("OD", drop))} feat) 재적합 중... ===', flush=True)
        r = evaluate(rows, drop, tag)
        out[tag] = r
        o = r['oof']
        print(f"  OOF XGB {o['xgb_rmse']:.3f}/{o['xgb_mae']:.3f} | CNN {o['cnn_rmse']:.3f}/{o['cnn_mae']:.3f}")
        for wl in ('w0.47','w_refit'):
            b = o[wl]
            print(f"  [{wl} w={b['w_xgb']}] fusion {b['fusion_rmse']:.3f}/{b['fusion_mae']:.3f} "
                  f"| fus>CNN p={b['fus_vs_cnn_p']:.4g} {b['fus_vs_cnn_win']} "
                  f"| fus>XGB p={b['fus_vs_xgb_p']:.4g} {b['fus_vs_xgb_win']}")
        print(f"  TEST fusion {r['test']['fusion_rmse']:.3f} vs XGB {r['test']['xgb_rmse']:.3f} "
              f"| fus>XGB p={r['test']['fus_vs_xgb_p']:.4g} {r['test']['fus_vs_xgb_win']}")
    (ROOT/'runs/ablation_vert_cd.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print('\n저장: runs/ablation_vert_cd.json')

if __name__ == '__main__':
    main()
