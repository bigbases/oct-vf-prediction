#!/usr/bin/env python3
"""Multi-seed 재현성: seed 42/43/44 의 IR-v2 CNN + 동일 XGB(seed42, 결정론적 baseline)로
fusion(w=0.47 고정) OOF/test 재계산. 헤드라인(fus>XGB)이 seed에 강건한지 확인.
출력: runs/multiseed_fusion.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz, align_oof  # noqa: E402

CNN_DIRS = {
    42: ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold',
    43: ROOT / 'runs/repro_final_ir_v2_5fold_s43',
    44: ROOT / 'runs/repro_final_ir_v2_5fold_s44',
}
XGB_VAL = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_val.npz' for k in range(5)}
XGB_TEST = {k: ROOT / f'runs/oof/xgb_90d_fold{k}_test.npz' for k in range(5)}
W = 0.47

def pooled(pred, lb, mk):
    m = mk.astype(bool); d = (pred - lb)[m]
    return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))
def per_eye(pred, lb, mk):
    return np.array([np.sqrt(np.mean((pred[i][mk[i].astype(bool)]-lb[i][mk[i].astype(bool)])**2))
                     if mk[i].any() else np.nan for i in range(pred.shape[0])])
def paired(a, b):
    ok=~(np.isnan(a)|np.isnan(b)); a,b=a[ok],b[ok]
    try: _,p=stats.wilcoxon(a,b)
    except ValueError: p=float('nan')
    return float(p), int((a<b).sum()), len(a)

def oof(cnn_dir):
    XS,CS,LS,MS=[],[],[],[]
    for k in range(5):
        xz=load_oof_npz(XGB_VAL[k]); cz=load_oof_npz(cnn_dir/f'val_preds_fold{k}.npz')
        ax,bx,_=align_oof(xz,cz); m=ax['mask']&bx['mask']
        XS.append(ax['pred']);CS.append(bx['pred']);LS.append(ax['labels']);MS.append(m)
    return (np.concatenate(a) for a in (XS,CS,LS,MS))

def test(cnn_dir):
    xk,ck,LB,MK=[],[],None,None
    for k in range(5):
        xz=load_oof_npz(XGB_TEST[k]); cz=load_oof_npz(cnn_dir/f'test_preds_fold{k}.npz')
        ax,bx,_=align_oof(xz,cz)
        if LB is None: LB,MK=ax['labels'],ax['mask']&bx['mask']
        xk.append(ax['pred']);ck.append(bx['pred'])
    return np.mean(np.stack(xk),0), np.mean(np.stack(ck),0), LB, MK

def block(XP,CP,LB,MK):
    F=W*XP+(1-W)*CP
    cr,cm=pooled(CP,LB,MK); fr,fm=pooled(F,LB,MK); xr,xm=pooled(XP,LB,MK)
    ef,ec,ex=per_eye(F,LB,MK),per_eye(CP,LB,MK),per_eye(XP,LB,MK)
    pC,wC,n=paired(ef,ec); pX,wX,_=paired(ef,ex)
    return {'cnn_rmse':cr,'fusion_rmse':fr,'fusion_mae':fm,'xgb_rmse':xr,
            'fus_vs_cnn_p':pC,'fus_vs_cnn_win':f'{wC}/{n}',
            'fus_vs_xgb_p':pX,'fus_vs_xgb_win':f'{wX}/{n}'}

def main():
    res={'w':W,'xgb':'shared seed42 baseline','seeds':{}}
    print(f"{'seed':>4} | {'OOF CNN':>7} {'OOF fus':>7} | {'fus>XGB p':>10} {'fus>CNN p':>10} | {'TEST fus':>8} {'test fus>XGB p':>13}")
    print('-'*80)
    for s,d in CNN_DIRS.items():
        o=block(*oof(d)); t=block(*test(d))
        res['seeds'][s]={'oof':o,'test':t}
        print(f"{s:>4} | {o['cnn_rmse']:>7.3f} {o['fusion_rmse']:>7.3f} | "
              f"{o['fus_vs_xgb_p']:>10.2e} {o['fus_vs_cnn_p']:>10.4f} | "
              f"{t['fusion_rmse']:>8.3f} {t['fus_vs_xgb_p']:>13.4f}")
    # 집계
    fr=[res['seeds'][s]['oof']['fusion_rmse'] for s in CNN_DIRS]
    cr=[res['seeds'][s]['oof']['cnn_rmse'] for s in CNN_DIRS]
    px=[res['seeds'][s]['oof']['fus_vs_xgb_p'] for s in CNN_DIRS]
    pc=[res['seeds'][s]['oof']['fus_vs_cnn_p'] for s in CNN_DIRS]
    res['summary']={'oof_fusion_rmse_mean':float(np.mean(fr)),'oof_fusion_rmse_std':float(np.std(fr)),
                    'oof_cnn_rmse_mean':float(np.mean(cr)),'oof_cnn_rmse_std':float(np.std(cr)),
                    'fus_vs_xgb_all_sig':bool(all(p<0.05 for p in px)),
                    'fus_vs_cnn_all_sig':bool(all(p<0.05 for p in pc))}
    print('-'*80)
    print(f"OOF fusion RMSE = {np.mean(fr):.3f} ± {np.std(fr):.3f} (seed간) | OOF CNN = {np.mean(cr):.3f} ± {np.std(cr):.3f}")
    print(f"fus>XGB 전 seed 유의(p<0.05): {res['summary']['fus_vs_xgb_all_sig']} | fus>CNN 전 seed 유의: {res['summary']['fus_vs_cnn_all_sig']}")
    (ROOT/'runs/multiseed_fusion.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
    print('저장: runs/multiseed_fusion.json')

if __name__=='__main__':
    main()
