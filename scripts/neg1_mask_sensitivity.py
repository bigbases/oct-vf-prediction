"""-1(원본) 지점을 마스크에서 제외했을 때 헤드라인이 얼마나 움직이는지.
재학습 없음. 예측 npz는 읽기만 한다.
행 대응: (pid,eye,vf_date) 정확일치 274 + 잔여 6은 (pid,eye) 유일후보로 1:1 확정.
검증: 대응된 모든 쌍에서 '-1 아닌 지점 값 불일치 0', '-1 지점 새값 != 0 개수 0'.
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(os.environ.get('HVF_ROOT', Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / 'scripts'))
from oof_common import load_oof_npz

PT52 = [f'p{i:02d}' for i in range(1, 55) if f'p{i:02d}' not in ('p26', 'p35')]
W = 0.47
CNN_DIR = ROOT / 'runs/phasec_b0_inception_resnet_v2_5fold'

def read(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))
old = read(ROOT/'runs/backup_vf_neg1_fix_20260530_193547/ml_final_90d_excl_empty_flip.csv')
new = read(ROOT/'ml_final_90d_excl_empty_flip.csv')
def vec(r): return np.array([float(r[p]) for p in PT52], dtype=np.float64)

# --- 1:1 행 대응 확정 ---
oi = {}
for i, r in enumerate(old): oi.setdefault((r['patient_id'],r['eye'],r['vf_date']), []).append(i)
pairs, used_o, used_n = [], set(), set()
for j, r in enumerate(new):
    k = (r['patient_id'],r['eye'],r['vf_date'])
    if oi.get(k):
        i = oi[k].pop(0); pairs.append((i,j)); used_o.add(i); used_n.add(j)
ro = [i for i in range(len(old)) if i not in used_o]
for j in [j for j in range(len(new)) if j not in used_n]:
    cand = [i for i in ro if (old[i]['patient_id'],old[i]['eye'])==(new[j]['patient_id'],new[j]['eye'])]
    assert len(cand)==1, (j, cand)
    i = cand[0]; ro.remove(i); pairs.append((i,j))
assert len(pairs)==len(new)==280

# --- 검증 + -1 마스크 ---
neg1 = {}   # new 행 인덱스 -> bool[52]
n_neg1 = bad_a = bad_b = 0
for i, j in pairs:
    o, n = vec(old[i]), vec(new[j]); m = (o == -1.0)
    neg1[j] = m; n_neg1 += int(m.sum())
    bad_a += int(np.sum((~m) & (o != n)))
    bad_b += int(np.sum(n[m] != 0.0))
print(f'행 대응 {len(pairs)}쌍 (triple 274 + 유일후보 6)')
print(f'검증: 비(-1) 지점 값 불일치 {bad_a}셀 / (-1) 지점 새값≠0 {bad_b}셀')
print(f'원본 -1 셀(52점): {n_neg1}  = {n_neg1/(280*52)*100:.2f}% of {280*52}')
assert bad_a == 0 and bad_b == 0

# new 행을 (pid,eye,vf_date)로 조회 — new 안의 중복 triple 여부 확인
nk = {}
for j, r in enumerate(new): nk.setdefault((r['patient_id'],r['eye'],r['vf_date']), []).append(j)
print('new 내 중복 triple 키:', {k:v for k,v in nk.items() if len(v)>1} or '없음')

def gather(split):
    cnn=xgb=None; C=[];X=[];L=[];M=[];K=[]
    if split=='oof':
        for kf in range(5):
            c = load_oof_npz(CNN_DIR/f'val_preds_fold{kf}.npz')
            x = load_oof_npz(ROOT/f'runs/oof/xgb_90d_fold{kf}_val.npz')
            xi = {kk:i for i,kk in enumerate(x['keys'])}
            for i,key in enumerate(c['keys']):
                if key not in xi: continue
                j=xi[key]
                C.append(c['pred'][i]); X.append(x['pred'][j])
                L.append(c['labels'][i]); M.append(c['mask'][i]&x['mask'][j]); K.append(key)
    else:
        cs=[load_oof_npz(CNN_DIR/f'test_preds_fold{kf}.npz') for kf in range(5)]
        xs=[load_oof_npz(ROOT/f'runs/oof/xgb_90d_fold{kf}_test.npz') for kf in range(5)]
        c0,x0=cs[0],xs[0]
        xi={kk:i for i,kk in enumerate(x0['keys'])}
        cp=np.mean([c['pred'] for c in cs],axis=0); xp=np.mean([x['pred'] for x in xs],axis=0)
        for i,key in enumerate(c0['keys']):
            if key not in xi: continue
            j=xi[key]
            C.append(cp[i]); X.append(xp[j]); L.append(c0['labels'][i])
            M.append(c0['mask'][i]&x0['mask'][j]); K.append(key)
    return np.stack(C),np.stack(X),np.stack(L),np.stack(M).astype(bool),K

def pooled(p,l,m):
    d=(p-l)[m]; return float(np.sqrt(np.mean(d**2))), float(np.mean(np.abs(d)))
def eyermse(p,l,m):
    return np.array([float(np.sqrt(np.mean((p[i][m[i]]-l[i][m[i]])**2))) if m[i].any() else np.nan
                     for i in range(p.shape[0])])
def wilc(a,b):
    ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    try: _,p=stats.wilcoxon(a,b)
    except ValueError: p=float('nan')
    return {'p':float(p),'wins':int(np.sum(a<b)),'n':int(a.size),'median_delta':float(np.median(a-b))}

out={}
for split in ('oof','test'):
    cnn,xgb,lab,msk,keys = gather(split)
    drop=np.zeros_like(msk); miss=0
    for i,k in enumerate(keys):
        kk=(str(k[0]),str(k[1]),str(k[2]))
        js=nk.get(kk)
        if not js: miss+=1; continue
        assert len(js)==1
        drop[i]=neg1[js[0]]
    assert miss==0, miss
    msk2 = msk & (~drop)
    fus = W*xgb+(1-W)*cnn
    rec={'n_eyes':len(keys),'cells_full':int(msk.sum()),'cells_excl':int(msk2.sum()),
         'cells_dropped':int((msk&drop).sum())}
    rec['dropped_pct']=round(rec['cells_dropped']/rec['cells_full']*100,2)
    for nm,arr in (('xgb',xgb),('cnn',cnn),('fusion',fus)):
        r1,m1=pooled(arr,lab,msk); r2,m2=pooled(arr,lab,msk2)
        rec[nm]={'rmse_full':round(r1,4),'rmse_excl':round(r2,4),'d_rmse':round(r2-r1,4),
                 'mae_full':round(m1,4),'mae_excl':round(m2,4),'d_mae':round(m2-m1,4)}
    for tag,mm in (('full',msk),('excl',msk2)):
        ef,ex,ec=eyermse(fus,lab,mm),eyermse(xgb,lab,mm),eyermse(cnn,lab,mm)
        rec[f'paired_{tag}']={'fusion_vs_xgb':wilc(ef,ex),'fusion_vs_cnn':wilc(ef,ec)}
    # 안 단위 제외 셀 분포
    per_eye=(msk&drop).sum(axis=1)
    rec['per_eye_dropped']={'mean':round(float(per_eye.mean()),2),'median':int(np.median(per_eye)),
                            'max':int(per_eye.max()),'eyes_with_zero':int((per_eye==0).sum())}
    out[split]=rec
    print(f'\n===== {split} (n={rec["n_eyes"]}, 제외 {rec["cells_dropped"]}/{rec["cells_full"]}셀 = {rec["dropped_pct"]}%) =====')
    print(f'{"model":8s} {"RMSE full":>10s} {"RMSE excl":>10s} {"Δ":>8s} {"MAE full":>10s} {"MAE excl":>10s} {"Δ":>8s}')
    for nm in ('xgb','cnn','fusion'):
        v=rec[nm]
        print(f'{nm:8s} {v["rmse_full"]:10.4f} {v["rmse_excl"]:10.4f} {v["d_rmse"]:+8.4f} '
              f'{v["mae_full"]:10.4f} {v["mae_excl"]:10.4f} {v["d_mae"]:+8.4f}')
    print('  안당 제외:', rec['per_eye_dropped'])
    for tag in ('full','excl'):
        for c in ('fusion_vs_xgb','fusion_vs_cnn'):
            d=rec[f'paired_{tag}'][c]
            print(f'  [{tag:4s}] {c:15s} p={d["p"]:.4g}  승 {d["wins"]}/{d["n"]}  medianΔ={d["median_delta"]:+.4f}')

(ROOT/'runs/neg1_mask_sensitivity.json').write_text(json.dumps(
    {'note':'-1 원본 지점을 마스크에서 제외한 재평가. 재학습 없음. W_FIXED=0.47.',
     'n_neg1_cells_52pt':n_neg1,'row_pairing':{'triple_exact':274,'unique_candidate':6},
     'integrity':{'nonneg1_value_mismatch':bad_a,'neg1_not_zero':bad_b},
     'splits':out}, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nsaved runs/neg1_mask_sensitivity.json')
