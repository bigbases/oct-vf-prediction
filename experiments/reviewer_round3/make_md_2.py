#!/usr/bin/env python3
"""symmetric_summary_ensemble.json -> md. 값은 읽기만 한다."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
J = json.loads((REPO / 'experiments/reviewer_round3/symmetric_summary_ensemble.json').read_text(encoding='utf-8'))
OUT = REPO / 'experiments/reviewer_round3/symmetric_summary_ensemble.md'


def fp(p):
    return f'{p:.3g}' if p >= 1e-4 else f'{p:.2e}'


g, rows = J['guards'], J['rows']
R = {r['label']: r for r in rows}
C = {(c['summary'], c['metric']): c for c in J['comparisons']}
single = 'summary: 단일 XGBoost (seed 42)'
seedens = 'summary: XGB seed 앙상블 (42/43/44)'
multi = 'summary: 다중 모델 앙상블 (XGB+RF+ET+Ridge+SVR)'
img = 'image: 5백본 이미지 앙상블'
c_single = C[('단일 XGBoost (seed 42)', 'rmse')]
c_seed = C[('XGB seed 앙상블 (42/43/44)', 'rmse')]
c_multi = C[('다중 모델 앙상블 (5종)', 'rmse')]

md = [
    '# (2) 요약 분지의 대칭적 강화',
    '',
    f"패스 B, out-of-fold, n = {R[single]['n_eyes']} 안. 요약 분지만 재적합했다.",
    '생성: `experiments/reviewer_round3/symmetric_summary_ensemble.py`. 정본 `runs/` 와 원고 미변경.',
    '',
    '## 규약',
    '',
    '- 입력 26 파라미터·폴드·52 지점별 개별 적합·결측 대치(학습 폴드 열 평균)는 전부',
    '  `step4_work/B/scripts/ablate_vert_cd.py` 에서 import 했다. 새로 정의하지 않았다.',
    '- **seed 앙상블**: 같은 XGBoost 를 `random_state` 42 / 43 / 44 로 적합해 예측 평균.',
    '  하이퍼파라미터는 `XGB_KW` 그대로, seed 만 바꿨다.',
    '- **다중 모델 앙상블**: XGBoost(seed 42) + RandomForest + ExtraTrees + Ridge + SVR 의 평균.',
    '  XGB 외 넷은 scikit-learn 기본값 그대로다. 조정하지 않았다.',
    '- 평가 기준선은 `fusion_eval_common.load_all()` (키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합)로,',
    '  −0.852 를 낸 `experiments/image_vs_summary/direct_test.py` 와 같은 분모다.',
    f"- CI 는 환자 군집 부트스트랩 95% percentile, seed {J['seed']}, N={J['n_boot']}.",
    '  p 는 안별 값 쌍의 Wilcoxon signed-rank. per-eye SD 는 모집단 분모(ddof=0).',
    '',
    '## 가드',
    '',
    '| 항목 | 재현값 | 기준 | 일치 |',
    '|---|---|---|---|',
    f"| 단일 XGBoost pooled RMSE | {g['single_xgb_pooled_rmse']['got']:.4f} | "
    f"{g['single_xgb_pooled_rmse']['want']} | {'OK' if g['single_xgb_pooled_rmse']['ok'] else '불일치'} |",
    f"| 단일 XGBoost pooled MAE | {g['single_xgb_pooled_mae']['got']:.4f} | "
    f"{g['single_xgb_pooled_mae']['want']} | {'OK' if g['single_xgb_pooled_mae']['ok'] else '불일치'} |",
    f"| 5백본 이미지 앙상블 pooled RMSE | {g['image_ensemble_pooled_rmse']['got']:.4f} | "
    f"{g['image_ensemble_pooled_rmse']['want']} | {'OK' if g['image_ensemble_pooled_rmse']['ok'] else '불일치'} |",
    '',
    f"재적합한 seed 42 XGB 는 저장된 정본 예측과 마스크된 셀에서 최대 "
    f"{g['refit_vs_stored_seed42_max_abs_cell_diff']:.3g} dB 차이다 "
    f"(저장본 pooled RMSE {g['refit_vs_stored_seed42_stored_pooled_rmse']:.4f}). "
    '재적합 절차가 정본을 재현한다.',
    '',
    '## 1. 요약 분지 구성원과 두 앙상블',
    '',
    '| 구성 | pooled RMSE | pooled MAE | per-eye RMSE mean ± SD |',
    '|---|---|---|---|',
]
for r in rows:
    md.append(f"| {r['label']} | {r['pooled_rmse']:.4f} | {r['pooled_mae']:.4f} | "
              f"{r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f} |")

md += ['',
       f"> {J['sklearn_defaults_caveat']}",
       '',
       '## 2. 5백본 이미지 앙상블 대 각 요약 구성',
       '',
       '`delta = per-eye M(image) − per-eye M(summary)`. **음수 = image 우세** (`direct_test.py` 와 같은 규약).',
       '',
       '| 지표 | 요약 쪽 상대 | Δ (dB) | 95% CI | Wilcoxon p | image 우세 안 |',
       '|---|---|---|---|---|---|']
for kind in ('rmse', 'mae'):
    for tag in ('단일 XGBoost (seed 42)', 'XGB seed 앙상블 (42/43/44)', '다중 모델 앙상블 (5종)'):
        c = C[(tag, kind)]
        star = ' *' if c['ci_lo'] < 0 and c['ci_hi'] < 0 else ''
        md.append(f"| {kind.upper()} | {tag} | {c['delta']:+.4f}{star} | "
                  f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {fp(c['p_wilcoxon'])} | "
                  f"{c['n_image_better']} / {c['n_eyes']} |")

d_seed = c_seed['delta'] - c_single['delta']
d_multi = c_multi['delta'] - c_single['delta']
md += ['',
       "`*` 는 CI 가 0 을 넘지 않는 행이다.",
       f"단일 XGBoost 행의 RMSE Δ = {c_single['delta']:+.4f} 는 `direct_test.md` 의 "
       f"{J['reference_delta_rmse_single_xgb']:+.3f} 를 재현한다.",
       '',
       '## 3. 판정 — 0.852 dB 중 앙상블 효과의 몫',
       '',
       '| 요약 쪽 상대 | RMSE Δ (dB) | 단일 XGB 대비 이동 |',
       '|---|---|---|',
       f"| 단일 XGBoost (원 head-to-head) | {c_single['delta']:+.4f} | — |",
       f"| XGB seed 앙상블 (42/43/44) | {c_seed['delta']:+.4f} | {d_seed:+.4f} |",
       f"| 다중 모델 앙상블 (5종) | {c_multi['delta']:+.4f} | {d_multi:+.4f} |",
       '',
       f"요약 분지를 seed 3개로 앙상블하면 image 우위가 {abs(c_single['delta']):.3f} → "
       f"{abs(c_seed['delta']):.3f} dB 로 {abs(d_seed):.3f} dB 줄어든다 — "
       f"원래 격차의 {100*abs(d_seed)/abs(c_single['delta']):.0f}% 다.",
       f"다중 모델 앙상블에서는 {abs(c_multi['delta']):.3f} dB 로 "
       f"{'줄어든다' if d_multi > 0 else '오히려 벌어진다'} "
       f"({100*abs(d_multi)/abs(c_single['delta']):.0f}%).",
       '',
       'pooled RMSE 축에서도 같은 방향이다: 단일 XGB '
       f"{R[single]['pooled_rmse']:.4f} → seed 앙상블 {R[seedens]['pooled_rmse']:.4f} "
       f"({R[seedens]['pooled_rmse'] - R[single]['pooled_rmse']:+.4f} dB), "
       f"다중 모델 앙상블 {R[multi]['pooled_rmse']:.4f} "
       f"({R[multi]['pooled_rmse'] - R[single]['pooled_rmse']:+.4f} dB). "
       f"이미지 앙상블은 {R[img]['pooled_rmse']:.4f} 다.",
       '']
OUT.write_text('\n'.join(md) + '\n', encoding='utf-8')
print('->', OUT)
