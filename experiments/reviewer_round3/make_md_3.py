#!/usr/bin/env python3
"""sixth_backbone.json -> md. 값은 읽기만 한다."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
J = json.loads((REPO / 'experiments/reviewer_round3/sixth_backbone.json').read_text(encoding='utf-8'))
OUT = REPO / 'experiments/reviewer_round3/sixth_backbone.md'


def fp(p):
    return f'{p:.3g}' if p >= 1e-4 else f'{p:.2e}'


g, bs, G = J['guards'], J['basis_shift'], J['gains']
R = {r['label']: r for r in J['rows_basisB']}
C = {(c['a'], c['b'], c['metric']): c for c in J['comparisons_basisB']}
_gl = {'oof_ensemble5_pooled_rmse_basisA': '5백본 앙상블 pooled RMSE (basis A)',
       'oof_ensemble5_fusion_pooled_rmse_basisA': '5백본 앙상블 + XGB fusion pooled RMSE (basis A)',
       'oof_summary_pooled_rmse_basisA': 'summary XGB pooled RMSE (basis A)'}

md = [
    '# (3) 6번째 CNN 백본 — 5→6 앙상블 이득 대 XGB 추가 이득',
    '',
    '§5 는 "gradient boosting 을 더해도 앙상블이 나아지지 않았으니 fusion 은 다양성이',
    '아니다"라고 논증한다. 그러나 5→6 의 앙상블 이득 포화를 통제하지 않는다. 같은 설정으로',
    '학습한 6번째 CNN 을 더했을 때의 이득이 유일하게 공정한 기준선이다.',
    '',
    '패스 B, out-of-fold. 생성: `experiments/reviewer_round3/sixth_backbone.py`.',
    '정본 `runs/` 와 원고 미변경.',
    '',
    '## 6번째 백본 학습',
    '',
    f"ResNet50. {J['resnet50_training']['note']}",
    f"실행: `{J['resnet50_training']['launcher']}` — `train.py` 는 수정하지 않고",
    '`build_image_backbone` 만 런타임에 확장했다. 산출물은',
    f"`{Path(J['resnet50_run']).relative_to(REPO)}` 로, 정본 `runs/` 밖이다.",
    '',
    '## 가드 (basis A = 원고와 같은 분모)',
    '',
    '| 항목 | 재현값 | 기준 | 일치 |',
    '|---|---|---|---|',
]
for k, v in g.items():
    md.append(f"| {_gl[k]} | {v['got']:.4f} | {v['want']} | {'OK' if v['ok'] else '불일치'} |")

md += [
    '',
    '## 두 기준선',
    '',
    'ResNet50 을 넣으면 키·마스크 교집합이 줄 수 있다. 5백본과 6백본을 like-for-like 로',
    '비교하려면 같은 분모여야 하므로 basis B 에서 잰다.',
    '',
    '| basis | 정의 | 안 | 마스크 셀 |',
    '|---|---|---|---|',
    f"| A | XGB ∩ 기존 5백본 (원고 분모) | {bs['basis_a_eyes']} | {bs['basis_a_cells']:,} |",
    f"| B | XGB ∩ 6백본 (ResNet50 포함) | {bs['basis_b_eyes']} | {bs['basis_b_cells']:,} |",
    '',
    f"basis B 에서 빠진 안 {bs['eyes_dropped']} 개, 셀 "
    f"{bs['basis_a_cells'] - bs['basis_b_cells']:,} 개. 아래 표는 전부 basis B 다.",
    '',
    '## 1. 구성별 성능 (basis B)',
    '',
    '| 구성 | pooled RMSE | pooled MAE | per-eye RMSE mean ± SD |',
    '|---|---|---|---|',
]
order = ['5백본 앙상블', '6백본 앙상블 (+ResNet50)', '5백본 앙상블 + XGB fusion',
         '6백본 앙상블 + XGB fusion', 'summary XGB 단독', 'ResNet50 단독'] + J['backbones5']
for lab in order:
    r = R[lab]
    md.append(f"| {lab} | {r['pooled_rmse']:.4f} | {r['pooled_mae']:.4f} | "
              f"{r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f} |")

md += ['',
       '## 2. 핵심 — 같은 출발점에서 두 이득 (basis B, pooled RMSE)',
       '',
       '| 5백본 앙상블에 더한 것 | pooled RMSE | 이득 (dB) |',
       '|---|---|---|',
       f"| — (5백본 앙상블) | {G['ens5_pooled_rmse']:.4f} | — |",
       f"| 6번째 CNN (ResNet50) | {R['6백본 앙상블 (+ResNet50)']['pooled_rmse']:.4f} | "
       f"{G['gain_add_6th_cnn']:+.4f} |",
       f"| XGB (late fusion, nested w) | {R['5백본 앙상블 + XGB fusion']['pooled_rmse']:.4f} | "
       f"{G['gain_add_xgb']:+.4f} |",
       f"| 6번째 CNN + XGB 둘 다 | {R['6백본 앙상블 + XGB fusion']['pooled_rmse']:.4f} | "
       f"{G['gain_add_6th_then_xgb']:+.4f} |",
       '',
       '음수 = 개선.',
       '',
       '## 3. 안 단위 검정 (basis B)',
       '',
       '`delta = per-eye M(a) − per-eye M(b)`, **음수 = a 우세**. CI 는 환자 군집 부트스트랩',
       f"95% percentile, seed {J['seed']}, N={J['n_boot']}. p 는 Wilcoxon signed-rank.",
       '',
       '| 지표 | a | b | Δ (dB) | 95% CI | Wilcoxon p | a 우세 안 |',
       '|---|---|---|---|---|---|---|']
for c in J['comparisons_basisB']:
    star = ' *' if (c['ci_lo'] < 0) == (c['ci_hi'] < 0) else ''
    md.append(f"| {c['metric'].upper()} | {c['a']} | {c['b']} | {c['delta']:+.4f}{star} | "
              f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {fp(c['p_wilcoxon'])} | "
              f"{c['n_a_better']} / {c['n_eyes']} |")

c6 = C[('6백본 앙상블', '5백본 앙상블', 'rmse')]
cx = C[('5백본+XGB fusion', '5백본 앙상블', 'rmse')]
ratio = (abs(G['gain_add_6th_cnn']) / abs(G['gain_add_xgb'])) if G['gain_add_xgb'] else float('nan')
md += ['',
       '`*` 는 CI 가 0 을 넘지 않는 행이다.',
       '',
       '## 4. 판정',
       '',
       f"6번째 CNN 을 더해 얻는 pooled RMSE 이득은 **{G['gain_add_6th_cnn']:+.4f} dB**,",
       f"XGB 를 더해 얻는 이득은 **{G['gain_add_xgb']:+.4f} dB** 다 "
       f"(6번째 CNN 이 XGB 이득의 {100*ratio:.0f}%).",
       '',
       f"안 단위로도 6백본은 5백본 대비 Δ = {c6['delta']:+.4f} dB "
       f"[{c6['ci_lo']:+.4f}, {c6['ci_hi']:+.4f}], p = {fp(c6['p_wilcoxon'])} 이고,",
       f"XGB fusion 은 Δ = {cx['delta']:+.4f} dB [{cx['ci_lo']:+.4f}, {cx['ci_hi']:+.4f}], "
       f"p = {fp(cx['p_wilcoxon'])} 다.",
       '',
       f"nested w (basis B): 5백본 fusion {J['nested_w_ens5_basisB']}, "
       f"6백본 fusion {J['nested_w_ens6_basisB']}.",
       '']
OUT.write_text('\n'.join(md) + '\n', encoding='utf-8')
print('->', OUT)
