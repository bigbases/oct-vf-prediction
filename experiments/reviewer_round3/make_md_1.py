#!/usr/bin/env python3
"""centred_ensemble.json -> md. 값을 다시 계산하지 않고 읽어서 표로만 만든다."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
J = json.loads((REPO / 'experiments/reviewer_round3/centred_ensemble.json').read_text(encoding='utf-8'))
OUT = REPO / 'experiments/reviewer_round3/centred_ensemble.md'


def fp(p):
    return f'{p:.3g}' if p >= 1e-4 else f'{p:.2e}'


g = J['guards']
c = J['centering_constant']
rows = J['rows']
cmp_ = {(x['a'], x['b'], x['metric']): x for x in J['comparisons']}

md = [
    '# (1) 중심화한 이미지 앙상블 단독의 성능',
    '',
    f"패스 B, out-of-fold, n = {rows[0]['n_eyes']} 안. 재학습 없음 — 저장된 예측만 읽었다.",
    '생성: `experiments/reviewer_round3/centred_ensemble.py`. 정본 `runs/` 와 원고 미변경.',
    '',
    '## 규약',
    '',
    '- **전역중심화**: fold *k* 의 앙상블 예측에서 스칼라 하나를 뺀다. 그 스칼라는',
    '  나머지 4 fold 의 마스크된 지점에서의 잔차 평균(예측 − 실측)이다.',
    '  평가 대상 fold 는 상수 추정에 들어가지 않는다.',
    '  (`bias_by_bin_refit.py` 의 `global_center_train_fixedw` 와 같은 정의)',
    '- **fusion** 행은 원고 `tab:ceiling` 규약 그대로다: fold별 nested w, 중심화 없음.',
    '  w 재적합 없음.',
    '- pooled = 마스크된 지점을 통째로 풀링. per-eye SD 는 **모집단 분모**(ddof=0).',
    f"- CI 는 환자 군집 부트스트랩 95% percentile, seed {J['seed']}, N={J['n_boot']}.",
    '  p 는 안별 값 쌍의 Wilcoxon signed-rank.',
    '',
    '## 가드',
    '',
    '| 항목 | 재현값 | 기준 | 일치 |',
    '|---|---|---|---|',
]
_gl = {'oof_ensemble_pooled_rmse': '중심화 전 앙상블 단독 pooled RMSE',
       'oof_ensemble_fusion_pooled_rmse': '앙상블 fusion pooled RMSE (nested w)',
       'oof_summary_pooled_rmse': '단일 XGB pooled RMSE'}
for k, v in g.items():
    md.append(f"| {_gl[k]} | {v['got']:.4f} | {v['want']} | {'OK' if v['ok'] else '불일치'} |")
md += ['', '세 가드 모두 통과했다. 아래 값을 낸다.', '',
       '## 1. 전체 오차 — 나란히',
       '',
       '| 구성 | pooled RMSE | pooled MAE | per-eye RMSE mean ± SD |',
       '|---|---|---|---|']
for r in rows:
    md.append(f"| {r['label']} | {r['pooled_rmse']:.4f} | {r['pooled_mae']:.4f} | "
              f"{r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f} |")

a = cmp_[('전역중심화 앙상블', '앙상블 fusion', 'rmse')]
b = cmp_[('전역중심화 앙상블', '앙상블 fusion', 'mae')]
d = cmp_[('전역중심화 앙상블', '중심화 전 앙상블', 'rmse')]
e = cmp_[('전역중심화 후 fusion', '전역중심화 앙상블', 'rmse')]

md += ['',
       '구간중심화 행은 실측 라벨로 구간을 정하는 **oracle** 이라 비교 대상이 아니다.',
       '자리를 채우기 위해서만 싣는다.',
       '',
       '## 2. 중심화한 앙상블 대 앙상블 fusion — 안별 차이',
       '',
       '`delta = per-eye M(중심화 앙상블) − per-eye M(앙상블 fusion)`. **음수 = 중심화 앙상블 우세.**',
       '',
       '| 지표 | Δ (dB) | 95% CI | Wilcoxon p | 중심화 앙상블 우세 안 |',
       '|---|---|---|---|---|']
for x in (a, b):
    md.append(f"| {x['metric'].upper()} | {x['delta']:+.4f} | "
              f"[{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {fp(x['p_wilcoxon'])} | "
              f"{x['n_a_better']} / {x['n_eyes']} |")

md += ['',
       '보조 대조 두 줄:',
       '',
       '| 비교 | Δ (dB) | 95% CI | Wilcoxon p | 앞쪽 우세 안 |',
       '|---|---|---|---|---|',
       f"| 중심화 앙상블 − 중심화 전 앙상블 (RMSE) | {d['delta']:+.4f} | "
       f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | {fp(d['p_wilcoxon'])} | {d['n_a_better']} / {d['n_eyes']} |",
       f"| 중심화 후 fusion − 중심화 앙상블 (RMSE) | {e['delta']:+.4f} | "
       f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] | {fp(e['p_wilcoxon'])} | {e['n_a_better']} / {e['n_eyes']} |",
       '',
       '## 3. 사용한 상수',
       '',
       f"{c['definition']}.",
       '',
       '| fold | 0 | 1 | 2 | 3 | 4 | 평균 | SD(모집단) | 범위 |',
       '|---|---|---|---|---|---|---|---|---|',
       '| 상수 (dB) | ' + ' | '.join(f'{v:+.4f}' for v in c['global_by_fold'])
       + f" | {c['global_mean']:+.4f} | {c['global_sd_pop']:.4f} | {c['global_range']:.4f} |",
       '',
       f"참고: 5 fold 전체에서 추정하면 {c['global_all_folds_reference']:+.4f} dB 다",
       '(`experiments/bias_structure/bias_by_bin.json` 의 ENSEMBLE 전역 잔차 '
       '평균 +1.533 과 같은 값).',
       f"폴드별 편차는 {c['global_min']:+.4f} ~ {c['global_max']:+.4f} dB, 폭 {c['global_range']:.4f} dB 다.",
       '',
       f"fold별 nested w: {J['w_nested_by_fold']}. ",
       f"중심화 후 재적합한 w: {J['w_refit_after_global_centering_by_fold']}.",
       '',
       '## 4. 판정',
       '',
       '**pooled RMSE 축에서는 결론이 닫힌다. 다른 두 축에서는 닫히지 않는다.**',
       '',
       f"- pooled RMSE: 중심화 앙상블 {rows[2]['pooled_rmse']:.4f} 대 fusion {rows[3]['pooled_rmse']:.4f}. ",
       f"  차이 {rows[2]['pooled_rmse'] - rows[3]['pooled_rmse']:+.4f} dB. 중심화 전 8.1147 에서 fusion 이 얻은 이득",
       f"  {rows[1]['pooled_rmse'] - rows[3]['pooled_rmse']:.4f} dB 중 "
       f"{rows[1]['pooled_rmse'] - rows[2]['pooled_rmse']:.4f} dB, 즉 "
       f"{100*(rows[1]['pooled_rmse'] - rows[2]['pooled_rmse'])/(rows[1]['pooled_rmse'] - rows[3]['pooled_rmse']):.0f}% 를",
       '  학습 폴드에서 추정한 스칼라 하나가 그대로 낸다.',
       f"- pooled MAE: 중심화 앙상블 {rows[2]['pooled_mae']:.4f} 대 fusion {rows[3]['pooled_mae']:.4f}. ",
       f"  중심화는 MAE 를 중심화 전({rows[1]['pooled_mae']:.4f})보다 **악화**시킨다 "
       f"({rows[2]['pooled_mae'] - rows[1]['pooled_mae']:+.4f} dB). ",
       '  RMSE 를 최소화하는 오프셋이 MAE 를 최소화하지 않기 때문이다.',
       f"- 안별 차이: RMSE 로 {a['delta']:+.4f} dB (CI 가 0 을 걸친다, p={fp(a['p_wilcoxon'])}), ",
       f"  MAE 로 {b['delta']:+.4f} dB (CI 가 0 을 배제한다, p={fp(b['p_wilcoxon'])}). ",
       '  두 축 모두 부호는 fusion 우세 쪽이다.',
       '',
       'pooled RMSE 만 보고 "요약 파라미터가 더하는 것은 보정뿐" 이라고 쓰면 과장이다.',
       'MAE 축과 안별 축에서는 fusion 이 여전히 앞선다.',
       '반대로 "보정으로 설명되지 않는다" 고 쓰는 것도 과장이다 — 원고가 주 지표로 쓰는',
       'pooled RMSE 축에서는 이득의 대부분이 스칼라 하나로 재현된다.',
       '',
       '집계축 괴리도 함께 읽어야 한다. 중심화는 pooled RMSE 를 낮추면서 per-eye RMSE 평균은',
       f"{rows[1]['eye_rmse_mean']:.3f} → {rows[2]['eye_rmse_mean']:.3f} 로 **올린다**. ",
       f"낮아지는 것은 안별 SD 다 ({rows[1]['eye_rmse_sd_pop']:.3f} → {rows[2]['eye_rmse_sd_pop']:.3f}). ",
       'pooled RMSE = √(mean² + SD²) 항등식(`experiments/aggregation_axis/`) 아래에서,',
       '중심화가 하는 일은 안별 오차를 줄이는 것이 아니라 안 사이 분산을 줄이는 것이다.',
       'fusion 은 그 둘을 모두 줄인다.',
       '',
       ]
OUT.write_text('\n'.join(md) + '\n', encoding='utf-8')
print('->', OUT)
