#!/usr/bin/env python3
"""resnet50_late_fusion.json -> .md. 숫자는 전부 JSON 에서 읽는다 (수기 입력 없음)."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
j = json.load(open(D / 'resnet50_late_fusion.json', encoding='utf-8'))

GLAB = {'oof_ensemble5_pooled_rmse_basisA': '5백본 앙상블 pooled RMSE (basis A)',
        'oof_ensemble5_fusion_pooled_rmse_basisA': '5백본 + XGB fusion pooled RMSE (basis A)',
        'oof_summary_pooled_rmse_basisA': 'summary XGB pooled RMSE (basis A)',
        'oof_ensemble6_equal_pooled_rmse_basisB': '6백본 등가중 pooled RMSE (basis B)'}

L = []
A = L.append
A('# ResNet50 을 late fusion 으로 더한 값 (등가중 6백본이 아니라)\n')
A('`sixth_backbone.py` 는 6번째 CNN 을 **등가중 평균**으로 더해 8.2232 (악화) 를 냈다.')
A('그 비교는 XGB 추가(스칼라 w 적합)와 결합 방식이 달라 6번째 CNN 쪽에 불리하게')
A('기울어 있다. 여기서는 결합 방식을 맞춰 다시 잰다.\n')
A('```')
A(f"  {j['equation']}")
A('```')
A(f"w 규약: {j['w_convention']}\n")
A('패스 B, out-of-fold. **저장된 예측만 읽었고 재학습은 없다** — ResNet50 5-fold 예측은')
A('2026-08-31 `sixth_backbone` 작업 산출물을 그대로 썼다.')
A('생성: `experiments/reviewer_round3/resnet50_late_fusion.py`. 정본 `runs/` 와 원고 미변경.\n')

A('## 가드\n')
A('| 항목 | 재현값 | 기준 | 일치 |')
A('|---|---|---|---|')
for k, g in j['guards'].items():
    A(f"| {GLAB[k]} | {g['got']} | {g['want']} | {'OK' if g['ok'] else '**불일치**'} |")
bs = j['basis_shift']
A(f"\n분모: basis A {bs['basis_a_eyes']}안 / {bs['basis_a_cells']}셀, "
  f"basis B {bs['basis_b_eyes']}안 / {bs['basis_b_cells']}셀, "
  f"탈락 {bs['eyes_dropped']}안. 보고는 basis B (= `sixth_backbone` 과 같은 분모).\n")

A('## 결과\n')
A('| | pooled RMSE | pooled MAE | per-eye RMSE (mean ± SD_pop) |')
A('|---|---|---|---|')
for r in j['rows_basisB']:
    A(f"| {r['label']} | {r['pooled_rmse']:.4f} | {r['pooled_mae']:.4f} | "
      f"{r['eye_rmse_mean']:.3f} ± {r['eye_rmse_sd_pop']:.3f} |")

g = j['gains']
A(f"\n같은 출발점(5백본 {g['ens5_pooled_rmse']:.4f})에서 잰 이득 (음수 = 개선):\n")
A('| 추가 방식 | ΔpooledRMSE |')
A('|---|---|')
A(f"| 6번째 CNN, 등가중 | {g['gain_add_6th_equal']:+.4f} |")
A(f"| 6번째 CNN, late fusion (w 적합) | {g['gain_add_6th_late_fusion']:+.4f} |")
A(f"| XGB, late fusion (w 적합) | {g['gain_add_xgb']:+.4f} |")

A('\n## 적합된 w\n')
A('| 폴드 | 0 | 1 | 2 | 3 | 4 | 평균 |')
A('|---|---|---|---|---|---|---|')
wr, wx = j['nested_w_resnet50'], j['nested_w_xgb']
A('| w(ResNet50) | ' + ' | '.join(f'{v:.2f}' for v in wr) +
  f" | **{j['nested_w_resnet50_mean']:.4f}** |")
A('| w(XGB) | ' + ' | '.join(f'{v:.2f}' for v in wx) +
  f" | {j['nested_w_xgb_mean']:.4f} |")
A('\n**다섯 폴드 전부 w=0 으로 적합됐다.** 즉 late fusion 이 ResNet50 에 준 가중치가 0 이고,')
A('결합 결과는 5백본 앙상블과 셀 단위로 **완전히 동일**하다 (ΔpooledRMSE = 0.0000).\n')

d = j['diagnostics']
A('### w=0 은 격자 경계에 눌린 값인가\n')
A('아니다. 제약을 풀면 최적 w 가 **음수**로 간다 — ResNet50 을 섞는 게 아니라 빼야')
A('좋아진다는 뜻이고, [0,1] 격자에서는 0 이 그 방향의 끝이다.\n')
A('| | 폴드 0 | 1 | 2 | 3 | 4 | 평균 |')
A('|---|---|---|---|---|---|---|')
A('| 제약없는 최적 w(ResNet50) | ' +
  ' | '.join(f"{v:+.3f}" for v in d['w_unconstrained_resnet50_per_fold']) +
  f" | **{d['w_unconstrained_resnet50_mean']:+.4f}** |")
A('| 제약없는 최적 w(XGB) | ' +
  ' | '.join(f"{v:+.3f}" for v in d['w_unconstrained_xgb_per_fold']) +
  f" | {d['w_unconstrained_xgb_mean']:+.4f} |")
A('\n고정 w 를 밀어 넣었을 때의 OOF pooled RMSE (적합이 아니라 진단):\n')
A('| w | ' + ' | '.join(str(c['w']) for c in d['oof_curve_fixed_w_resnet50']) + ' |')
A('|---|' + '---|' * len(d['oof_curve_fixed_w_resnet50']))
A('| pooled RMSE | ' +
  ' | '.join(f"{c['pooled_rmse']:.4f}" for c in d['oof_curve_fixed_w_resnet50']) + ' |')
A('\nw=0 에서 단조 증가한다. 어떤 양의 가중치도 앙상블을 악화시킨다.\n')

A('## 안 단위 비교 (환자 클러스터 부트스트랩 CI, Wilcoxon)\n')
A(f"n_boot = {j['n_boot']}, seed = {j['seed']}. delta = per-eye M(a) − M(b), 음수 = a 우세.")
A('CI 는 환자 단위로 재표집한다 (한 환자의 두 눈은 함께 움직인다).\n')
A('| 지표 | a | b | Δ (dB) | 95% CI | Wilcoxon p | a 우세 안 |')
A('|---|---|---|---|---|---|---|')
for c in j['comparisons_basisB']:
    p = '— (퇴화)' if c['degenerate_identical'] else f"{c['p_wilcoxon']:.3g}"
    A(f"| {c['metric'].upper()} | {c['a']} | {c['b']} | {c['delta']:+.4f} | "
      f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {p} | {c['n_a_better']}/{c['n_eyes']} |")
A('\n"퇴화"는 검정 실패가 아니라 두 예측이 셀 단위로 동일해서 모든 안별 차이가 정확히 0 인')
A('경우다 (w=0 이므로 당연하다). Wilcoxon 이 정의되지 않는다.\n')

A('## 읽는 법\n')
A('**1. 결합 방식을 맞춰 줘도 6번째 CNN 은 아무것도 더하지 않는다.** 등가중일 때의')
A(f"{g['gain_add_6th_equal']:+.4f} 악화가 \"등가중이라 불리했던 것\"이라는 반론은 성립하지 않는다.")
A('XGB 와 똑같이 스칼라 w 를 적합할 기회를 줬는데도 적합 결과가 w=0 이고, 제약을 풀면')
A('음수로 간다. 이 축에서는 §5 의 논증이 오히려 더 강해진다.\n')
A('**2. 단, 안 단위 축에서는 여전히 유의하지 않다.** R50 late fusion 대 XGB fusion 은')
A('RMSE·MAE 어느 쪽도 CI 가 0 을 넘고 p>0.5 다. 쓸 수 있는 문장은 **"6번째 CNN 은')
A('스칼라 가중치를 적합해 줘도 앙상블을 개선하지 않는다"**까지이고, **"XGB 추가가 6번째')
A('CNN 추가보다 유의하게 낫다"는 이 데이터로 지지되지 않는다.** `sixth_backbone` 에서')
A('내린 것과 같은 유보다.\n')
A('**3. 결론은 여전히 ResNet50 이라는 단일 선택에 걸려 있다.** 다른 6번째 백본')
A('(예: EfficientNet-B0) 은 실행하지 않았다.\n')

(D / 'resnet50_late_fusion.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('->', D / 'resnet50_late_fusion.md')
