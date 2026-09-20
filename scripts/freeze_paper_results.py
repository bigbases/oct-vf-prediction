"""Copy the JSON files the manuscript quotes into paper/results_frozen/.

The producers write into runs/ and experiments/, which this repository does not
publish (they also hold per-eye intermediates). This script takes the small
number of summary JSONs that tables and figures read, strips anything that
identifies an eye or a machine, and writes them to paper/results_frozen/ so a
reader can check a printed number without the withheld data.

Three transforms are applied, and only these three:

  * absolute paths are rewritten to repository-relative ones, and any home
    directory becomes /home/<user>;
  * in case_profile.json the pseudonym, the laterality and the examination date
    are removed. Each alone is harmless; together they identify a study eye.
    The metrics and the cohort position, which is what the manuscript cites,
    are kept unchanged;
  * the producers annotate their output in Korean. Those prose fields are
    replaced with English, string for string, from the PROSE table below.
    Numeric fields are not eligible: the table is applied to string values
    only, never to a key and never to a number.

Numbers are never rewritten. The producers are not touched either -- the
translation happens here, at freeze time, so that no number has to be
recomputed in order to publish a readable artefact.

Usage:  python scripts/freeze_paper_results.py --source <path to the working tree>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'results_frozen'

# The manuscript reports the pass-B numbers: the laterality correction in
# make_rnfl_flip.py (QUAD_SWAP) was applied upstream and every analysis re-run
# into the pass-B tree. The top-level runs/ directory of the working tree still
# holds the pass-A values (for example XGB out-of-fold RMSE 8.65 rather than
# 8.66), so the summary JSONs are taken from the pass-B tree, not from runs/.
# The experiments/ scripts already read pass B directly and have no pass-B copy.
PASS_B = 'experiments/laterality_qfix/step4_work/B'

# frozen filename -> (source-relative path, pass)
SOURCES = {
    'trivial_baselines.json':         (f'{PASS_B}/runs/trivial_baselines.json', 'B'),
    'final_model_comparison.json':    (f'{PASS_B}/runs/final_model_comparison.json', 'B'),
    'backbone_matrix_table.json':     (f'{PASS_B}/runs/backbone_matrix_table.json', 'B'),
    'aggregation_comparability.json': (f'{PASS_B}/runs/aggregation_comparability.json', 'B'),
    'ensemble_decisive.json':         (f'{PASS_B}/runs/ensemble_decisive.json', 'B'),
    'severity_region.json':           (f'{PASS_B}/runs/severity_region.json', 'B'),
    'skeleton_numbers.json':          (f'{PASS_B}/runs/skeleton_numbers.json', 'B'),
    'fusion_noninferiority.json':     (f'{PASS_B}/runs/fusion_noninferiority.json', 'B'),
    'fusion_consistency_matrix.json': (f'{PASS_B}/runs/fusion_consistency_matrix.json', 'B'),
    'patient_cluster_stats.json':     (f'{PASS_B}/runs/patient_cluster_stats.json', 'B'),
    'robustness_gains.json':          (f'{PASS_B}/runs/robustness_gains.json', 'B'),
    'multiseed_fusion.json':          (f'{PASS_B}/runs/multiseed_fusion.json', 'B'),
    'heldout_mae_reversal.json':      (f'{PASS_B}/runs/heldout_mae_reversal.json', 'B'),
    'fusion_sensitivity_trend.json':   (f'{PASS_B}/runs/fusion_sensitivity_trend.json', 'B'),
    'reliability_sensitivity.json':    (f'{PASS_B}/runs/reliability_sensitivity.json', 'B'),
    'stepsize_sensitivity.json':       (f'{PASS_B}/runs/stepsize_sensitivity.json', 'B'),
    'forest_rows.json':          ('experiments/forest_robustness/forest_rows.json', 'B'),
    'bias_by_bin.json':          ('experiments/bias_structure/bias_by_bin.json', 'B'),
    'bias_by_bin_refit.json':    ('experiments/bias_structure/bias_by_bin_refit.json', 'B'),
    'case_profile.json':         ('experiments/case_study/case_profile.json', 'A+B'),
    'resnet50_late_fusion.json': ('experiments/reviewer_round3/resnet50_late_fusion.json', 'B'),

    # Added 2026-09-17. These six back sentences that entered the manuscript
    # after this list was first written - the summary-branch ensemble, the
    # centred ensemble, the image-vs-summary head-to-head, the precision /
    # minimum-detectable-effect table, the error-structure decomposition and
    # the vertical cup-to-disc ablation. The experiments had all been run; the
    # results were simply never frozen, so check_numbers.py reported 42 numbers
    # as unsupported. They close 33 of them. All six declare pass B.
    #
    # sixth_backbone.json earns its place even though it lowers the MISS count
    # by nothing: its comparisons_basisB[1] (p 0.538, 120 of 240 eyes) is the
    # actual source of "not significant (p = 0.54, fusion better in 120 of 240
    # eyes)" in the ceiling subsection. Without it that sentence was passing
    # against the 0.54 in the weight column of the table just above it - a
    # coincidence, not evidence.
    'sixth_backbone.json':       ('experiments/reviewer_round3/sixth_backbone.json', 'B'),
    'symmetric_summary_ensemble.json': ('experiments/reviewer_round3/symmetric_summary_ensemble.json', 'B'),
    'centred_ensemble.json':     ('experiments/reviewer_round3/centred_ensemble.json', 'B'),
    'direct_test.json':          ('experiments/image_vs_summary/direct_test.json', 'B'),
    'fusion_precision_mde.json': (f'{PASS_B}/runs/fusion_precision_mde.json', 'B'),
    'fusion_error_structure.json': (f'{PASS_B}/runs/fusion_error_structure.json', 'B'),
    'ablation_vert_cd.json':     (f'{PASS_B}/runs/ablation_vert_cd.json', 'B'),

    # 반대편 눈 ablation (§4.3 "ten fellow-eye values ... 26 to 46").
    # 이 파일은 2026-09-17 **게이트 버그를 고치고 나서야** 필요해졌다. extract()
    # 가 LaTeX 이스케이프 `\%` 를 주석 시작으로 읽어 그 뒤를 통째로 버리던 탓에,
    # "(95\% CI $-0.089$ to $+0.238$" 의 CI 상한이 **한 번도 검사된 적이 없었다**.
    # 실험은 사전에 고정한 계획(원고 §4.2)대로 돌아 있었고 값은
    # primary.{delta,ci_lo,ci_hi,p_wilcoxon} 에 그대로 있다. 패스 B.
    'fellow_eye_ablation.json':  (f'{PASS_B}/runs/fellow_eye_ablation.json', 'B'),

    # 아래 둘은 2026-09-17 **우연 통과 훑기**로 필요해졌다. 두 실험 모두 돌아
    # 있었고 값도 정확한데 동결만 안 돼 있었다 — 그래서 원고 수치가 근거 풀의
    # 무관한 값에 우연히 맞아 [OK] 로 떠 있었다. MISS 50 때와 같은 구조다.
    #
    # neg1_mask_sensitivity: §4.6 "p moves from 0.026 to 1.9e-4 on the held-out
    #   set". 진짜 근거는 splits.test.paired_excl.fusion_vs_xgb.p = 1.886e-4.
    #   여태는 reliability_sensitivity.json 의 **vgg16 OOF fusion_vs_cnn**
    #   1.852e-4 에 맞아 통과했다 — 백본도 split 도 비교쌍도 전부 다른 값이다.
    # osflip_compare: §4.4 좌우 미러링 절제 네 수치(+0.147/p=0.79/-0.261/p=0.19).
    #   p 둘은 paired_wilcoxon_flip_vs_noflip 에 그대로 있고, delta 둘은
    #   flip - noflip 차라 DERIVED 로 등록한다.
    'neg1_mask_sensitivity.json': (f'{PASS_B}/runs/neg1_mask_sensitivity.json', 'B'),
    'osflip_compare.json':       (f'{PASS_B}/runs/osflip_compare.json', 'B'),
}

# keys dropped from case_profile.json: pseudonym + eye + date is re-identifying
CASE_DROP = {'pseudonym', 'eye', 'vf_date'}

_ABS = re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+(?:/hvf_project)?')
_HANGUL = re.compile(r'[\uac00-\ud7a3]')


# 세 번째 변환. 생산자가 남긴 한국어 주석을 영어로 치환한다 — 값이 아니라 설명만.
# 열쇠는 생산자가 쓴 문자열 그대로이고, 매칭은 부분일치가 아니라 완전일치다.
# 빠진 문장이 있으면 freeze 가 실패하지, 한국어가 조용히 남지 않는다.
PROSE = {
    "동일 예측을 pooled / per-eye mean±SD 두 방식으로 집계. 선행연구(Park·Shin·Abdullahi)는 per-eye mean±SD 방식이므로 그쪽 열로 비교할 것.":
        "The same predictions, aggregated two ways: pooled, and per-eye mean ± SD. Earlier studies (Park, Shin, Abdullahi) report the per-eye mean ± SD axis, so compare against that column.",
    "pooled >= per-eye mean (Jensen). 섞어 비교하면 본 연구가 부당하게 나빠 보인다.":
        "pooled >= per-eye mean (Jensen). Mixing the two axes makes this study look unfairly poor.",
    "IR-v2 / SS-OCT / test 305안":
        "IR-v2 / SS-OCT / 305 test eyes",
    "안별 RMSE → pooled 환산 가능":
        "per-eye RMSE can be converted to the pooled axis",
    "제안모델":
        "proposed model",
    "MAE는 pooled와 안별평균이 동일값이므로 환산 불필요 (본 연구 eye_mae_mean과 직접 비교)":
        "for MAE the pooled value and the per-eye mean coincide, so no conversion is needed (compare directly with this study eye_mae_mean)",
    "재학습 IR-v2 baseline":
        "retrained IR-v2 baseline",
    "재학습 InceptionV3 baseline":
        "retrained InceptionV3 baseline",
    "위 문헌값은 사용자 2차보고 기반. 1차출처 재확인 전 원고 인용 금지.":
        "The literature values above come from a second-hand report. Do not cite them in the manuscript until the primary sources have been checked.",
    "본 연구 OOF late fusion (IR-v2): pooled RMSE 8.069, 안별 RMSE 7.138 ± 3.77":
        "this study, OOF late fusion (IR-v2): pooled RMSE 8.069, per-eye RMSE 7.138 ± 3.77",
    "explained_by_aggregation_* 는 [pooled축 기준, 안별평균축 기준] 두 값. 집계방식이 설명하는 몫은 일부일 뿐 격차 대부분은 남는다 — 과대해석 금지.":
        "explained_by_aggregation_* holds two values: one on the pooled axis, one on the per-eye mean axis. The aggregation choice explains only part of the gap; most of it remains - do not over-read this.",
    "모든 안의 유효지점이 52로 동일하므로 pooled MAE = 안별 MAE 평균 (정확히 일치).":
        "every eye has the same 52 valid points, so pooled MAE equals the mean of the per-eye MAEs exactly.",
    "MAE로 보고된 선행연구(Abdullahi, Kihara PMAE)와의 비교에는 집계방식 보정이 아예 필요 없다. 이 경로는 변명 없이 그대로 성립.":
        "comparison with the studies that report MAE (Abdullahi, Kihara PMAE) needs no aggregation correction at all. That route holds as it stands, with nothing to excuse.",
    "구간·MIN_PTS·w·마스크·기울기는 fusion_sensitivity_trend / fusion_eval_common 에서 import. w 재적합 없음. delta = fusion RMSE - image branch RMSE (같은 변형 안에서 비교).":
        "the bins, MIN_PTS, w, the mask and the slope are imported from fusion_sensitivity_trend / fusion_eval_common. w is not refitted. delta = fusion RMSE - image branch RMSE (compared within the same variant).",
    "중심화 후 w 재적합(fec.fit_w, 101-그리드, 학습 폴드 RMSE 최소화)과 학습 폴드 기반 상수 추정을 적용한 재계산. 재학습 없음. delta = fusion RMSE - (중심화된) image branch RMSE.":
        "recomputed with w refitted after centring (fec.fit_w, 101-point grid, minimising RMSE on the training folds) and with the constant estimated on the training folds. No model is retrained. delta = fusion RMSE - (centred) image branch RMSE.",
    "fig_case_representative 대표안 프로파일. 값만 산출, 그림 없음.":
        "profile of the representative eye for fig_case_representative. Values only; no figure is drawn.",
    "prediction - measurement, 양수 = 과대예측":
        "prediction - measurement; positive = over-prediction",
    "(1) 중심화한 이미지 앙상블 단독의 성능":
        "(1) how the centred image ensemble performs on its own",
    "환자 군집 부트스트랩 95% percentile":
        "patient-clustered bootstrap, 95% percentile",
    "전역중심화 = 앙상블 예측에서 학습 폴드 잔차 평균 스칼라 하나를 뺀다. w 재적합 없음(보조 행 제외). 재학습 없음, 저장된 예측만 사용. per-eye SD 는 모집단 분모(ddof=0).":
        "global centring = subtract one scalar, the mean training-fold residual, from the ensemble prediction. w is not refitted (except in the auxiliary row). No model is retrained; only stored predictions are used. The per-eye SD uses the population denominator (ddof=0).",
    "fold k 의 상수 = 나머지 4 fold 의 마스크된 지점에서의 (앙상블 예측 - 실측) 평균":
        "the constant for fold k = the mean of (ensemble prediction - measurement) over the masked points of the other four folds",
    "summary (XGBoost 단독)":
        "summary (XGBoost alone)",
    "이미지 앙상블 (중심화 전)":
        "image ensemble (before centring)",
    "이미지 앙상블 (전역중심화, 학습폴드 상수)":
        "image ensemble (globally centred, training-fold constant)",
    "앙상블 fusion (nested w, 중심화 없음)":
        "ensemble fusion (nested w, no centring)",
    "[참고/oracle] 이미지 앙상블 (구간중심화, 학습폴드 상수)":
        "[reference/oracle] image ensemble (bin-wise centring, training-fold constant)",
    "[보조] 전역중심화 앙상블 + fusion (w 재적합)":
        "[auxiliary] globally centred ensemble + fusion (w refitted)",
    "전역중심화 앙상블":
        "globally centred ensemble",
    "앙상블 fusion":
        "ensemble fusion",
    "delta = per-eye RMSE(전역중심화 앙상블) - RMSE(앙상블 fusion); 음수 = 전역중심화 앙상블 우세":
        "delta = per-eye RMSE(globally centred ensemble) - RMSE(ensemble fusion); negative = the globally centred ensemble wins",
    "delta = per-eye MAE(전역중심화 앙상블) - MAE(앙상블 fusion); 음수 = 전역중심화 앙상블 우세":
        "delta = per-eye MAE(globally centred ensemble) - MAE(ensemble fusion); negative = the globally centred ensemble wins",
    "중심화 전 앙상블":
        "ensemble before centring",
    "delta = per-eye RMSE(전역중심화 앙상블) - RMSE(중심화 전 앙상블); 음수 = 전역중심화 앙상블 우세":
        "delta = per-eye RMSE(globally centred ensemble) - RMSE(ensemble before centring); negative = the globally centred ensemble wins",
    "전역중심화 후 fusion":
        "fusion after global centring",
    "delta = per-eye RMSE(전역중심화 후 fusion) - RMSE(전역중심화 앙상블); 음수 = 전역중심화 후 fusion 우세":
        "delta = per-eye RMSE(fusion after global centring) - RMSE(globally centred ensemble); negative = fusion after global centring wins",
    "image branch 단독 vs summary branch 단독. 패스 B, 재학습 없음, 저장된 예측만 사용. delta = mean_eyes[M(image) - M(summary)], 음수 = image 우세.":
        "image branch alone vs summary branch alone. Pass B; no model is retrained, only stored predictions are used. delta = mean_eyes[M(image) - M(summary)]; negative = the image branch wins.",
    "앙상블=5백본 CNN 평균. 결정적: fus>앙상블CNN(=정량이 강한 영상에 더하나)":
        "ensemble = the mean of the five CNN backbones. The decisive question: is fusion > the CNN ensemble, i.e. do the summary parameters add anything to an already strong image branch?",
    "반대편 눈 ablation. 사전등록 docs/fellow_eye_ablation_prereg.md. 주 종말점 = XGB(46) vs XGB(26) OOF RMSE. 융합(46) vs 영상단독은 정보 비대칭이므로 C5 주장에 쓰지 않는다.":
        "fellow-eye ablation, fixed in advance in the plan (manuscript §4.2). Primary endpoint = XGB(46) vs XGB(26) OOF RMSE. Fusion(46) against the image branch alone is an unequal-information comparison and is not used for the C5 claim.",
    "(C): XGB(46) ≈ XGB(26). 반대편 눈 요약값은 대상안 시야 예측에 정보를 더하지 않는다. 정량 브랜치의 천장이 피처 부족이 아니라 표현의 한계에서 온다.":
        "(C): XGB(46) ~= XGB(26). Fellow-eye summary values add no information to the prediction of the study eye field. The ceiling of the summary branch comes from the limits of the representation, not from a shortage of features.",
    "XGB는 백본 무관 동일 baseline":
        "the XGB baseline is the same one for every backbone",
    "Figure 6 forest plot 행 값. 패스 B. 재학습 없음, 저장된 예측만 사용. delta = mean_eyes[RMSE(fusion) - RMSE(summary)] dB, 음수 = fusion 우세.":
        "the row values of the Figure 6 forest plot. Pass B. No model is retrained; only stored predictions are used. delta = mean_eyes[RMSE(fusion) - RMSE(summary)] dB; negative = fusion wins.",
    "summary = XGB(25), fusion 도 XGB(25) 기반. w 는 이 arm 에서 재적합 (w=0.47 이면 pooled 8.0832)":
        "summary = XGB(25), and the fusion arm is built on XGB(25) as well. w is refitted within this arm (at w = 0.47 the pooled value is 8.0832)",
    "원본 -1 셀을 평가 마스크에서 제외":
        "the cells that were -1 in the source report are dropped from the evaluation mask",
    "안 단위 제외, 마스크는 그대로":
        "whole eyes are dropped; the mask is left as it is",
    "summary = XGB(46), fusion 도 XGB(46) 기반. w 는 이 arm 에서 재적합":
        "summary = XGB(46), and the fusion arm is built on XGB(46) as well. w is refitted within this arm",
    "w = 전체 OOF 적합, test 예측은 5fold 평균":
        "w fitted on the whole OOF set; the test prediction is the mean of the five folds",
    "global w — 대조용, 그림에 넣지 않는다":
        "global w - for contrast only, not drawn in the figure",
    "5백본 x {OOF,TEST} x {RMSE,MAE} = 20칸. w=RMSE기준, OOF는 nested, test는 전체 OOF 적합. 안 단위 paired Wilcoxon 양측.":
        "five backbones x {OOF, TEST} x {RMSE, MAE} = 20 cells. w is fitted on RMSE, nested for OOF and fitted on the whole OOF set for test. Per-eye paired Wilcoxon, two-sided.",
    "A=브랜치 균형 이동, B=pooled RMSE/MAE 비, C=꼬리 대 전형(안 단위 분포통계), D=평균 대 순위 검정 엇갈림. w_test_oracle 은 test 적합이므로 진단 전용이며 어떤 주장에도 쓰지 않는다.":
        "A = the shift in balance between the branches, B = the pooled RMSE/MAE ratio, C = tail against typical (per-eye distribution statistics), D = the mean and rank tests disagreeing. w_test_oracle is fitted on the test split, so it is diagnostic only and supports no claim.",
    "delta = eye-level(fusion) - eye-level(CNN). 음수=fusion 우세. 환자 클러스터 부트스트랩 10000회, seed 42. w=RMSE 기준, OOF는 nested.":
        "delta = eye-level(fusion) - eye-level(CNN); negative = fusion wins. Patient-clustered bootstrap, 10000 resamples, seed 42. w is fitted on RMSE, nested for OOF.",
    "test fold 정밀도. delta = 안 단위(fusion) - 안 단위(comparator), 음수=fusion 우세. MDE 는 양측 0.05, 검정력 0.80. primary 는 sd_oof 기반(사전 정밀도 진술). sd_test 기반은 참고. n_required_80_wilcoxon 은 t 기반 필요표본을 Wilcoxon 점근효율 3/pi 로 보정한 값.":
        "precision on the test fold. delta = per-eye(fusion) - per-eye(comparator); negative = fusion wins. The MDE is two-sided at 0.05 with power 0.80. The primary figure is the one based on sd_oof (the precision stated in advance); the sd_test version is for reference. n_required_80_wilcoxon is the t-based required sample size corrected by the Wilcoxon asymptotic efficiency 3/pi.",
    "감도 구간별 융합 이득의 단조 추세. delta = 안·구간 단위 (fusion RMSE - CNN RMSE). 기울기는 구간 지수 1..4 에 대한 OLS. 양수 = 저감도에서 fusion 이 더 유리. 주 검정은 is_primary=true 인 한 칸이며 나머지는 보조.":
        "the monotone trend of the fusion gain across sensitivity bins. delta = per eye and per bin (fusion RMSE - CNN RMSE). The slope is an OLS fit against bin index 1..4. Positive = fusion helps more at low sensitivity. The primary test is the single cell with is_primary=true; the rest are secondary.",
    "-1 원본 지점을 마스크에서 제외한 재평가. 재학습 없음. W_FIXED=0.47.":
        "re-evaluation with the points that were -1 in the source report dropped from the mask. No model is retrained. W_FIXED=0.47.",
    "OS 이미지 좌우반전(OD 정규화) 유무 대조. 하이퍼파라미터 동일, seed 42, num_workers 0. test 집계 = 5-fold 예측 평균(정본).":
        "with and without mirroring the OS images left-right (normalising to OD). Identical hyperparameters, seed 42, num_workers 0. Test aggregation = the mean of the five fold predictions (the canonical choice).",
    "환자 평균 검정(양안 상관 제거) + 환자 클러스터 부트스트랩. GEE 대체(statsmodels 부재).":
        "a test on patient means (removing the correlation between the two eyes of a patient) plus a patient-clustered bootstrap. Stands in for a GEE, statsmodels not being available.",
    "ResNet50 을 5백본 앙상블에 late fusion(스칼라 w) 으로 더한 값. 패스 B, out-of-fold, 저장된 예측만 사용(재학습 없음). delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.":
        "ResNet50 added to the five-backbone ensemble by late fusion (a scalar w). Pass B, out-of-fold, stored predictions only (no retraining). delta = per-eye M(a) - per-eye M(b); negative = a wins.",
    "폴드 k 의 w 는 나머지 4폴드의 유효 셀 pooled RMSE 를 최소화 (grid linspace(0,1,101)). 식 (2)/XGB 추가와 동일 규약.":
        "w for fold k minimises the pooled RMSE over the valid cells of the other four folds (grid linspace(0,1,101)). The same convention as Equation (2) and the XGB addition.",
    "basis B 는 ResNet50 의 키/마스크까지 교집합에 넣은 것이다.":
        "basis B is the one that intersects the keys and the mask of ResNet50 as well.",
    "5백본 앙상블":
        "five-backbone ensemble",
    "6백본 등가중 (+ResNet50)":
        "six backbones, equally weighted (+ResNet50)",
    "5백본 + ResNet50 late fusion (w 적합)":
        "five backbones + ResNet50 late fusion (w fitted)",
    "5백본 + XGB fusion (w 적합)":
        "five backbones + XGB fusion (w fitted)",
    "ResNet50 단독":
        "ResNet50 alone",
    "summary XGB 단독":
        "summary XGB alone",
    "6백본 등가중":
        "six backbones, equally weighted",
    "음수 = 개선. 세 이득 전부 같은 basis B, 같은 출발점(5백본 앙상블)에서 잰다.":
        "negative = improvement. All three gains are measured on the same basis B, from the same starting point (the five-backbone ensemble).",
    "제약 없는 최소제곱 최적 w 가 음수면, 적합된 w=0 은 격자 경계에 눌린 것이 아니라 \"ResNet50 을 조금이라도 섞으면 나빠진다\"는 뜻이다. 곡선은 모든 폴드에 같은 고정 w 를 적용한 OOF 값으로, 적합 절차가 아니라 진단이다.":
        "if the unconstrained least-squares optimum for w is negative, then a fitted w=0 is not pinned against the edge of the grid: it means that mixing in any ResNet50 at all makes things worse. The curve is the OOF value with one fixed w applied to every fold - a diagnostic, not a fitting procedure.",
    "ResNet50 5-fold 예측은 2026-08-31 sixth_backbone 작업에서 이미 저장된 것을 읽기만 했다. 이번 작업에서 학습한 모델은 없다.":
        "the ResNet50 five-fold predictions were only read back: they had already been stored by the sixth_backbone work of 2026-08-31. No model was trained for this analysis.",
    "§4 Robustness 산문 수치의 산출물 근거. 정렬 규약은 explore_fusion_gains.py 와 동일 (공통 키 = XGB ∩ 5백본, 마스크 = 전 분지 교집합).":
        "the artefact behind the prose numbers of §4 Robustness. The alignment convention is the one in explore_fusion_gains.py (common keys = XGB and the five backbones, mask = the intersection over all branches).",
    "w 를 전체 OOF 로 적합하고 같은 OOF 에서 평가 — development estimate.":
        "w fitted on the whole OOF set and evaluated on that same OOF set - a development estimate.",
    "평가 fold 를 뺀 4 fold 에서 w 적합 — 선택 낙관 제거.":
        "w fitted on the four folds that exclude the evaluation fold - removes selection optimism.",
    "적합 없이 두 분지를 동일 가중으로 평균.":
        "the two branches averaged with equal weight, nothing fitted.",
    "52 개 field 위치마다 w 를 따로 적합, nested.":
        "a separate w fitted at each of the 52 field locations, nested.",
    "위치별 비음수 최소제곱 스택(합 1 로 정규화), nested.":
        "a per-location non-negative least-squares stack (normalised to sum to one), nested.",
    "fusion(XGB + 5백본 평균 CNN), nested global-w":
        "fusion (XGB + the mean of the five CNN backbones), nested global w",
    "XGB 단독":
        "XGB alone",
    "5백본 평균 CNN":
        "the mean of the five CNN backbones",
    "안 단위 RMSE 의 대응표본 Wilcoxon. 음수 delta = fusion 우세.":
        "paired Wilcoxon on the per-eye RMSE. A negative delta = fusion wins.",
    "6번째 CNN(ResNet50) 추가 이득 대 XGB 추가 이득. 패스 B, out-of-fold. delta = per-eye M(a) - per-eye M(b), 음수 = a 우세.":
        "the gain from adding a sixth CNN (ResNet50) against the gain from adding XGB. Pass B, out-of-fold. delta = per-eye M(a) - per-eye M(b); negative = a wins.",
    "6백본 앙상블 (+ResNet50)":
        "six-backbone ensemble (+ResNet50)",
    "5백본 앙상블 + XGB fusion":
        "five-backbone ensemble + XGB fusion",
    "6백본 앙상블 + XGB fusion":
        "six-backbone ensemble + XGB fusion",
    "6백본 앙상블":
        "six-backbone ensemble",
    "5백본+XGB fusion":
        "five backbones + XGB fusion",
    "6백본+XGB fusion":
        "six backbones + XGB fusion",
    "음수 = 개선. 두 이득을 같은 basis B, 같은 출발점(5백본 앙상블)에서 잰다.":
        "negative = improvement. Both gains are measured on the same basis B, from the same starting point (the five-backbone ensemble).",
    "train.py 를 수정하지 않고 build_image_backbone 만 런타임에 확장했다. 설정은 정본 5백본과 동일(300 epoch, RMSprop, lr 1e-4, wd 1e-5, patience 100, early_stop mae, seed 42, 증강 없음, use_deviation False).":
        "train.py was not modified; only build_image_backbone was extended at runtime. The settings are those of the canonical five backbones (300 epochs, RMSprop, lr 1e-4, wd 1e-5, patience 100, early_stop mae, seed 42, no augmentation, use_deviation False).",
    "raw npz/csv에서 독립 재계산. 낮을수록 우수(RMSE/MAE, dB).":
        "recomputed independently from the raw npz/csv. Lower is better (RMSE/MAE, dB).",
    "cv_fold는 280안 전체에 환자 단위로 배정(split-first) 후 양 모달 존재 안만 필터(filter-second)":
        "cv_fold is assigned patient-wise over all 280 eyes (split-first), and only then are the eyes that have both modalities kept (filter-second)",
    "split=oof (OOF, CV fold별 val 예측 연결), IR-v2 + XGB. 축 1(clamp_both)·2(clamp_label)는 재학습 없는 사후 재집계, 축 3(clamp_retrain)은 정규화 타깃으로 재학습한 별도 런. Hasan 2025 normalised TS = floor 14 dB. raw 축이 본 연구의 정본.":
        "split=oof (OOF, the per-fold validation predictions concatenated), IR-v2 + XGB. Axes 1 (clamp_both) and 2 (clamp_label) are post-hoc re-aggregations with no retraining; axis 3 (clamp_retrain) is a separate run retrained on the normalised target. Hasan 2025 normalised TS = floor 14 dB. The raw axis is the canonical one for this study.",
    "예측·라벨 동시 사후 클램프 (학습 후 예측을 눌러줌)":
        "prediction and label clamped together after the fact (the trained predictions are pushed down)",
    "라벨만 사후 클램프, 예측은 원본 (모델 무수정)":
        "only the label is clamped after the fact, the prediction is left as produced (the model is untouched)",
    "정규화 타깃으로 재학습 — Hasan과 유일하게 동등":
        "retrained on the normalised target - the only setting equivalent to Hasan",
    "세 축은 하나의 값을 감싸는 구간이 아니다. 2026-08-08 XGB 실측에서 retrain(4.180) < both(4.363) < label(5.177) 로 재학습이 both보다 좋아 both를 상한이라 부를 수 없다. 각 축을 조작 내용과 함께 개별 보고할 것.":
        "the three axes are not an interval bracketing one value. In the XGB measurement of 2026-08-08, retrain(4.180) < both(4.363) < label(5.177): retraining beat both, so both cannot be called an upper bound. Report each axis separately, together with what it manipulates.",
    "(2) 요약 분지의 대칭적 강화":
        "(2) strengthening the summary branch symmetrically",
    "요약 분지만 재적합. 입력 26 파라미터·폴드·52 지점별 구조·결측 대치는 ablate_vert_cd.py 에서 import. XGB 는 XGB_KW 에서 seed 만 변경, RF/ET/Ridge/SVR 은 scikit-learn 기본값 그대로 (스케일링 없음). per-eye SD 는 모집단 분모(ddof=0).":
        "only the summary branch is refitted. The 26 input parameters, the folds, the per-point structure over the 52 locations and the missing-value imputation are imported from ablate_vert_cd.py. XGB changes only the seed in XGB_KW; RF/ET/Ridge/SVR keep the scikit-learn defaults as they are (no scaling). The per-eye SD uses the population denominator (ddof=0).",
    "Ridge / SVR 은 표준화 없이 원 스케일 26 입력에 적합했다. SVR 기본값(RBF, C=1, epsilon=0.1)은 이 스케일에서 사실상 상수 예측에 가깝다 — 다중 모델 앙상블 행을 읽을 때 감안할 것. 지시대로 조정하지 않았다.":
        "Ridge and SVR were fitted on the 26 inputs at their original scale, without standardisation. At this scale the SVR defaults (RBF, C=1, epsilon=0.1) amount to something close to a constant prediction - bear that in mind when reading the multi-model ensemble row. They were left unadjusted, as instructed.",
    "summary: 단일 XGBoost (seed 42)":
        "summary: single XGBoost (seed 42)",
    "summary 구성원: xgb (seed 43)":
        "summary member: xgb (seed 43)",
    "summary 구성원: xgb (seed 44)":
        "summary member: xgb (seed 44)",
    "summary 구성원: rf (seed 42)":
        "summary member: rf (seed 42)",
    "summary 구성원: et (seed 42)":
        "summary member: et (seed 42)",
    "summary 구성원: ridge (seed 42)":
        "summary member: ridge (seed 42)",
    "summary 구성원: svr (seed 42)":
        "summary member: svr (seed 42)",
    "summary: XGB seed 앙상블 (42/43/44)":
        "summary: XGB seed ensemble (42/43/44)",
    "summary: 다중 모델 앙상블 (XGB+RF+ET+Ridge+SVR)":
        "summary: multi-model ensemble (XGB+RF+ET+Ridge+SVR)",
    "image: 5백본 이미지 앙상블":
        "image: five-backbone image ensemble",
    "5백본 이미지 앙상블":
        "five-backbone image ensemble",
    "단일 XGBoost (seed 42)":
        "single XGBoost (seed 42)",
    "delta = per-eye M(image) - M(summary); 음수 = image 우세":
        "delta = per-eye M(image) - M(summary); negative = image wins",
    "XGB seed 앙상블 (42/43/44)":
        "XGB seed ensemble (42/43/44)",
    "다중 모델 앙상블 (5종)":
        "multi-model ensemble (five models)",
    "90d, 5-fold, seed42, IR-v2 대표, pooled RMSE/MAE(dB), 52-point masked":
        "90d, 5-fold, seed42, IR-v2 as the representative backbone, pooled RMSE/MAE(dB), 52-point masked",
    "grand-mean (상수)":
        "grand-mean (constant)",
}

_USED: set[str] = set()


def scrub_text(s: str) -> str:
    """Translate the prose, then remove the machine paths. Values only."""
    if s in PROSE:
        _USED.add(s)
        s = PROSE[s]
    elif _HANGUL.search(s):
        raise SystemExit(f'untranslated prose, add it to PROSE: {s!r}')
    return _ABS.sub('/home/<user>', s)


def scrub(node):
    if isinstance(node, dict):
        for k in node:
            if _HANGUL.search(k):
                # 키는 구조다. 치환하지 않고, 생산자에서 고치라고 멈춘다.
                raise SystemExit(f'Korean key, rename it in the producer: {k!r}')
        return {k: scrub(v) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str):
        return scrub_text(node)
    return node


def numbers(node, path='') -> dict:
    """Every numeric leaf, by JSON path. Booleans are not numbers here."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(numbers(v, f'{path}.{k}'))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(numbers(v, f'{path}[{i}]'))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[path] = repr(node)
    return out


def drop_case_identifiers(doc: dict) -> dict:
    """The second transform, applied before the other two so that the prose it
    removes never has to be translated."""
    case = doc.get('case')
    if isinstance(case, dict):
        doc['case'] = {k: v for k, v in case.items() if k not in CASE_DROP}
    prov = doc.get('provenance')
    if isinstance(prov, dict):
        doc['provenance'] = {
            'note': 'The eye is supplied on the command line to '
                    'scripts/make_case_heatmap.py; no identifier is stored here '
                    'or in the code. The selection rule was not recorded at the '
                    'time, and the eye sits in the favourable tail of the '
                    'cohort - see distribution_position below.',
        }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    help='working tree that holds runs/ and experiments/')
    ap.add_argument('--check', action='store_true',
                    help='compare against what is already frozen, write nothing')
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    missing, changed = [], []

    for name, (rel, _pass) in SOURCES.items():
        p = src / rel
        if not p.exists():
            missing.append(rel)
            continue
        doc = json.loads(p.read_text(encoding='utf-8'))
        if name == 'case_profile.json':
            doc = drop_case_identifiers(doc)
        before = numbers(doc)
        doc = scrub(doc)
        if numbers(doc) != before:
            raise SystemExit(f'{name}: a transform moved a number')
        text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + '\n'
        dst = OUT / name
        if args.check:
            if not dst.exists() or dst.read_text(encoding='utf-8') != text:
                changed.append(name)
        else:
            dst.write_text(text, encoding='utf-8')

    unused = set(PROSE) - _USED
    if unused and not missing:
        # 안 쓰인 항목은 생산자 문구가 바뀌었다는 뜻이다. 표가 낡으면 알려준다.
        for s in sorted(unused):
            print(f'unused PROSE entry: {s!r}')
        return 1

    for rel in missing:
        print(f'missing: {rel}')
    if args.check:
        for name in changed:
            print(f'drifted: {name}')
        print(f'{len(SOURCES) - len(missing) - len(changed)} / {len(SOURCES)} match')
        return 1 if (missing or changed) else 0
    print(f'froze {len(SOURCES) - len(missing)} / {len(SOURCES)} into {OUT}')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
