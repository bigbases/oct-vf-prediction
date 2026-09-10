#!/usr/bin/env python3
"""crop_sensitivity.json -> crop_sensitivity.md. 값을 새로 계산하지 않는다."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'crop_sensitivity.json').read_text(encoding='utf-8'))
P, EX, DIR = D['passes'], D['exclusions'], D['direction']
SETS = ['full', 'thr300', 'thr320', 'thr335']
NAMES = ['Inception-v3', 'IR-v2', 'VGG16', 'Xception', 'DenseNet121', 'ENSEMBLE']
L = []
w = L.append


w('# GCA crop 이탈안 민감도 분석')
w('')
w(f"생성: `experiments/gca_crop/crop_sensitivity.py` → `crop_sensitivity.json`, "
  f"이 문서는 `render_md.py` 가 그 json 만 읽어 렌더한다.")
w('')
w(D['note'])
w('')
w('재학습·w 재적합 없음. 저장된 예측에서 **안 단위 평가 마스크**만 줄였다. '
  'OOF fusion 은 원래 규약대로 fold별 nested w 와 전체 OOF 적합 global w 두 판을 '
  '모두 낸다(원고 표 `tab:backbones`/사다리는 global, `tab:ceiling`/§4.4 는 nested). '
  'held-out 은 원래부터 global w 하나뿐이다.')
w('')
w('패스 A 가 원고 숫자의 출처다(§4.4 기울기 +0.3612 / +0.2523 / +0.3225 가 정확히 '
  '재현된다). 패스 B(라테랄리티 수정본)는 강건성 대조로 함께 낸다.')
w('')

# ---------------------------------------------------------------- (1)
w('## (1) 이탈안 정의')
w('')
w('기준은 `min(crop_W, crop_H) < thr`. GCA crop 은 `_crop_gca_thickness` 가 '
  '파란 화소(얇은 GCIPL 구역)의 bounding box 로 잡으므로 크기가 자료에 의존한다. '
  '주군집은 W·H 가 모두 335 이상인 무리다.')
w('')
w('| split | n | thr300 이탈 | thr320 이탈 | thr335 이탈 |')
w('|---|---|---|---|---|')
for sp in ('OOF', 'TEST'):
    e = EX[sp]
    w(f"| {sp} | {e['thr335']['n_total']} | {e['thr300']['n_excluded']} | "
      f"{e['thr320']['n_excluded']} | {e['thr335']['n_excluded']} |")
w('')
w('임계를 300 → 320 → 335 로 옮겨도 OOF 이탈 수는 9 → 13 → 17 로 완만히 늘 뿐 '
  '급변점이 없다. held-out 은 세 임계 모두 같은 1안이다. 아래 본분석은 가장 '
  '보수적인 thr335 를 주 기준으로 쓰고, 300·320 도 함께 싣는다.')
w('')
w('thr335 이탈안 (OOF 17안):')
w('')
for k in EX['OOF']['thr335']['eyes']:
    w(f"- `{k}`")
w('')
w(f"thr335 이탈안 (held-out 1안): `{EX['TEST']['thr335']['eyes'][0]}`")
w('')

# ---------------------------------------------------------------- (2)
w('## (2) 이탈안 제외 후 재계산')
w('')
w('### 2-1. pooled RMSE / MAE (패스 A)')
w('')
w('summary 와 image 는 w 와 무관하다. fusion 은 global w 판이다(원고 사다리와 같은 규약).')
w('')
for sp in ('OOF', 'TEST'):
    w(f'**{sp}**')
    w('')
    w('| 백본 | set | n | RMSE summary | image | fusion | MAE summary | image | fusion |')
    w('|---|---|---|---|---|---|---|---|---|')
    for nm in NAMES:
        for st in SETS:
            c = P['A'][nm][sp][st]; g = c['global']
            r, m = g['pooled_rmse'], g['pooled_mae']
            w(f"| {nm} | {st} | {c['n_eyes']} | {r['summary']:.3f} | {r['image']:.3f} | "
              f"{r['fusion']:.3f} | {m['summary']:.3f} | {m['image']:.3f} | {m['fusion']:.3f} |")
    w('')
w('summary 열이 백본마다 같은 것은 정상이다 — XGB 가지는 백본과 무관하다.')
w('')

w('### 2-2. fusion 대 summary, Wilcoxon p (안별 RMSE, 백본 5종 + 앙상블)')
w('')
for sp in ('OOF', 'TEST'):
    w(f'**{sp}** (fusion = global w)')
    w('')
    w('| 백본 | ' + ' | '.join(SETS) + ' |')
    w('|---|---|---|---|---|')
    for nm in NAMES:
        row = [f"{P['A'][nm][sp][st]['global']['p_fusion_vs_summary_rmse']:.2e}" for st in SETS]
        w(f"| {nm} | " + ' | '.join(row) + ' |')
    w('')
w('OOF 는 전 백본이 이탈안 제외 후에도 p < 1e-9 를 유지한다. held-out 은 원래도 '
  '경계선인 백본(Inception-v3 5.3e-02, Xception 9.4e-02)이 있는데, 이탈 1안을 '
  '빼도 자릿수가 그대로다 — 뒤집히지도 구제되지도 않는다.')
w('')

w('### 2-3. fusion 대 image(앙상블), 안별 평균 차이와 p')
w('')
w('원고 `tab:ceiling` 규약(nested w)이다. 차이는 fusion − image 라 음수가 fusion 우세.')
w('')
w('| 패스 | split | set | n | image | fusion | 평균차 | p | fusion 우세 안 |')
w('|---|---|---|---|---|---|---|---|---|')
for pk in ('A', 'B'):
    for sp in ('OOF', 'TEST'):
        wm = 'nested' if sp == 'OOF' else 'global'
        for st in SETS:
            c = P[pk]['ENSEMBLE'][sp][st]; g = c[wm]
            w(f"| {pk} | {sp} | {st} | {c['n_eyes']} | {g['pooled_rmse']['image']:.3f} | "
              f"{g['pooled_rmse']['fusion']:.3f} | {g['mean_diff_fusion_minus_image_rmse']:+.3f} | "
              f"{g['p_fusion_vs_image_rmse']:.3f} | {g['n_eyes_fusion_better_than_image']}/{c['n_eyes']} |")
w('')
w('원고가 이미 "앙상블 영상 가지와 fusion 은 구별되지 않는다"(p=0.46)고 쓴 자리다. '
  '이탈안을 빼면 차이가 더 줄고 p 는 더 커진다 — 주장 방향이 바뀌지 않는다.')
w('')

w('### 2-4. §4.4 구간별 차이와 기울기')
w('')
w('구간은 실측 감도 [0,10) [10,20) [20,30) [30,∞), MIN_PTS=3, 안별 OLS 기울기, '
  '환자 군집 부트스트랩 CI(N=10000, seed=42), 일표본 Wilcoxon. '
  '구간차 = fusion 구간RMSE − image 구간RMSE (음수면 fusion 우세).')
w('')
for pk in ('A', 'B'):
    for nm in ('IR-v2', 'ENSEMBLE'):
        for sp in ('OOF', 'TEST'):
            w(f'**패스 {pk} · {nm} · {sp}**')
            w('')
            w('| set | 기울기 안수 | 기울기 | 95% CI | p | [0,10) | [10,20) | [20,30) | [30,∞) |')
            w('|---|---|---|---|---|---|---|---|---|')
            for st in SETS:
                t = P[pk][nm][sp][st]['trend']['original']
                b = t['per_bin']
                w(f"| {st} | {t['n_eyes_with_slope']} | {t['mean_slope']:+.4f} | "
                  f"[{t['slope_ci_lo']:+.3f}, {t['slope_ci_hi']:+.3f}] | {t['p_wilcoxon']:.2e} | "
                  + ' | '.join(f"{x['mean_delta']:+.2f} ({x['n_eyes']})" for x in b) + ' |')
            w('')
w('나머지 3개 백본의 OOF 기울기 (패스 A, original):')
w('')
w('| 백본 | ' + ' | '.join(SETS) + ' |')
w('|---|---|---|---|---|')
for nm in ('Inception-v3', 'VGG16', 'Xception', 'DenseNet121'):
    w(f"| {nm} | " + ' | '.join(
        f"{P['A'][nm]['OOF'][st]['trend']['original']['mean_slope']:+.4f}" for st in SETS) + ' |')
w('')
w('기울기는 이탈안을 빼면 **커진다**(IR-v2 OOF +0.361 → +0.408, 앙상블 +0.252 → +0.277). '
  '§4.4 의 주장은 "저감도 구간에서 fusion 이 영상보다 낫고 고감도로 갈수록 그 이점이 '
  '사라지거나 뒤집힌다"인데, 이탈안이 저감도에 몰려 있어 그들을 빼면 남은 저감도 안의 '
  'fusion 이점이 오히려 뚜렷해진다.')
w('')

# ---------------------------------------------------------------- (3)
w('## (3) 방향 확인 — 이탈 17안 vs 유지 223안')
w('')
w('세 비교를 함께 낸다. raw = 그대로 비교(Mann-Whitney). MD매칭 = 이탈안마다 MD 가 '
  '가장 가까운 유지안을 caliper 1.0 dB 안에서 1:1 로 붙인 뒤 쌍내 차(Wilcoxon). '
  'MD보정 = `rmse ~ 1 + MD + is_outlier` OLS 의 is_outlier 계수와 환자 군집 '
  '부트스트랩 95% CI. 차이는 이탈 − 유지라 양수면 이탈안이 더 나쁘다.')
w('')
for pk in ('A', 'B'):
    dd = DIR[pk]
    for nm in ('IR-v2', 'ENSEMBLE'):
        b = dd[nm]
        w(f"**패스 {pk} · {nm}** (thr{dd['threshold']}, 이탈 {b['n_outlier']} / 유지 {b['n_kept']}; "
          f"MD 중앙값 이탈 {b['md_median_outlier']:.2f} vs 유지 {b['md_median_kept']:.2f} dB; "
          f"MD 결측 이탈 {b['n_md_missing_outlier']} / 유지 {b['n_md_missing_kept']})")
        w('')
        w('| 가지 | raw 이탈 | raw 유지 | raw 차 | p | MD매칭 차 (n쌍) | p | MD보정 계수 [95% CI] |')
        w('|---|---|---|---|---|---|---|---|')
        for br in ('summary', 'image', 'fusion'):
            x = b['branches'][br]; r, m, a = x['raw'], x['md_matched'], x['md_adjusted']
            w(f"| {br} | {r['mean_outlier']:.2f} | {r['mean_kept']:.2f} | "
              f"{r['diff_mean']:+.2f} | {r['p_mannwhitney']:.4f} | "
              f"{m['mean_diff']:+.2f} ({m['n_pairs']}) | {m['p_wilcoxon']:.3f} | "
              f"{a['coef_outlier']:+.2f} [{a['ci_lo']:+.2f}, {a['ci_hi']:+.2f}] |")
        w('')
w('읽는 법: raw 로는 이탈안이 세 가지 모두에서 2.8~3.1 dB 더 나쁘다(p < 0.005). '
  '그런데 MD 를 맞추면 그 벌점이 사라지고 부호가 오히려 뒤집히며(이탈안이 더 낫다) '
  '유의하지 않다. MD 보정 계수도 0 근처이고 CI 가 0 을 크게 물고 있다. '
  '**summary 가지가 결정적이다** — summary 는 영상을 아예 쓰지 않으므로 crop 실패의 '
  '영향을 받을 수 없는데, raw 격차(+2.84)가 image·fusion 과 같은 크기다. '
  '즉 이탈안의 raw 열세는 crop 이 아니라 중증도로 설명된다.')
w('')

# ---------------------------------------------------------------- (4)
w('## (4) 판정')
w('')
w('**주 주장 (fusion > summary): 유지된다.**')
w('')
a = P['A']
w(f"- OOF, 전 임계, 백본 6종 전부 p < 1e-9. 예: IR-v2 "
  f"{a['IR-v2']['OOF']['full']['global']['p_fusion_vs_summary_rmse']:.2e} → "
  f"{a['IR-v2']['OOF']['thr335']['global']['p_fusion_vs_summary_rmse']:.2e}, 앙상블 "
  f"{a['ENSEMBLE']['OOF']['full']['global']['p_fusion_vs_summary_rmse']:.2e} → "
  f"{a['ENSEMBLE']['OOF']['thr335']['global']['p_fusion_vs_summary_rmse']:.2e}.")
w(f"- pooled RMSE 격차도 유지된다: 사다리 "
  f"{a['IR-v2']['OOF']['full']['global']['pooled_rmse']['summary']:.3f} / "
  f"{a['IR-v2']['OOF']['full']['global']['pooled_rmse']['image']:.3f} / "
  f"{a['IR-v2']['OOF']['full']['global']['pooled_rmse']['fusion']:.3f} → "
  f"{a['IR-v2']['OOF']['thr335']['global']['pooled_rmse']['summary']:.3f} / "
  f"{a['IR-v2']['OOF']['thr335']['global']['pooled_rmse']['image']:.3f} / "
  f"{a['IR-v2']['OOF']['thr335']['global']['pooled_rmse']['fusion']:.3f}. "
  f"세 값이 모두 내려가지만 간격은 그대로다.")
w('- held-out 은 이탈이 1안뿐이라 p 가 소수 셋째 자리에서만 움직인다. 원래 경계선인 '
  '백본은 여전히 경계선이다.')
w('')
w('**§4.4 기울기: 유지된다(오히려 강해진다).**')
w('')
for nm in ('IR-v2', 'ENSEMBLE'):
    for sp in ('OOF', 'TEST'):
        f0 = a[nm][sp]['full']['trend']['original']; f3 = a[nm][sp]['thr335']['trend']['original']
        w(f"- {nm} {sp}: {f0['mean_slope']:+.4f} (CI {f0['slope_ci_lo']:+.3f}~{f0['slope_ci_hi']:+.3f}, "
          f"p={f0['p_wilcoxon']:.2g}) → {f3['mean_slope']:+.4f} "
          f"(CI {f3['slope_ci_lo']:+.3f}~{f3['slope_ci_hi']:+.3f}, p={f3['p_wilcoxon']:.2g})")
w('')
w('패스 A 는 위 네 경우 모두 CI 하한이 0 위에 남는다(IR-v2 held-out 은 원래도 Wilcoxon '
  'p=0.14 로 유의하지 않았고, 제외 후에도 유의하지 않다 — 상태가 바뀌지 않는다). 패스 B 는 IR-v2 held-out 만 CI 하한이 원래부터 0 아래이고(-0.042), 제외 후에도 0 아래다(-0.023) — 이 역시 상태가 바뀌지 않는다.')
w('')
w('**중심화 반사실: 유지된다.**')
w('')
w('| 패스 | 백본 | split | set | original | bin_centered | global_centered |')
w('|---|---|---|---|---|---|---|')
for pk in ('A', 'B'):
    for nm in ('IR-v2', 'ENSEMBLE'):
        for sp in ('OOF', 'TEST'):
            for st in SETS:
                t = P[pk][nm][sp][st]['trend']
                w(f"| {pk} | {nm} | {sp} | {st} | " + ' | '.join(
                    f"{t[k]['mean_slope']:+.4f} (p={t[k]['p_wilcoxon']:.2g})"
                    for k in ('original', 'bin_centered', 'global_centered')) + ' |')
w('')
w('반사실의 요지는 "구간별 CNN 잔차 평균을 빼면 기울기가 뒤집히고(bin_centered), '
  '전역 평균 하나만 빼면 뒤집히지 않는다(global_centered)" 였다. 이탈안을 빼도 '
  'bin_centered 는 여전히 음수이고 유의하며, global_centered 는 여전히 0 근처이고 '
  '유의하지 않다. 세 판정이 모두 그대로다.')
w('')
w('**한계.** 이탈안은 17안뿐이고 MD 중앙값이 −16.9 dB 로 유지군(−3.8 dB)보다 훨씬 '
  '나쁘다. MD 매칭은 17쌍에 불과해 검정력이 낮다 — "crop 실패의 벌점이 없다"가 아니라 '
  '"이 표본에서 중증도를 넘어서는 벌점을 검출하지 못했다"가 정확한 진술이다. '
  '또한 이 분석은 평가에서만 뺐다. 이탈안은 학습에도 들어가 있었고, 그것까지 '
  '제거하려면 재학습이 필요하다 — 이번 지시 범위 밖이다.')
w('')

(HERE / 'crop_sensitivity.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print(f"저장: {HERE / 'crop_sensitivity.md'}  ({len(L)} 줄)")
