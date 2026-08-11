# Cardiomyocyte Analyzer

Fiji/ImageJ 없이 심근세포(cardiomyocyte) 2D/3D 현미경 영상(mp4)을 분석하는 가벼운 웹 앱입니다.
Python(FastAPI + OpenCV + scikit-image) 기반으로, Fiji 전체 배포판(수백 MB ~ 1GB 이상)을 설치하지
않고도 아래 세 가지 핵심 분석을 브라우저에서 바로 수행할 수 있습니다.

- **박동 분석 (Beating analysis)**: Fiji **MUSCLEMOTION** 플러그인(Sala et al., 2018, *Circulation
  Research*)과 동일한 원리로 픽셀 강도 변화를 수축 신호로 씁니다. 기본값(`reference` 모드)은 기준
  (이완기) 프레임 대비 각 프레임의 차이를 사용해 박동 1회당 피크 1개가 나오는 변위(displacement)형
  신호를 만듭니다 — 프레임 간 차이(`consecutive` 모드, 속도형 신호라 박동마다 피크 2개로 잡힐 수
  있음)도 비교용으로 선택할 수 있습니다. 스무딩 폭과 최소 피크 간격은 자기상관(autocorrelation)으로
  추정한 그 영상의 실제 박동 주기에 맞춰 자동으로 조정되어(예: `min_bpm_gap`을 비워두면 자동), 느린
  박동(hiPSC-CM, 대략 20-90 BPM)에서 프레임 단위 노이즈를 박동으로 오검출하는 문제를 줄입니다.
  박동수(BPM), 박동 간격(IBI)과 변동계수, 진폭, 수축/이완 시간, 최대 수축/이완 속도를 계산합니다.
  세 번째 옵션으로 **`piv` 모드**(PIV, particle image velocimetry)도 있습니다 — 프레임을
  촘촘한 격자(interrogation window)로 나눠 각 창의 FFT 기반 교차상관으로 국소 변위 벡터를 구하는
  방식으로, PIVlab의 `piv_FFTmulti`와 같은 알고리즘 계열입니다(심장 오가노이드 수축력 분석 도구인
  [PIV-MyoMonitor](https://github.com/soahleelab/PIV-MyoMonitor)가 이 방식을 씁니다). 스칼라
  강도 차이 대신 실제 2D 변위 벡터장을 얻을 수 있어 가장 강한 박동 시점의 벡터장(화살표 + 크기
  히트맵)을 함께 시각화합니다. 다만 PIV는 교차상관이 걸릴 수 있는 명암 텍스처(반점/알갱이 무늬)가
  영상에 있어야 동작합니다 — 배경이 밋밋한 명시야 영상에서는 상관 피크가 모호해져 결과가 부정확해질
  수 있으므로, `piv_median_window_std`가 낮으면(대략 3 미만) 응답에 `piv_low_texture_warning: true`가
  붙고 프론트엔드에도 경고 배너가 뜹니다 — 이 경고가 뜨면 `reference` 또는 `consecutive` 모드를
  쓰세요. 또한 PIVlab처럼 반복적인 윈도우 변형/다단계(multi-pass) 정제는 하지 않는 단일 패스
  구현이라 큰 변위나 미세 구조에는 PIVlab만큼 정확하지 않을 수 있습니다.
- **칼슘 이미징 분석 (Calcium imaging)**: 형광 강도 트레이스를 ΔF/F0로 정규화하고 각 트랜지언트의
  피크 시각, 진폭, rise time(10–90%), 지수 감쇠 시간상수(τ)를 계산합니다.
- **형태 분석 (Morphology, 2D/3D)**: 2D는 대표 이미지(최대 강도 투영)를, 3D는 영상의 각 프레임을
  z-slice로 간주한 부피를 분할해 세포 개수·면적(2D)/부피(3D)·둘레·이심률을 계산합니다. 맞닿은
  세포는 distance-transform 기반 watershed로 자동 분리됩니다(끌 수도 있음). 세포/구조의 방향
  정렬도(alignment score, 0=무작위 ~ 1=완전 정렬)도 계산합니다 — 2D는 원형 순서 매개변수(circular
  order parameter), 3D는 각 객체의 관성텐서로 구한 주축을 이용한 nematic order parameter를 씁니다.
  옵션으로 **구조 텐서 기반 정렬도**(Fiji OrientationJ, Cardiotensor와 같은 방식)도 켤 수 있습니다 —
  세그멘테이션 없이 픽셀/voxel 단위로 국소 방향과 coherence(신뢰도)를 계산하므로, sarcomere 줄무늬나
  섬유 조직처럼 개별 세포로 나누기 애매한 텍스처의 정렬도를 볼 때 더 적합합니다.
- **그룹 통계 비교**: 같은 분석을 여러 영상(그룹 A/B, 예: 대조군 vs 처리군)에 대해 돌린 뒤 각
  지표를 Mann-Whitney U 검정으로 비교합니다. 그룹별 평균±표준편차, p-value, 지표별 dot plot을
  제공합니다. 한 샘플(배치/웰)에서 여러 영상을 찍은 경우, 파일 순서대로 배치/샘플 라벨을 넣으면
  **선형 혼합효과 모델(LMM, `value ~ group + (1|sample)`)**로 샘플 ID를 랜덤효과로 넣어 계산한
  p-value도 함께 제공합니다 — 같은 샘플의 여러 측정을 독립 표본처럼 취급했을 때 생기는
  pseudoreplication(거짓양성 증가) 문제를 보정합니다. Lee et al., *Circulation Research*, 2025의
  통계 방법(R `lmerTest`, REML, 랜덤 절편)과 같은 접근입니다.
- **Colocalization 분석**: 멀티채널 형광 이미지 두 장(또는 영상의 최대 강도 투영)에서 Pearson
  상관계수, Manders M1/M2(각 채널 신호가 상대 채널과 겹치는 비율, Otsu 임계값 기준), Manders
  overlap coefficient를 계산합니다. RGB 병합 이미지 + 픽셀 강도 산점도를 함께 제공합니다.

모든 분석 결과는 요약 지표(JSON), 개별 이벤트별 표(CSV), 신호/세그멘테이션 플롯(PNG)으로 제공됩니다.

## 아키텍처

```
backend/   FastAPI 서버 + 분석 파이프라인 (OpenCV, numpy, scipy, scikit-image, matplotlib)
frontend/  바닐라 HTML/CSS/JS 단일 페이지 (빌드 도구 불필요, 백엔드가 정적 파일로 함께 서빙)
```

백엔드가 `/`, `/api/*`, `/results/*` 를 한 프로세스에서 모두 서빙하므로 별도의 프론트엔드
빌드/배포 없이 `uvicorn` 하나만 실행하면 됩니다.

## 빠른 시작

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

브라우저에서 `http://localhost:8000` 접속 → 탭 선택 → mp4 업로드 → 분석 시작.

### Docker

```bash
docker build -f backend/Dockerfile -t cardiomyocyte-analyzer .
docker run -p 8000:8000 cardiomyocyte-analyzer
```

## API

| Endpoint | 설명 |
|---|---|
| `POST /api/analyze/beating` | `file`(mp4), `fps_override`, `min_bpm_gap`(선택, 비우면 자동 추정), `prominence_frac`, `signal_mode`(`reference`/`consecutive`/`piv`), `reference_index`, `piv_window_size`(기본 32px), `piv_step`(기본 window_size/2) |
| `POST /api/analyze/calcium` | `file`(mp4), `fps_override`, `min_transients_per_min`, `prominence_frac` |
| `POST /api/analyze/morphology` | `file`(mp4), `mode`(`2d`/`3d`), `min_object_size`, `separate_touching`, `separation_min_distance`, `compute_texture_alignment` |
| `POST /api/analyze/compare` | `analysis_type`(`beating`/`calcium`/`morphology`), `morphology_mode`, `group_a_label`, `group_b_label`, `group_a_files`(다중), `group_b_files`(다중), `group_a_batches`(선택, 줄바꿈/쉼표로 구분된 배치 라벨), `group_b_batches` |
| `POST /api/analyze/batch` | `analysis_type`, `morphology_mode`, `files`(다중) — 영상별 결과를 CSV 하나로 |
| `POST /api/analyze/colocalization` | `channel_a_file`, `channel_b_file`, `label_a`, `label_b` |
| `POST /api/validate/agreement` | `file`(CSV), `column_a`, `column_b`, `label_a`, `label_b` |

단일 분석 엔드포인트는 `{result_id, summary, urls: {plot, csv, summary}}` 형태의 JSON을,
`/compare`는 `{result_id, comparison, urls}` 형태를 반환합니다. `urls`는 `/results/...` 하위의
정적 파일 경로입니다.

## 테스트

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```

합성 영상으로 각 분석 파이프라인 및 엔드포인트를 검증하는 단위 테스트가 포함되어 있습니다.
`tests/make_synthetic_video.py` 를 실행하면 데모용 mp4 3종(박동/칼슘/형태)을 생성할 수 있습니다.
`tests/fixtures/fiji_musclemotion_reference_signal.tsv`는 실제 사용자가 Fiji MUSCLEMOTION으로
뽑은 수축 신호이고, `test_fiji_regression.py`가 이 신호로 박동 검출이 실제 육안 확인치(~5-6박동,
~30 BPM)와 맞는지 회귀 테스트로 고정해둡니다.

## 알려진 한계 및 향후 개선 방향

- **픽셀/복셀 단위**: 영상 자체에는 물리적 스케일(µm/px) 정보가 없으므로 모든 크기 지표는
  픽셀·복셀 단위입니다. 캘리브레이션 값을 입력받아 환산하는 기능은 아직 없습니다.
- **ROI 선택**: 칼슘 이미징은 현재 전체 프레임 평균 강도를 사용합니다. 특정 세포/영역만 골라
  분석하는 ROI 드로잉 UI는 향후 추가가 필요합니다.
- **기준 프레임 자동 선택**: 박동 분석의 `reference` 모드는 기준 프레임을 자동으로(가장 넓은
  저모션 구간의 중앙 프레임) 고릅니다. 영상이 매우 짧거나 이완기 구간이 거의 없으면 잘못된 프레임이
  선택될 수 있으니, 그런 경우 `reference_index`를 직접 지정하세요.
- **정확도 검증**: 실제 사용자 영상 1건에 대해 Fiji MUSCLEMOTION 결과(5박동, ~31 BPM)와
  나란히 비교해 `prominence_frac`을 튜닝한 뒤 일치를 확인했습니다(아래 항목 참고). 다른 영상에도
  똑같이 잘 맞는다는 보장은 없으니, 중요한 실험에는 같은 영상을 Fiji로도 돌려 교차 확인하세요.
- **`prominence_frac`(피크 민감도)은 영상마다 수동으로 맞춰야 합니다**: 박동 주기(`min_bpm_gap`)는
  자기상관 분석으로 자동 추정되지만, "진짜 박동과 노이즈를 가르는 세기 임계값"을 완전 자동으로
  맞히려는 시도(Otsu 이진화, 정렬 후 최대 간격 분리)는 시뮬레이션 검증 결과 고정 비율 방식보다
  오히려 불안정해서 채택하지 않았습니다. 박동 수가 비정상적으로 많거나(노이즈를 박동으로 오검출)
  적게(약한 박동이 걸러짐) 나오면, `prominence_frac`을 0.05~0.2 사이에서 조금씩 조정하며 플롯을
  보고 실제 박동 개수와 맞는 값을 찾으세요. 기본값 0.15가 잘 맞지 않는 영상도 있습니다(예:
  진폭이 박동마다 크게 다른 노이즈 많은 저속 촬영본은 0.08~0.10 정도가 더 맞을 수 있음).
- **맞닿은 세포 분리 (watershed)**: 기본으로 켜져 있지만 `separation_min_distance`(기본 10
  px/voxel, 예상되는 세포 중심 간 최소 거리)를 세포 크기에 맞게 조정해야 정확합니다. 너무 작으면
  세포 하나가 여러 개로 잘못 쪼개지고, 너무 크면 맞닿은 세포가 다시 하나로 합쳐집니다.
- **정렬도(alignment_score, texture_alignment_score) 자동 검증 안 됨**: 합성 이미지(균일하게
  기울어진 줄무늬/타원 vs 무작위 방향)로는 검증했지만, 실제 Fiji의 OrientationJ 같은 도구 출력과
  나란히 비교한 적은 없습니다. `texture_alignment_score`(구조 텐서 방식)는 이미지 가장자리에서
  가우시안 필터의 경계 효과로 값이 부정확할 수 있으니, 관심 영역이 이미지 가장자리에 붙어있다면
  주의하세요.
- **그룹 비교는 2그룹만 지원**: 3그룹 이상 비교(예: 대조군/저용량/고용량)나 ANOVA 등은 아직 없고,
  `/api/analyze/compare`는 각 그룹의 모든 영상을 해당 분석의 기본 파라미터로만 돌립니다(영상별로
  `prominence_frac` 등을 따로 맞추려면 단일 분석 엔드포인트로 개별 확인 후 사용하세요). 표본 수가
  적을 때(그룹당 3개 이하) Mann-Whitney U의 최소 p-value는 통계적으로 0.05 근처에서 막혀 "유의미한
  차이"를 통계적으로 확정하기 어려울 수 있습니다.
- **LMM은 클러스터(샘플) 수가 적으면 불안정할 수 있습니다**: `lmm_converged`가 `True`여도 분산
  성분 추정이 경계값(0)에 가까우면 statsmodels가 ConvergenceWarning을 낼 수 있습니다 — 클러스터가
  양쪽 그룹 합쳐 6개 미만이면 LMM 결과보다 Mann-Whitney U를 우선 참고하고, 가능하면 클러스터(샘플)
  수를 늘리세요.
- **3D 해석**: "3D mp4"는 실제로는 z-slice 순서로 인코딩된 프레임 시퀀스라고 가정합니다. 실제
  현미경 장비가 다른 방식으로 3D 데이터를 mp4에 담는다면 `read_video_frames` 사용 부분을
  조정해야 할 수 있습니다.
- **Colocalization의 Manders M1/M2는 Otsu 임계값을 씁니다**: 전용 colocalization 툴이 흔히 쓰는
  Costes 자동 임계값보다 단순한 방식입니다. 두 채널이 각각 배경 대비 뚜렷하게 분리되는 경우엔
  괜찮지만, 논문에 쓸 정도로 엄밀한 비교가 필요하면 이 차이를 밝히거나 더 정교한 임계값 방법으로
  바꿔야 할 수 있습니다.
- 업로드 용량 상한(`MAX_UPLOAD_BYTES`, 기본 300MB)과 최대 프레임 수(`MAX_FRAMES`, 기본 3000)는
  `backend/app/config.py`에서 조정할 수 있습니다.
- **PIV 모드는 텍스처가 있는 영상에서만 신뢰할 수 있습니다**: 배경이 밋밋한 명시야(bright-field)
  영상(예: 저장소의 기본 `beating.mp4` 데모)에서 실제로 박동을 크게 과다검출하는 것을 확인했습니다
  (참 박동 6회/60 BPM인 영상에서 15회/152 BPM로 오검출). 원인은 알고리즘 버그가 아니라 인터로게이션
  창 내부에 상관 매칭에 쓸 명암 무늬가 거의 없어서였습니다(배경 패치 표준편차 ≈ 0.5). 같은 영상에
  합성 반점 텍스처를 더하자(`piv_median_window_std` ≈ 11.5) 정확히 6회/60 BPM으로 검출되었습니다.
  이 문제를 자동으로 걸러내기 위해 `piv_median_window_std` < 3(경험적 임계값, 두 케이스의 로그
  스케일 중간값)이면 `piv_low_texture_warning`을 띄웁니다 — 완전 자동 필터링이 아니라 경고이므로,
  실제 세포 영상에서는 이 임계값이 항상 맞는다는 보장이 없습니다. 중요한 분석에는 벡터장 시각화를
  눈으로 확인하세요.

## 라이선스 및 인용

MIT 라이선스([LICENSE](LICENSE))입니다. `CITATION.cff`에 인용 메타데이터가 있고(GitHub의
"Cite this repository" 버튼에 자동으로 뜹니다), `paper/`에는 JOSS(Journal of Open Source
Software) 제출용 논문 초안 틀이 있습니다 — 저자 정보와 검증 결과를 채워 넣으면 됩니다. 사용 전에
`LICENSE`와 `CITATION.cff`의 `[YOUR NAME]` 등 대괄호 자리표시자를 실제 정보로 바꿔주세요.

## 논문 게재를 위한 검증 워크플로우

이 도구를 학술지에 게재하려면(예: JOSS), Fiji 같은 기존 검증된 방법과의 정량적 일치도를 보여주는
것이 핵심입니다. 아래 순서로 진행하세요.

1. **배치 분석으로 이 도구의 값 뽑기**: `POST /api/analyze/batch`에 같은 분석 종류의 영상을
   여러 개 올리면 영상별 요약 지표가 담긴 CSV 하나로 나옵니다.
2. **Fiji로 같은 영상들 분석**: 기존에 쓰시던 Fiji 워크플로우(MUSCLEMOTION 등)로 같은 영상들을
   분석해서 기준값(정답값) CSV를 만드세요.
3. **두 CSV를 지표 하나 기준으로 짝지어 하나의 CSV로 합치기**: 각 행에 `이 도구 값`과
   `Fiji 값` 두 열이 있어야 합니다(영상 파일명으로 순서를 맞추면 됩니다).
4. **일치도 분석**: `POST /api/validate/agreement`에 그 CSV와 두 열 이름을 넘기면 Pearson/
   Spearman 상관계수, ICC(2,1) (절대적 일치도), Bland-Altman bias와 95% 일치 한계(limits of
   agreement)를 계산하고, 산점도 + Bland-Altman 플롯을 만들어줍니다. 이 결과를 그대로 논문의
   Validation 섹션/Figure로 쓸 수 있습니다.

n(영상 수)이 클수록, 여러 조건(세포주, 배양일 등)에 걸쳐 있을수록 통계적으로 설득력이 커집니다.
정확한 표본 크기 기준은 목표 저널의 가이드라인을 확인하세요.
