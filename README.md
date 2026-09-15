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
  박동수(BPM), 박동 간격(IBI)과 변동계수, 진폭, 수축/이완 시간, 최대 수축/이완 속도, 수축 시작/종료
  시각, 그리고 감쇠 시간(time-to-decay T10/T50/T90 — 피크에서 진폭의 10%/50%/90%만큼 감쇠하는 데
  걸리는 시간, [PIV-MyoMonitor](https://doi.org/10.3389/fbioe.2024.1367141) 논문과 같은 정의)을
  계산합니다.
  세 번째 옵션으로 **`piv` 모드**(PIV, particle image velocimetry)도 있습니다 — 프레임을
  촘촘한 격자(interrogation window)로 나눠 각 창의 FFT 기반 교차상관으로 국소 변위 벡터를 구하는
  방식으로, PIVlab의 `piv_FFTmulti`와 같은 알고리즘 계열입니다(심장 오가노이드 수축력 분석 도구인
  [PIV-MyoMonitor](https://github.com/soahleelab/PIV-MyoMonitor)가 이 방식을 씁니다). 스칼라
  강도 차이 대신 실제 2D 변위 벡터장을 얻을 수 있어 가장 강한 박동 시점의 벡터장(화살표 + 크기
  히트맵)을 함께 시각화합니다. 다만 PIV는 교차상관이 걸릴 수 있는 명암 텍스처(반점/알갱이 무늬)가
  영상에 있어야 동작합니다 — 배경이 밋밋한 명시야 영상에서는 상관 피크가 모호해져 결과가 부정확해질
  수 있으므로, 창(window) 단위로 텍스처가 부족하면(명암 표준편차가 경험적 임계값 미만) 그 창은
  박동 신호 평균 계산에서 자동으로 제외됩니다(PIV-MyoMonitor 논문이 설명하는 "organoid 중심의 어두운
  영역 마스킹"과 같은 취지) — 응답의 `piv_n_low_texture_windows_masked` / `piv_n_windows_total`로
  몇 개나 제외됐는지 확인할 수 있습니다. 영상 전체(중앙값)가 부족하면 `piv_median_window_std`가
  낮게(대략 3 미만) 나오고 `piv_low_texture_warning: true`가 붙어 프론트엔드에도 경고 배너가
  뜹니다 — 이 경우 마스킹만으로는 부족할 수 있으니 `reference` 또는 `consecutive` 모드를 쓰세요.
  또한 PIVlab처럼 반복적인 윈도우 변형/다단계(multi-pass) 정제는 하지 않는 단일 패스 구현이라 큰
  변위나 미세 구조에는 PIVlab만큼 정확하지 않을 수 있습니다. **PIV는 `reference`/`consecutive`보다
  훨씬 느립니다** — 프레임마다 격자 전체를 FFT 교차상관하기 때문입니다. 대략 640×480/30fps/10초
  영상이면 수십 초, 그보다 크거나 긴 영상은 1분 이상 걸릴 수 있습니다. 느리면 `piv_window_size`를
  32에서 64 이상으로 키우거나 `piv_step`을 `piv_window_size`와 같은 값(창을 겹치지 않게)으로
  늘려서 창 개수를 줄이면 빨라집니다(공간 해상도는 낮아지지만 박동 검출용 신호 품질에는 보통 큰
  영향이 없습니다). **파라미터를 잘 모르겠으면 `piv_window_size=32`, `piv_step=8`로 시작하세요** —
  PIV-MyoMonitor 논문이 8~64px 윈도우와 4~16px 스텝 조합을 직접 비교해서 심장 오가노이드 모델에
  대해 실측으로 확정한 값입니다(윈도우 64px는 인접 창 사이 속도가 끊겨 보였고, 32px 이하에서
  매끈했습니다). 다만 스텝을 줄이면 창 개수가 늘어 계산 시간도 늘어난다는 점은 감안하세요.
- **관심영역(ROI) 선택**: 박동/칼슘 이미징 탭에서 영상을 고르면 첫 프레임 미리보기가 나타나고,
  마우스로 드래그해서 분석할 영역만 지정할 수 있습니다(지정하지 않으면 전체 프레임 사용). 배경/
  organoid의 어두운 중심부처럼 신호가 없거나 신뢰할 수 없는 영역을 미리 제외하면 특히 PIV 모드에서
  결과가 더 안정적입니다. 미리보기는 브라우저의 `<video>` 태그가 아니라 백엔드가 실제 분석에 쓰는
  것과 같은 OpenCV 디코더로 첫 프레임을 추출해서 보여줍니다 — 이 프로젝트의 합성 테스트 영상처럼
  일부 코덱(MPEG-4 Part 2 등)은 브라우저 `<video>`가 재생하지 못하는 경우가 있어서, 미리보기가
  실제 분석 가능 여부와 어긋나지 않도록 한 선택입니다.
- **칼슘 이미징 분석 (Calcium imaging)**: 형광 강도 트레이스를 (선택적으로 배경 형광을 뺀 뒤)
  ΔF/F0로 정규화하고 각 트랜지언트의 피크 시각, 진폭(평균 및 최대 ΔF/F0), **시작→피크(Ca²⁺
  방출), 피크→끝(재흡수), 트랜지언트 지속시간, CTD50/CTD90**(50% 높이 폭 / 90% 감쇠 폭 —
  BeatProfiler의 Contract50/90에 해당), rise time(10–90%), 지수 감쇠 시간상수(τ), 감쇠 시간
  (time-to-decay T10/T50/T90)을 계산합니다. 빈도는 분당 횟수와 Hz 두 단위로 제공됩니다. 배경
  형광 빼기는 세포 없는 영역을 직접 사각형으로 지정하거나(파란색), 가장 어두운 5% 픽셀을
  자동으로 배경으로 삼을 수 있습니다 — 세포가 화면을 꽉 채운 영상에서는 자동 방식이 어두운
  세포를 배경으로 오인할 수 있어 기본은 "안 함"입니다. 결과 그림에는 **픽셀 단위 최대 ΔF/F0
  지도**가 함께 나와 신호가 시야의 어디에서 나오는지 볼 수 있습니다.
- **형태 분석 (Morphology, 2D/3D)**: 2D는 대표 이미지(최대 강도 투영)를, 3D는 영상의 각 프레임을
  z-slice로 간주한 부피를 분할해 세포 개수·면적(2D)/부피(3D)·둘레·이심률을 계산합니다. 맞닿은
  세포는 distance-transform 기반 watershed로 자동 분리됩니다(끌 수도 있음). 세포/구조의 방향
  정렬도(alignment score, 0=무작위 ~ 1=완전 정렬)도 계산합니다 — 2D는 원형 순서 매개변수(circular
  order parameter), 3D는 각 객체의 관성텐서로 구한 주축을 이용한 nematic order parameter를 씁니다.
  옵션으로 **구조 텐서 기반 정렬도**(Fiji OrientationJ, Cardiotensor와 같은 방식)도 켤 수 있습니다 —
  세그멘테이션 없이 픽셀/voxel 단위로 국소 방향과 coherence(신뢰도)를 계산하므로, sarcomere 줄무늬나
  섬유 조직처럼 개별 세포로 나누기 애매한 텍스처의 정렬도를 볼 때 더 적합합니다.
- **그룹 통계 비교**: 같은 분석을 여러 영상(2개 이상의 그룹, 예: 대조군 vs 처리군, 또는 대조군/
  저용량/고용량)에 대해 돌린 뒤 각 지표를 비교합니다. **그룹이 2개면 Mann-Whitney U**, **3개
  이상이면 Kruskal-Wallis 전체검정 + Dunn's post-hoc**(그룹 쌍별 비교, 순위 기반이라 Kruskal-Wallis와
  같은 가정을 공유하며, 다중비교는 Bonferroni로 보정)를 씁니다. **모수 검정 옵션**(`test_family=
  parametric`)을 고르면 GraphPad Prism 방식 그대로 2그룹은 Welch's t-test(Student's t p도 함께),
  3그룹 이상은 one-way ANOVA + Tukey HSD post-hoc, 그리고 분산이 다를 때를 위해 Welch's ANOVA와
  쌍별 Welch t-test(Holm 보정; Prism의 Dunnett T3/Games-Howell 역할의 보수적 대용)를 나란히
  보고합니다. 그룹마다 Shapiro-Wilk 정규성 p-value도 나오므로 어느 검정을 쓸지 근거를 댈 수
  있습니다 — 그룹당 n이 8개 미만이면 정규성 검정 자체의 검정력이 낮아서 비모수가 기본값입니다.
  결과 그림은 GraphPad Prism 논문 그림 스타일입니다 — 지표마다 그룹별 막대(평균) + 개별 영상 점 +
  오차막대(**SEM 또는 SD 선택**, y축에 표시), 그 위에 **첫 번째 그룹(대조군/기준) 대비 각 그룹의
  p-value 브래킷**을 층층이 표시합니다(3개 이상이면 Dunn's Bonferroni 보정 p 또는 Tukey HSD p,
  2개면 Mann-Whitney U 또는 Welch t p; 모든 쌍의 p-value는 표와 `summary.json`에 있음). 한
  샘플(배치/웰)에서 여러 영상을 찍은 경우, 그룹별로 파일 순서대로
  배치/샘플 라벨을 넣으면 **선형 혼합효과 모델(LMM, `value ~ group + (1|sample)`)**로 샘플 ID를
  랜덤효과로 넣어 계산한 p-value를 그룹 쌍마다(그룹이 몇 개든 모든 쌍) 함께 제공합니다 — 같은
  샘플의 여러 측정을 독립 표본처럼 취급했을 때 생기는 pseudoreplication(거짓양성 증가) 문제를
  보정합니다. Lee et al., *Circulation Research*, 2025의 통계 방법(R `lmerTest`, REML, 랜덤
  절편)을 2그룹에서 N그룹으로 일반화한 접근입니다.
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

### 인터넷에 공개할 때 비밀번호로 보호하기

Render 같은 곳에 배포해서 외부에서 접근 가능한 URL이 생기면, 실험 데이터가 담긴 결과 파일이
누구나 볼 수 있는 상태가 됩니다. 환경변수 `APP_PASSWORD`를 설정하면 앱 전체(프론트엔드 화면,
API, `/results/*` 결과 파일 전부)가 HTTP Basic 인증으로 잠깁니다 — Render의 "Environment" 탭에서
`APP_PASSWORD`(필수)와 `APP_USERNAME`(선택, 기본값 `myteam`)을 추가하면 됩니다. 이 값을 설정하지
않으면(로컬 개발 시 기본 상태) 인증 없이 그대로 동작합니다. `GET /api/health`는 호스팅 플랫폼의
헬스체크를 위해 항상 인증 없이 열려 있습니다.

인증 로직을 FastAPI의 `Depends()`가 아니라 **미들웨어**로 구현한 이유가 있습니다 —
`app.mount("/", ...)`(프론트엔드)는 별도의 ASGI 서브 앱이라서, 개별 API 경로에만 거는 `Depends()`
인증은 이 mount를 그냥 지나쳐 갑니다. 즉 `Depends()` 방식으로는 API 호출은 막혀도 프론트엔드
화면은 그대로 인증 없이 노출됩니다. 미들웨어는 라우팅과 무관하게 모든 요청을 감싸므로 이 문제가
없습니다. (`/results/*` 결과 파일은 원래 별도 mount였지만 아래 Supabase 연동을 위해 일반 라우트로
바꿔서, 지금은 이 문제 자체가 없습니다 — 다만 방어적으로 미들웨어가 계속 감싸고 있습니다.)

HTTP Basic 인증은 자격증명을 base64로만 인코딩할 뿐 암호화하지 않으므로, **반드시 HTTPS URL로만
접속하세요**(Render가 기본 제공하는 `https://...onrender.com` 주소면 안전합니다 — `http://`로 직접
접속하면 안 됩니다).

### 결과 영구 저장하기 (Supabase, 선택사항)

Render 같은 곳의 기본/무료 플랜은 디스크가 **영구 저장이 아닙니다** — 재배포하거나 서비스가
재시작되면 그동안 `backend/storage/results/`에 쌓인 분석 결과(그래프, CSV, summary.json)가
사라집니다. [Supabase](https://supabase.com)(무료 플랜 있음)를 연결하면 분석할 때마다 결과가
Supabase Storage(파일)와 Postgres 테이블(요약 지표)에도 같이 저장되어, 재배포 이후에도
`/results/<id>/...` 링크가 계속 살아있습니다. 로컬 디스크에 파일이 있으면 그걸 먼저 쓰고, 없으면
(재배포 후 등) Supabase에서 가져오는 식으로 동작합니다 — 이 값들을 설정하지 않으면 지금까지처럼
로컬 디스크만 씁니다.

**1. Supabase 프로젝트 만들기**
1. [supabase.com](https://supabase.com) 에서 가입 → "New Project" 생성 (리전은 아무거나, 무료
   플랜으로 충분합니다).
2. 프로젝트 생성이 끝나면 왼쪽 메뉴 **Project Settings → API**로 가서 **Project URL**과
   **service_role** 키(⚠️ `anon` 키가 아니라 `service_role` 키입니다 — 이 앱은 서버에서만 이 키를
   쓰므로 안전합니다)를 복사해두세요.

**2. 결과 테이블 만들기**
왼쪽 메뉴 **SQL Editor**에서 새 쿼리를 열고 아래를 실행하세요.
```sql
create table if not exists analysis_results (
  id text primary key,
  analysis_type text not null,
  created_at timestamptz not null default now(),
  summary jsonb not null
);
```

**3. Storage 버킷 만들기**
왼쪽 메뉴 **Storage** → "New bucket" → 이름을 정확히 `results`로 입력 → **Public bucket 체크는
끄고(비공개로) 생성**하세요. 이 앱의 비밀번호 보호(위 섹션)가 결과 파일에도 적용되게 하려면
버킷이 비공개여야 합니다 — 서버가 `service_role` 키로만 파일을 읽고 쓰고, 사용자에게는 항상
이 앱의 `/results/...` 경로를 통해서만 전달하기 때문입니다.

**4. Render에 연결하기**
Render 대시보드 → 서비스 → **Environment** 탭에서 아래 두 개를 추가하고 저장(재배포)하세요.
- `SUPABASE_URL` = 1번에서 복사한 Project URL
- `SUPABASE_SERVICE_ROLE_KEY` = 1번에서 복사한 service_role 키

두 값이 모두 설정된 순간부터 새로 분석하는 결과부터 Supabase에도 같이 저장됩니다(그 이전에
로컬에만 저장됐던 결과는 소급 적용되지 않습니다). Supabase 업로드가 실패해도(네트워크 문제 등)
분석 자체는 항상 정상적으로 끝나고 로컬 결과는 그대로 보여집니다 — Supabase는 어디까지나 보조
저장소입니다.

## API

| Endpoint | 설명 |
|---|---|
| `POST /api/analyze/beating` | `file`(mp4), `fps_override`, `min_bpm_gap`(선택, 비우면 자동 추정), `prominence_frac`, `signal_mode`(`reference`/`consecutive`/`piv`), `reference_index`, `piv_window_size`(기본 32px), `piv_step`(기본 window_size/2), `roi_x`/`roi_y`/`roi_w`/`roi_h`(선택, 관심영역 픽셀 좌표) |
| `POST /api/analyze/calcium` | `file`(mp4), `fps_override`, `min_transients_per_min`, `prominence_frac`, `roi_x`/`roi_y`/`roi_w`/`roi_h`(선택), `background_mode`(`none`/`auto`/`manual`), `bg_x`/`bg_y`/`bg_w`/`bg_h`(`manual`일 때, 원본 프레임 좌표) |
| `POST /api/analyze/morphology` | `file`(mp4), `mode`(`2d`/`3d`), `min_object_size`, `separate_touching`, `separation_min_distance`, `compute_texture_alignment` |
| `POST /api/analyze/compare` | `analysis_type`(`beating`/`calcium`/`morphology`), `morphology_mode`, `test_family`(`nonparametric` 기본/`parametric`), `error_bar`(`sem` 기본/`sd`), 그룹마다 인덱스가 붙은 필드 `group_{i}_label`, `group_{i}_files`(다중), `group_{i}_batches`(선택, 줄바꿈/쉼표로 구분된 배치 라벨) — `i`는 0부터, 그룹 2개 이상(연속 번호일 필요는 없음) |
| `POST /api/analyze/batch` | `analysis_type`, `morphology_mode`, `files`(다중) — 영상별 결과를 CSV 하나로 |
| `POST /api/analyze/colocalization` | `channel_a_file`, `channel_b_file`, `label_a`, `label_b` |
| `POST /api/validate/agreement` | `file`(CSV), `column_a`, `column_b`, `label_a`, `label_b` |
| `POST /api/preview_frame` | `file`(mp4) — 첫 프레임을 PNG로 반환 (프론트엔드 ROI 선택기용) |

단일 분석 엔드포인트는 `{result_id, summary, urls: {plot, csv, summary}}` 형태의 JSON을,
`/compare`는 `{result_id, comparison, urls}` 형태를 반환합니다. `urls`는 `/results/...` 하위의
정적 파일 경로입니다. `/api/analyze/beating`과 `/api/analyze/calcium`은 ROI가 지정된 경우
`{"roi": {"x":.., "y":.., "w":.., "h":..}}`(실제 적용된, 프레임 경계에 맞춰 clamp된 값)를,
지정되지 않았으면 `"roi": null`을 함께 반환합니다.

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

### 합성 영상 벤치마크 (박동 검출 정확도)

`backend/benchmarks/beating_accuracy.py`는 **정답을 아는 합성 영상**(속도가 빠른 수축·느린 이완
파형으로 중심을 향해 수축하는 텍스처 조직, 박동 간격 ±8% 흔들림)을 참 BPM(30/60/90/120/180) ×
프레임속도(10/15/30 fps) × 센서 노이즈(표준편차 0/5/10/20) × 텍스처 대비(강/약) × 3회 반복 =
영상 360개로 만들고, 세 가지 신호 모드를 **기본 파라미터 그대로** 돌려서 박동 수가 정답과 얼마나
일치하는지 셉니다(`PYTHONPATH=. python benchmarks/beating_accuracy.py`, 결과는
`backend/benchmarks/results/`). 이건 실제 영상으로 Fiji/PIVlab과 비교하는 검증(아래 워크플로우)을
대체하지 못하고, "검출기 자체가 어떤 촬영 조건에서 무너지는가"만 답합니다 — 초점 흔들림, 부유물,
시야 밖 세포의 움직임, 불균일 조명 같은 실제 데이터의 문제는 여기 없습니다.

| 조건 | 모드 | 박동 수 정확 일치 | ±1개 이내 | BPM 오차 중앙값 |
|---|---|---|---|---|
| **30 fps, 노이즈 ≤10** (일반적인 촬영 조건, 30-180 BPM 전부) | reference | **98%** | **100%** | 0.14% |
| 30 fps, 노이즈 ≤10 | consecutive | **100%** | **100%** | 0.13% |
| 30 fps, 노이즈 ≤10, 텍스처 강함 | piv | 91% | 100% | 0.17% |
| 30 fps, 노이즈 ≤10, 텍스처 약함 | piv | 93% | 100% | 0.21% |
| 15 fps 이상, 노이즈 ≤10, 60 BPM 이상 | reference | 88% | 96% | 0.15% |
| 전체 그리드 (10 fps·노이즈 20 같은 극단 조건 포함) | reference / consecutive / piv | 61% / 84% / 69% | 68% / 92% / 84% | 0.3% / 0.2% / 0.3% |

이 벤치마크를 처음 돌렸을 때 30 BPM에서 특정 박동 시퀀스가 노이즈와 무관하게 5개 중 3개로만 잡히는
경우가 있었고, 원인은 자기상관 기반 주기 추정의 **주기 2배 오류**였습니다(박동 간격이 2.1초/1.9초로
번갈아 오면 "박동 두 개" 간격이 "박동 하나" 간격보다 자기상관에서 더 깨끗하게 맞아떨어져 4초를
주기로 잡고, 그러면 최소 피크 간격이 2초가 되어 1.9초 뒤의 진짜 박동을 억제). 부분조화(1/2, 1/3
주기) 검사를 넣어 고쳤고(`estimate_dominant_period_s`), 이 수정으로 그리드 전체에서 238개 실행이
정답에 가까워지고 5개(모두 노이즈 20 조건)가 나빠졌습니다 — 위 표는 수정 후 값입니다.

읽는 법과 실제로 확인된 약점:
- **일반 조건(30 fps, 보통 노이즈)에서는 reference/consecutive가 사실상 다 맞습니다.** 박동 수가
  맞은 경우 BPM 오차는 1% 미만입니다.
- **틀리는 방향**: reference 모드는 거의 항상 박동을 놓치는 쪽(과소검출)입니다. 과다검출(+1개)은
  주로 PIV 모드(일반 조건 90개 실행 중 7개)와 노이즈 20 조건에서 나오며, 크기는 대부분 +1개입니다.
- **노이즈가 심하면(표준편차 20) reference 모드가 무너집니다** — 대부분 박동 1개로 검출. 기준
  프레임과의 차이가 노이즈 바닥에 묻혀 피크 민감도(`prominence_frac`)가 못 잡는 것입니다. 화면이
  거칠게 지글거리는 영상이면 `consecutive` 모드(같은 조건에서 훨씬 안정적)를 쓰거나
  `prominence_frac`을 낮추세요. 이 조건의 자동 처리(프레임 사전 블러 등)는 향후 개선 항목입니다.
- **10 fps는 부족합니다.** 박동당 프레임이 몇 장 안 되면 피크가 뭉개집니다. 15 fps 이상, 가능하면
  30 fps로 찍으세요.
- **PIV 모드는 30 fps + 텍스처가 있을 때 reference에 근접**하고(±1개 이내 100%), 프레임속도가
  낮거나 노이즈가 크면 더 빨리 나빠집니다. 벡터장이 필요할 때만 쓰고 박동 수 자체는
  reference/consecutive 모드를 기준으로 삼는 게 안전합니다.

## 알려진 한계 및 향후 개선 방향

- **픽셀/복셀 단위**: 영상 자체에는 물리적 스케일(µm/px) 정보가 없으므로 모든 크기 지표는
  픽셀·복셀 단위입니다. 캘리브레이션 값을 입력받아 환산하는 기능은 아직 없습니다.
- **ROI 선택은 사각형만 지원합니다**: 박동/칼슘 이미징 탭에서 마우스로 드래그해 관심영역을
  지정할 수 있지만, 사각형 하나만 가능합니다(자유곡선 선택이나 여러 영역 동시 선택은 없음).
  형태 분석(morphology)에는 아직 ROI 기능이 없습니다.
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
- **그룹 비교는 각 그룹의 영상을 기본 파라미터로만 분석합니다**: 영상별로 `prominence_frac` 등을
  따로 맞추려면 단일 분석 엔드포인트로 개별 확인 후 사용하세요. 표본 수가 적을 때(그룹당 3개 이하)
  Mann-Whitney U/Kruskal-Wallis의 최소 p-value는 통계적으로 0.05 근처에서 막혀 "유의미한 차이"를
  통계적으로 확정하기 어려울 수 있습니다. Dunn's post-hoc의 Bonferroni 보정은 그룹 수가 많아질수록
  (쌍의 개수가 조합 폭발하므로) 보수적이 되어 개별 쌍 비교의 검정력이 떨어집니다 — 그룹이 5개를
  넘어가면 결과 해석에 유의하세요.
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
