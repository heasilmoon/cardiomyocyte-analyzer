# Cardiomyocyte Analyzer

Fiji/ImageJ 없이 심근세포(cardiomyocyte) 2D/3D 현미경 영상(mp4)을 분석하는 가벼운 웹 앱입니다.
Python(FastAPI + OpenCV + scikit-image) 기반으로, Fiji 전체 배포판(수백 MB ~ 1GB 이상)을 설치하지
않고도 아래 세 가지 핵심 분석을 브라우저에서 바로 수행할 수 있습니다.

- **박동 분석 (Beating analysis)**: 밝은 시야(bright-field) 영상에서 프레임 간 픽셀 변화량으로
  수축/이완 신호를 추출하고(MUSCLEMOTION 계열 방식), 박동수(BPM), 박동 간격(IBI)과 변동계수,
  진폭, 수축/이완 시간을 계산합니다.
- **칼슘 이미징 분석 (Calcium imaging)**: 형광 강도 트레이스를 ΔF/F0로 정규화하고 각 트랜지언트의
  피크 시각, 진폭, rise time(10–90%), 지수 감쇠 시간상수(τ)를 계산합니다.
- **형태 분석 (Morphology, 2D/3D)**: 2D는 대표 이미지(최대 강도 투영)를 분할하여 세포 개수·면적·
  둘레·이심률을 계산하고, 3D는 영상의 각 프레임을 z-slice로 간주해 3차원 연결요소 기반 부피를
  계산합니다.

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
| `POST /api/analyze/beating` | `file`(mp4), `fps_override`, `min_bpm_gap`, `prominence_frac` |
| `POST /api/analyze/calcium` | `file`(mp4), `fps_override`, `min_transients_per_min`, `prominence_frac` |
| `POST /api/analyze/morphology` | `file`(mp4), `mode`(`2d`/`3d`), `min_object_size` |

모든 엔드포인트는 `{result_id, summary, urls: {plot, csv, summary}}` 형태의 JSON을 반환합니다.
`urls`는 `/results/...` 하위의 정적 파일 경로입니다.

## 테스트

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```

합성 영상으로 각 분석 파이프라인 및 엔드포인트를 검증하는 단위 테스트가 포함되어 있습니다.
`tests/make_synthetic_video.py` 를 실행하면 데모용 mp4 3종(박동/칼슘/형태)을 생성할 수 있습니다.

## 알려진 한계 및 향후 개선 방향

- **픽셀/복셀 단위**: 영상 자체에는 물리적 스케일(µm/px) 정보가 없으므로 모든 크기 지표는
  픽셀·복셀 단위입니다. 캘리브레이션 값을 입력받아 환산하는 기능은 아직 없습니다.
- **ROI 선택**: 칼슘 이미징은 현재 전체 프레임 평균 강도를 사용합니다. 특정 세포/영역만 골라
  분석하는 ROI 드로잉 UI는 향후 추가가 필요합니다.
- **맞닿은 세포 분리**: 형태 분석의 연결요소 분할은 서로 맞닿은 세포를 하나로 합칠 수 있습니다.
  watershed 기반 분리 등으로 개선할 수 있습니다.
- **3D 해석**: "3D mp4"는 실제로는 z-slice 순서로 인코딩된 프레임 시퀀스라고 가정합니다. 실제
  현미경 장비가 다른 방식으로 3D 데이터를 mp4에 담는다면 `read_video_frames` 사용 부분을
  조정해야 할 수 있습니다.
- 업로드 용량 상한(`MAX_UPLOAD_BYTES`, 기본 300MB)과 최대 프레임 수(`MAX_FRAMES`, 기본 3000)는
  `backend/app/config.py`에서 조정할 수 있습니다.
