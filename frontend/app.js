// Relative API base: works out of the box when the frontend is served by
// the same FastAPI app (see backend/app/main.py static mount). If you serve
// the frontend separately, set `window.CM_API_BASE = "http://localhost:8000"`
// before this script loads.
const API_BASE = window.CM_API_BASE || "";

function switchTab(tabName) {
  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

document.querySelectorAll("nav.tabs button").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function fieldLabel(key) {
  const labels = {
    signal_mode: "신호 방식",
    reference_frame_index: "기준 프레임 인덱스",
    estimated_period_s: "추정 박동 주기 (s)",
    smoothing_window_s: "스무딩 윈도우 (s)",
    min_bpm_gap_used: "적용된 최소 피크 간격 (bpm 상한)",
    n_beats: "박동 수",
    duration_s: "영상 길이 (s)",
    mean_bpm: "평균 박동수 (BPM)",
    mean_inter_beat_interval_s: "평균 박동 간격 (s)",
    ibi_std_s: "박동 간격 표준편차 (s)",
    ibi_cv_percent: "박동 간격 변동계수 (%)",
    mean_amplitude: "평균 진폭",
    amplitude_cv_percent: "진폭 변동계수 (%)",
    mean_contraction_time_s: "평균 수축 시간 (s)",
    mean_relaxation_time_s: "평균 이완 시간 (s)",
    mean_max_contraction_velocity: "평균 최대 수축 속도 (/s)",
    mean_max_relaxation_velocity: "평균 최대 이완 속도 (/s)",
    piv_median_window_std: "PIV 윈도우별 명암 표준편차 중앙값 (텍스처 지표)",
    piv_low_texture_warning: "PIV 텍스처 부족 경고",
    n_transients: "트랜지언트 수",
    mean_frequency_per_min: "평균 빈도 (회/분)",
    mean_inter_peak_interval_s: "평균 피크 간격 (s)",
    mean_amplitude_df_f0: "평균 진폭 (ΔF/F0)",
    mean_decay_tau_s: "평균 감쇠 시간상수 τ (s)",
    n_objects: "객체 수",
    mean_area_px: "평균 면적 (px²)",
    median_area_px: "중앙값 면적 (px²)",
    mean_eccentricity: "평균 이심률",
    total_covered_area_px: "총 커버 면적 (px²)",
    image_area_px: "이미지 전체 면적 (px²)",
    coverage_fraction: "커버 비율",
    mean_volume_voxels: "평균 부피 (voxel)",
    median_volume_voxels: "중앙값 부피 (voxel)",
    total_volume_voxels: "총 부피 (voxel)",
    stack_shape_zyx: "스택 크기 (Z,Y,X)",
    alignment_score: "정렬도 (0-1)",
    mean_orientation_deg: "평균 방향 (도)",
    alignment_score_3d: "정렬도 3D (0-1)",
    mean_direction_zyx: "평균 방향 벡터 (Z,Y,X)",
    n: "표본 수 (n)",
    pearson_r: "Pearson r",
    pearson_p: "Pearson p-value",
    spearman_r: "Spearman ρ",
    spearman_p: "Spearman p-value",
    icc_2_1: "ICC(2,1) 절대 일치도",
    bland_altman_bias: "Bland-Altman bias",
    bland_altman_sd_diff: "차이의 표준편차",
    bland_altman_loa_lower: "95% 일치 한계 (하한)",
    bland_altman_loa_upper: "95% 일치 한계 (상한)",
    regression_slope: "회귀 기울기",
    regression_intercept: "회귀 절편",
    regression_r_squared: "회귀 R²",
    texture_alignment_score: "구조 텐서 정렬도 (0-1)",
    texture_mean_orientation_deg: "구조 텐서 평균 방향 (도)",
    texture_mean_coherence: "평균 coherence (0-1)",
    texture_alignment_score_3d: "구조 텐서 정렬도 3D (0-1)",
    texture_mean_direction_zyx: "구조 텐서 평균 방향 벡터 (Z,Y,X)",
    texture_mean_fractional_anisotropy: "평균 비등방성 (FA, 0-1)",
    n_pixels: "픽셀 수",
    manders_overlap_coefficient: "Manders overlap coefficient",
    manders_m1: "Manders M1",
    manders_m2: "Manders M2",
    threshold_a: "채널 A 임계값 (Otsu)",
    threshold_b: "채널 B 임계값 (Otsu)",
    fraction_a_positive: "채널 A 양성 픽셀 비율",
    fraction_b_positive: "채널 B 양성 픽셀 비율",
    fraction_both_positive: "두 채널 모두 양성인 픽셀 비율",
  };
  return labels[key] || key;
}

function formatValue(v) {
  if (v === null || v === undefined) return "&mdash;";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(3);
  return String(v);
}

function renderResults(container, data) {
  const { summary, urls } = data;
  const rows = Object.entries(summary)
    .map(([k, v]) => `<tr><td>${fieldLabel(k)}</td><td>${formatValue(v)}</td></tr>`)
    .join("");

  const pivWarning = summary.piv_low_texture_warning
    ? `<div class="warning-banner">
        ⚠️ PIV 텍스처 부족 경고: 이 영상은 세포 표면의 명암 무늬(텍스처)가 부족해
        PIV(입자영상유속계) 분석 결과가 신뢰하기 어려울 수 있습니다.
        영상에 뚜렷한 반점/알갱이 무늬가 없다면 reference 또는 consecutive
        모드를 사용하는 것을 권장합니다.
      </div>`
    : "";

  container.innerHTML = `
    ${pivWarning}
    ${urls.plot ? `<img src="${API_BASE}${urls.plot}" alt="result plot" />` : ""}
    <table class="summary">${rows}</table>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
}

const beatingSignalMode = document.getElementById("beating-signal-mode");
const beatingPivFields = document.getElementById("beating-piv-fields");
if (beatingSignalMode) {
  const syncPivFieldsVisibility = () => {
    beatingPivFields.style.display = beatingSignalMode.value === "piv" ? "" : "none";
  };
  beatingSignalMode.addEventListener("change", syncPivFieldsVisibility);
  syncPivFieldsVisibility();
}

const compareAnalysisType = document.getElementById("compare-analysis-type");
const compareMorphologyModeField = document.getElementById("compare-morphology-mode-field");
if (compareAnalysisType) {
  const syncMorphologyModeVisibility = () => {
    compareMorphologyModeField.style.display = compareAnalysisType.value === "morphology" ? "" : "none";
  };
  compareAnalysisType.addEventListener("change", syncMorphologyModeVisibility);
  syncMorphologyModeVisibility();
}

function renderComparisonResults(container, data) {
  const { comparison, urls } = data;
  const hasLmm = comparison.metrics.some((m) => m.lmm_p_value !== undefined);
  const rows = comparison.metrics
    .map((m) => {
      const sig = m.p_value !== null && m.p_value < 0.05 ? " *" : "";
      const pText = m.p_value !== null ? m.p_value.toFixed(4) + sig : "&mdash;";
      let lmmCell = "";
      if (hasLmm) {
        if (m.lmm_p_value !== undefined) {
          const lmmSig = m.lmm_p_value < 0.05 ? " *" : "";
          lmmCell = `<td>${m.lmm_p_value.toFixed(4)}${lmmSig} (${m.lmm_n_clusters} clusters)</td>`;
        } else {
          lmmCell = "<td>&mdash;</td>";
        }
      }
      return `<tr>
        <td>${fieldLabel(m.metric)}</td>
        <td>${formatValue(m.mean_a)} &plusmn; ${formatValue(m.std_a)} (n=${m.n_a})</td>
        <td>${formatValue(m.mean_b)} &plusmn; ${formatValue(m.std_b)} (n=${m.n_b})</td>
        <td>${pText}</td>
        ${lmmCell}
      </tr>`;
    })
    .join("");

  container.innerHTML = `
    ${urls.plot ? `<img src="${API_BASE}${urls.plot}" alt="comparison plot" />` : ""}
    <table class="summary compare-table">
      <thead><tr><th>지표</th><th>${comparison.label_a} (n=${comparison.n_videos_a})</th><th>${comparison.label_b} (n=${comparison.n_videos_b})</th><th>Mann-Whitney U p-value</th>${hasLmm ? "<th>LMM p-value (샘플 보정)</th>" : ""}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
}

const compareForm = document.getElementById("compare-form");
if (compareForm) {
  compareForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const statusEl = compareForm.parentElement.querySelector(".status");
    const resultsEl = compareForm.parentElement.querySelector(".results");
    const submitBtn = compareForm.querySelector("button[type=submit]");

    const formData = new FormData(compareForm);
    submitBtn.disabled = true;
    statusEl.textContent = "그룹 비교 분석 중입니다... 영상 수에 따라 시간이 걸릴 수 있습니다.";
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}/api/analyze/compare`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `요청 실패 (HTTP ${res.status})`);
      }
      const data = await res.json();
      statusEl.textContent = "완료되었습니다.";
      renderComparisonResults(resultsEl, data);
    } catch (err) {
      statusEl.textContent = `오류: ${err.message}`;
      statusEl.classList.add("error");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

const batchAnalysisType = document.getElementById("batch-analysis-type");
const batchMorphologyModeField = document.getElementById("batch-morphology-mode-field");
if (batchAnalysisType) {
  const syncBatchMorphologyModeVisibility = () => {
    batchMorphologyModeField.style.display = batchAnalysisType.value === "morphology" ? "" : "none";
  };
  batchAnalysisType.addEventListener("change", syncBatchMorphologyModeVisibility);
  syncBatchMorphologyModeVisibility();
}

function renderBatchResults(container, data) {
  const { summaries, urls } = data;
  if (!summaries.length) {
    container.innerHTML = "<p>결과가 없습니다.</p>";
    return;
  }
  const columns = Object.keys(summaries[0]);
  const header = columns.map((c) => `<th>${c === "filename" ? "파일명" : fieldLabel(c)}</th>`).join("");
  const rows = summaries
    .map((s) => `<tr>${columns.map((c) => `<td>${formatValue(s[c])}</td>`).join("")}</tr>`)
    .join("");

  container.innerHTML = `
    <div style="overflow-x:auto;">
      <table class="summary compare-table">
        <thead><tr>${header}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
}

const batchForm = document.getElementById("batch-form");
if (batchForm) {
  batchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const statusEl = document.getElementById("batch-status");
    const resultsEl = document.getElementById("batch-results");
    const submitBtn = batchForm.querySelector("button[type=submit]");

    const formData = new FormData(batchForm);
    submitBtn.disabled = true;
    statusEl.textContent = "배치 분석 중입니다... 영상 수에 따라 시간이 걸릴 수 있습니다.";
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}/api/analyze/batch`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `요청 실패 (HTTP ${res.status})`);
      }
      const data = await res.json();
      statusEl.textContent = `완료되었습니다 (영상 ${data.n_videos}개).`;
      renderBatchResults(resultsEl, data);
    } catch (err) {
      statusEl.textContent = `오류: ${err.message}`;
      statusEl.classList.add("error");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

function renderAgreementResults(container, data) {
  const { stats, urls } = data;
  const rows = Object.entries(stats)
    .map(([k, v]) => `<tr><td>${fieldLabel(k)}</td><td>${formatValue(v)}</td></tr>`)
    .join("");

  container.innerHTML = `
    ${urls.plot ? `<img src="${API_BASE}${urls.plot}" alt="agreement plot" />` : ""}
    <table class="summary">${rows}</table>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
}

const agreementForm = document.getElementById("agreement-form");
if (agreementForm) {
  agreementForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const statusEl = document.getElementById("agreement-status");
    const resultsEl = document.getElementById("agreement-results");
    const submitBtn = agreementForm.querySelector("button[type=submit]");

    const formData = new FormData(agreementForm);
    submitBtn.disabled = true;
    statusEl.textContent = "일치도 분석 중입니다...";
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}/api/validate/agreement`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `요청 실패 (HTTP ${res.status})`);
      }
      const data = await res.json();
      statusEl.textContent = "완료되었습니다.";
      renderAgreementResults(resultsEl, data);
    } catch (err) {
      statusEl.textContent = `오류: ${err.message}`;
      statusEl.classList.add("error");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

function renderColocalizationResults(container, data) {
  const { stats, urls } = data;
  const rows = Object.entries(stats)
    .map(([k, v]) => `<tr><td>${fieldLabel(k)}</td><td>${formatValue(v)}</td></tr>`)
    .join("");

  container.innerHTML = `
    ${urls.plot ? `<img src="${API_BASE}${urls.plot}" alt="colocalization plot" />` : ""}
    <table class="summary">${rows}</table>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
}

const colocalizationForm = document.getElementById("colocalization-form");
if (colocalizationForm) {
  colocalizationForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const statusEl = document.getElementById("colocalization-status");
    const resultsEl = document.getElementById("colocalization-results");
    const submitBtn = colocalizationForm.querySelector("button[type=submit]");

    const formData = new FormData(colocalizationForm);
    submitBtn.disabled = true;
    statusEl.textContent = "Colocalization 분석 중입니다...";
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}/api/analyze/colocalization`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `요청 실패 (HTTP ${res.status})`);
      }
      const data = await res.json();
      statusEl.textContent = "완료되었습니다.";
      renderColocalizationResults(resultsEl, data);
    } catch (err) {
      statusEl.textContent = `오류: ${err.message}`;
      statusEl.classList.add("error");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

document.querySelectorAll("form[data-endpoint]").forEach((form) => {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const endpoint = form.dataset.endpoint;
    const statusEl = form.parentElement.querySelector(".status");
    const resultsEl = form.parentElement.querySelector(".results");
    const submitBtn = form.querySelector("button[type=submit]");

    const formData = new FormData(form);
    // Drop empty optional numeric fields so the backend uses its defaults.
    for (const [key, value] of Array.from(formData.entries())) {
      if (value === "" && key !== "file") formData.delete(key);
    }

    submitBtn.disabled = true;
    statusEl.textContent = "분석 중입니다... 영상 길이에 따라 다소 시간이 걸릴 수 있습니다.";
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `요청 실패 (HTTP ${res.status})`);
      }
      const data = await res.json();
      statusEl.textContent = "완료되었습니다.";
      renderResults(resultsEl, data);
    } catch (err) {
      statusEl.textContent = `오류: ${err.message}`;
      statusEl.classList.add("error");
    } finally {
      submitBtn.disabled = false;
    }
  });
});
