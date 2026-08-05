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

  container.innerHTML = `
    ${urls.plot ? `<img src="${API_BASE}${urls.plot}" alt="result plot" />` : ""}
    <table class="summary">${rows}</table>
    <div class="links">
      ${urls.csv ? `<a href="${API_BASE}${urls.csv}" download>CSV 다운로드</a>` : ""}
      ${urls.summary ? `<a href="${API_BASE}${urls.summary}" download>요약 JSON 다운로드</a>` : ""}
    </div>
  `;
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
