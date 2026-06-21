let selectedFile = null;
let lastReportMarkdown = "";

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const uploadIdle = document.getElementById("uploadIdle");
const uploadPreview = document.getElementById("uploadPreview");
const previewImg = document.getElementById("previewImg");
const previewName = document.getElementById("previewName");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const resultsSection = document.getElementById("resultsSection");
const rupeeFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});

function setAnalyzeEnabled(enabled) {
  analyzeBtn.disabled = !enabled;
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    previewName.textContent = file.name;
    uploadIdle.classList.add("d-none");
    uploadPreview.classList.remove("d-none");
    setAnalyzeEnabled(true);
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  selectedFile = null;
  imageInput.value = "";
  previewImg.src = "";
  previewName.textContent = "";
  uploadPreview.classList.add("d-none");
  uploadIdle.classList.remove("d-none");
  setAnalyzeEnabled(false);
}

function resetForm() {
  clearImage();
  resultsSection.classList.add("d-none");
  lastReportMarkdown = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function validateAndStoreFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file.");
    return;
  }
  selectedFile = file;
  showPreview(file);
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  validateAndStoreFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files || [];
  validateAndStoreFile(file);
});

function updateMetric(id, value, suffix = "") {
  const el = document.getElementById(id);
  if (el) el.textContent = `${value}${suffix}`;
}

function updateBar(id, value, max = 100) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.max(0, Math.min(100, (value / max) * 100))}%`;
}

function setLoadingStep(activeStep) {
  for (let i = 1; i <= 4; i += 1) {
    const step = document.getElementById(`lstep${i}`);
    if (!step) continue;
    step.classList.toggle("active", i === activeStep);
    const icon = step.querySelector("i");
    if (icon) {
      icon.className = i === activeStep
        ? "fas fa-spinner fa-spin me-2"
        : "fas fa-circle me-2";
    }
  }
}

function renderProbabilities(probabilities) {
  const container = document.getElementById("probBars");
  if (!container) return;

  const order = ["Low Priority", "Medium Priority", "High Priority", "Critical Priority"];
  container.innerHTML = "";

  order.forEach((label) => {
    const pct = probabilities?.[label] ?? 0;
    const row = document.createElement("div");
    row.className = "prob-item";
    row.innerHTML = `
      <div class="d-flex justify-content-between align-items-center mb-1">
        <span>${label}</span>
        <span>${pct.toFixed(1)}%</span>
      </div>
      <div class="mc-bar"><div class="mc-fill" style="width:${pct}%"></div></div>
    `;
    container.appendChild(row);
  });
}

function renderReport(markdown) {
  const reportBody = document.getElementById("reportBody");
  if (!reportBody) return;
  lastReportMarkdown = markdown || "";
  if (window.marked) {
    reportBody.innerHTML = marked.parse(lastReportMarkdown);
  } else {
    reportBody.textContent = lastReportMarkdown;
  }
}

function analyzeImage() {
  if (!selectedFile) {
    alert("Please upload an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile, selectedFile.name);

  loadingOverlay.classList.remove("d-none");
  setLoadingStep(1);
  setTimeout(() => setLoadingStep(2), 600);
  setTimeout(() => setLoadingStep(3), 1200);
  setTimeout(() => setLoadingStep(4), 1800);

  analyzeBtn.disabled = true;

  fetch("/analyze", {
    method: "POST",
    body: formData,
  })
    .then(async (response) => {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Analysis failed.");
      }
      return data;
    })
    .then((data) => {
      document.getElementById("systemStatus").innerHTML = '<span class="status-dot"></span><span>Analysis Complete</span>';
      document.getElementById("priorityValue").textContent = data.priority.priority;
      document.getElementById("confValue").textContent = data.priority.confidence.toFixed(1);
      document.getElementById("costValue").textContent = rupeeFormatter.format(Number(data.cost.estimated_cost));

      updateMetric("metDamage", data.analysis.damage_percentage.toFixed(2), "%");
      updateMetric("metRegions", data.analysis.num_damaged_regions);
      updateMetric("metCrack", data.analysis.crack_density.toFixed(4));
      updateMetric("metRoughness", data.analysis.texture_roughness.toFixed(4));
      updateMetric("metDark", data.analysis.dark_surface_percentage.toFixed(2), "%");
      updateMetric("metCost", rupeeFormatter.format(Number(data.cost.estimated_cost)));

      updateBar("barDamage", data.analysis.damage_percentage);
      updateBar("barCrack", data.analysis.crack_density, 1);
      updateBar("barRoughness", data.analysis.texture_roughness, 1);
      updateBar("barDark", data.analysis.dark_surface_percentage);

      document.getElementById("origImg").src = data.original_image;
      document.getElementById("procImg").src = data.processed_image;

      renderProbabilities(data.priority.probabilities);
      renderReport(data.report);
      resultsSection.classList.remove("d-none");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    })
    .catch((error) => {
      console.error(error);
      alert(error.message || "Analysis failed. Please try again.");
    })
    .finally(() => {
      loadingOverlay.classList.add("d-none");
      setLoadingStep(1);
      setAnalyzeEnabled(!!selectedFile);
    });
}

function copyReport() {
  if (!lastReportMarkdown) return;
  navigator.clipboard.writeText(lastReportMarkdown).catch(() => {
    alert("Copy failed. Please try again.");
  });
}

function downloadReport() {
  if (!lastReportMarkdown) return;
  const blob = new Blob([lastReportMarkdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "road-damage-report.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

window.clearImage = clearImage;
window.resetForm = resetForm;
window.analyzeImage = analyzeImage;
window.copyReport = copyReport;
window.downloadReport = downloadReport;

setAnalyzeEnabled(false);
