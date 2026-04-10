const apiStatus = document.getElementById("apiStatus");
const modelStatus = document.getElementById("modelStatus");
const classCount = document.getElementById("classCount");
const statusMessage = document.getElementById("statusMessage");
const classList = document.getElementById("classList");
const uploadForm = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewWrapper = document.getElementById("previewWrapper");
const analyzeButton = document.getElementById("analyzeButton");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultPanel = document.getElementById("resultPanel");
const predictionName = document.getElementById("predictionName");
const predictionDescription = document.getElementById("predictionDescription");
const predictionConfidence = document.getElementById("predictionConfidence");
const topPredictions = document.getElementById("topPredictions");
const disclaimerText = document.getElementById("disclaimerText");
const errorMessage = document.getElementById("errorMessage");

function setError(message) {
  errorMessage.hidden = !message;
  errorMessage.textContent = message || "";
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.textContent = isLoading ? "Analyzing..." : "Analyze Image";
  loadingState.hidden = !isLoading;
  if (isLoading) {
    emptyState.hidden = true;
    resultPanel.hidden = true;
  }
}

function renderClasses(classes) {
  classList.innerHTML = "";
  classes.forEach((item) => {
    const card = document.createElement("article");
    card.className = "class-item";
    card.innerHTML = `
      <strong>${item.name}</strong>
      <p><code>${item.code}</code></p>
      <p>${item.description}</p>
    `;
    classList.appendChild(card);
  });
}

function renderPredictions(predictions) {
  topPredictions.innerHTML = "";
  predictions.forEach((item) => {
    const row = document.createElement("article");
    row.className = "prediction-item";
    row.innerHTML = `
      <div class="prediction-item-header">
        <strong>${item.name}</strong>
        <strong>${(item.confidence * 100).toFixed(2)}%</strong>
      </div>
      <p><code>${item.code}</code></p>
      <p>${item.description}</p>
    `;
    topPredictions.appendChild(row);
  });
}

async function loadStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    apiStatus.textContent = data.status === "ok" ? "Online" : "Offline";
    modelStatus.textContent = data.modelReady ? "Ready" : "Not trained";
    classCount.textContent = String(data.classCount || 0);
    statusMessage.textContent = data.modelReady
      ? "Backend is healthy and the trained model is available for predictions."
      : data.error || "Backend is running, but the trained model file is not available yet.";

    if (Array.isArray(data.classes)) {
      renderClasses(data.classes);
    }
  } catch (error) {
    apiStatus.textContent = "Offline";
    modelStatus.textContent = "Unknown";
    statusMessage.textContent = "Unable to reach the Flask backend. Start the backend server and refresh.";
  }
}

imageInput.addEventListener("change", () => {
  setError("");
  const [file] = imageInput.files;
  if (!file) {
    previewWrapper.hidden = true;
    imagePreview.removeAttribute("src");
    return;
  }

  const imageUrl = URL.createObjectURL(file);
  imagePreview.src = imageUrl;
  previewWrapper.hidden = false;
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");

  const [file] = imageInput.files;
  if (!file) {
    setError("Please choose an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction request failed.");
    }

    emptyState.hidden = true;
    resultPanel.hidden = false;
    predictionName.textContent = data.prediction.name;
    predictionDescription.textContent = data.prediction.description;
    predictionConfidence.textContent = `${(data.prediction.confidence * 100).toFixed(2)}% confidence`;
    disclaimerText.textContent = data.disclaimer || "";
    renderPredictions(data.topPredictions || []);
  } catch (error) {
    setError(error.message);
    emptyState.hidden = false;
    resultPanel.hidden = true;
  } finally {
    setLoading(false);
  }
});

loadStatus();
