const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const sampleBtn = document.getElementById("sample-btn");
const predictedDigit = document.getElementById("predicted-digit");
const confidenceLabel = document.getElementById("confidence");
const bars = document.getElementById("bars");

const offscreen = document.createElement("canvas");
offscreen.width = 28;
offscreen.height = 28;
const offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });

let drawing = false;

function resetCanvas() {
  ctx.fillStyle = "#02060c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "#f7fbff";
  ctx.lineWidth = 22;
}

function buildBars() {
  bars.innerHTML = "";
  for (let i = 0; i < 10; i += 1) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${i}</span>
      <div class="bar-track"><div class="bar-fill" data-bar="${i}"></div></div>
      <span data-value="${i}">0.0%</span>
    `;
    bars.appendChild(row);
  }
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  drawing = true;
  const { x, y } = pointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(event) {
  if (!drawing) return;
  const { x, y } = pointerPosition(event);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

function loadSampleSeven() {
  resetCanvas();
  ctx.strokeStyle = "#f7fbff";
  ctx.lineWidth = 20;
  ctx.beginPath();
  ctx.moveTo(72, 56);
  ctx.lineTo(214, 56);
  ctx.lineTo(138, 222);
  ctx.stroke();
}

function extractPixels() {
  offscreenCtx.fillStyle = "#000";
  offscreenCtx.fillRect(0, 0, 28, 28);
  offscreenCtx.drawImage(canvas, 0, 0, 28, 28);
  const { data } = offscreenCtx.getImageData(0, 0, 28, 28);
  const pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    pixels.push(data[i] / 255);
  }
  return pixels;
}

async function predictDigit() {
  predictedDigit.textContent = "...";
  confidenceLabel.textContent = "Confidence: running";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(extractPixels()),
  });

  if (!response.ok) {
    predictedDigit.textContent = "!";
    confidenceLabel.textContent = "Prediction failed";
    return;
  }

  const result = await response.json();
  predictedDigit.textContent = result.predicted;
  confidenceLabel.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

  result.probabilities.forEach((value, index) => {
    const bar = document.querySelector(`[data-bar="${index}"]`);
    const label = document.querySelector(`[data-value="${index}"]`);
    bar.style.width = `${value * 100}%`;
    label.textContent = `${(value * 100).toFixed(1)}%`;
  });
}

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
predictBtn.addEventListener("click", predictDigit);
clearBtn.addEventListener("click", resetCanvas);
sampleBtn.addEventListener("click", loadSampleSeven);

buildBars();
resetCanvas();
