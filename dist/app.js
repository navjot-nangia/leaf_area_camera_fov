import {
  PAPER_SIZES,
  calculateLeafArea,
  largestComponent,
  orderCorners,
  perspectiveTransform,
  polygonArea,
  transformPoint
} from "./measurement.js";

const elements = {
  paperSize: document.querySelector("#paper-size"),
  customSize: document.querySelector("#custom-size"),
  paperWidth: document.querySelector("#paper-width"),
  paperHeight: document.querySelector("#paper-height"),
  photoInput: document.querySelector("#photo-input"),
  analysisCard: document.querySelector("#analysis-card"),
  paperStatus: document.querySelector("#paper-status"),
  photoCanvas: document.querySelector("#photo-canvas"),
  manualCorners: document.querySelector("#manual-corners"),
  measureButton: document.querySelector("#measure-button"),
  cornerInstruction: document.querySelector("#corner-instruction"),
  sensitivity: document.querySelector("#sensitivity"),
  resultCard: document.querySelector("#result-card"),
  areaValue: document.querySelector("#area-value"),
  coverageValue: document.querySelector("#coverage-value"),
  resultCanvas: document.querySelector("#result-canvas"),
  qualityTitle: document.querySelector("#quality-title"),
  qualityCopy: document.querySelector("#quality-copy"),
  adjustButton: document.querySelector("#adjust-button")
};

const state = { image: null, imageData: null, corners: [], manualMode: false, paperConfidence: 0 };

elements.paperSize.addEventListener("change", () => {
  elements.customSize.hidden = elements.paperSize.value !== "custom";
});

elements.photoInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    setStatus("Loading photo…");
    state.image = await loadScaledImage(file, 1600);
    const context = elements.photoCanvas.getContext("2d", { willReadFrequently: true });
    elements.photoCanvas.width = state.image.width;
    elements.photoCanvas.height = state.image.height;
    context.drawImage(state.image, 0, 0);
    state.imageData = context.getImageData(0, 0, state.image.width, state.image.height);
    state.manualMode = false;
    state.corners = [];
    elements.analysisCard.hidden = false;
    elements.resultCard.hidden = true;
    detectAndDrawPaper();
    elements.analysisCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message || "This photo could not be opened.", true);
  }
});

elements.manualCorners.addEventListener("click", () => {
  state.manualMode = true;
  state.corners = [];
  state.paperConfidence = 1;
  elements.cornerInstruction.hidden = false;
  elements.measureButton.disabled = true;
  elements.manualCorners.textContent = "Reset corners";
  setStatus("Tap the top-left corner first.");
  drawPhoto();
});

elements.photoCanvas.addEventListener("pointerdown", (event) => {
  if (!state.manualMode || !state.image || state.corners.length >= 4) return;
  const rect = elements.photoCanvas.getBoundingClientRect();
  state.corners.push({
    x: (event.clientX - rect.left) * elements.photoCanvas.width / rect.width,
    y: (event.clientY - rect.top) * elements.photoCanvas.height / rect.height
  });
  const cornerNames = ["top-right", "bottom-right", "bottom-left"];
  if (state.corners.length < 4) {
    setStatus(`Now tap the ${cornerNames[state.corners.length - 1]} corner.`);
  } else {
    state.manualMode = false;
    elements.measureButton.disabled = false;
    elements.cornerInstruction.hidden = true;
    elements.manualCorners.textContent = "Set corners manually";
    setStatus("Manual paper outline is ready.");
  }
  drawPhoto();
});

elements.measureButton.addEventListener("click", () => {
  if (state.corners.length !== 4) {
    setStatus("Set all four paper corners before measuring.", true);
    return;
  }
  setStatus("Measuring leaf…");
  requestAnimationFrame(() => {
    try { measureLeaf(); } catch (error) { setStatus(error.message || "The leaf could not be measured.", true); }
  });
});

elements.adjustButton.addEventListener("click", () => {
  elements.analysisCard.scrollIntoView({ behavior: "smooth", block: "start" });
  elements.analysisCard.querySelector("details").open = true;
});

function getPaperDimensions() {
  if (elements.paperSize.value !== "custom") return PAPER_SIZES[elements.paperSize.value];
  const widthMm = Number(elements.paperWidth.value);
  const heightMm = Number(elements.paperHeight.value);
  if (!Number.isFinite(widthMm) || !Number.isFinite(heightMm) || widthMm < 50 || heightMm < 50) {
    throw new Error("Enter valid paper dimensions of at least 50 mm.");
  }
  return { widthMm, heightMm, label: "Custom" };
}

async function loadScaledImage(file, maxDimension) {
  let source;
  if ("createImageBitmap" in window) {
    source = await createImageBitmap(file, { imageOrientation: "from-image" });
  } else {
    source = await loadImageElement(file);
  }
  const scale = Math.min(1, maxDimension / Math.max(source.width, source.height));
  const width = Math.round(source.width * scale);
  const height = Math.round(source.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(source, 0, 0, width, height);
  source.close?.();
  return canvas;
}

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("This photo could not be opened.")); };
    image.src = url;
  });
}

function detectAndDrawPaper() {
  const detected = detectPaper(state.imageData, state.image.width, state.image.height);
  if (detected) {
    state.corners = detected.corners;
    state.paperConfidence = detected.confidence;
    elements.measureButton.disabled = false;
    setStatus(detected.confidence > 0.55 ? "Paper found. Check that the outline follows all four edges." : "Paper found with low confidence. Check the outline carefully.");
  } else {
    state.corners = [];
    state.paperConfidence = 0;
    elements.measureButton.disabled = true;
    setStatus("Paper outline was not clear. Set the corners manually.", true);
  }
  drawPhoto();
}

function detectPaper(imageData, width, height) {
  const maxSide = 360;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const smallWidth = Math.max(1, Math.round(width * scale));
  const smallHeight = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = smallWidth;
  canvas.height = smallHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const source = document.createElement("canvas");
  source.width = width;
  source.height = height;
  source.getContext("2d").putImageData(imageData, 0, 0);
  context.drawImage(source, 0, 0, smallWidth, smallHeight);
  const pixels = context.getImageData(0, 0, smallWidth, smallHeight).data;
  const paperMask = new Uint8Array(smallWidth * smallHeight);

  for (let i = 0; i < paperMask.length; i += 1) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];
    const brightness = (r + g + b) / 3;
    const chroma = Math.max(r, g, b) - Math.min(r, g, b);
    paperMask[i] = brightness > 150 && chroma < 62 ? 1 : 0;
  }

  const component = largestComponent(paperMask, smallWidth, smallHeight);
  if (component.count < smallWidth * smallHeight * 0.08) return null;
  const points = [];
  for (let i = 0; i < component.mask.length; i += 1) {
    if (component.mask[i]) points.push({ x: i % smallWidth, y: Math.floor(i / smallWidth) });
  }
  const smallCorners = orderCorners(points);
  const corners = smallCorners.map((point) => ({ x: point.x / scale, y: point.y / scale }));
  const ratio = polygonArea(corners) / (width * height);
  if (ratio < 0.07) return null;
  const touches = smallCorners.filter((point) => point.x < 2 || point.y < 2 || point.x > smallWidth - 3 || point.y > smallHeight - 3).length;
  const confidence = Math.max(0.15, Math.min(0.95, ratio * 1.8 - touches * 0.12));
  return { corners, confidence };
}

function drawPhoto() {
  const context = elements.photoCanvas.getContext("2d");
  context.clearRect(0, 0, elements.photoCanvas.width, elements.photoCanvas.height);
  context.drawImage(state.image, 0, 0);
  if (!state.corners.length) return;
  const radius = Math.max(10, Math.min(elements.photoCanvas.width, elements.photoCanvas.height) * 0.012);
  context.save();
  context.lineWidth = Math.max(5, radius * 0.45);
  context.strokeStyle = "#b8ef65";
  context.fillStyle = "#173f2a";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(state.corners[0].x, state.corners[0].y);
  for (let i = 1; i < state.corners.length; i += 1) context.lineTo(state.corners[i].x, state.corners[i].y);
  if (state.corners.length === 4) context.closePath();
  context.stroke();
  state.corners.forEach((point, index) => {
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#fff";
    context.font = `700 ${Math.round(radius * 1.05)}px system-ui`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(String(index + 1), point.x, point.y);
    context.fillStyle = "#173f2a";
  });
  context.restore();
}

function measureLeaf() {
  const paper = getPaperDimensions();
  const horizontalEdge = (distance(state.corners[0], state.corners[1]) + distance(state.corners[3], state.corners[2])) / 2;
  const verticalEdge = (distance(state.corners[0], state.corners[3]) + distance(state.corners[1], state.corners[2])) / 2;
  const longSideMm = Math.max(paper.widthMm, paper.heightMm);
  const shortSideMm = Math.min(paper.widthMm, paper.heightMm);
  const paperWidthMm = horizontalEdge >= verticalEdge ? longSideMm : shortSideMm;
  const paperHeightMm = horizontalEdge >= verticalEdge ? shortSideMm : longSideMm;
  const shortSide = 620;
  const warpWidth = paperWidthMm <= paperHeightMm ? shortSide : Math.round(shortSide * paperWidthMm / paperHeightMm);
  const warpHeight = paperWidthMm <= paperHeightMm ? Math.round(shortSide * paperHeightMm / paperWidthMm) : shortSide;
  const warped = warpPaper(state.imageData, state.corners, warpWidth, warpHeight);
  const segmented = segmentLeaf(warped, 100 - Number(elements.sensitivity.value));
  if (segmented.count < warpWidth * warpHeight * 0.0005) {
    throw new Error("No clear leaf was found. Check the paper outline or increase sensitivity.");
  }
  const area = calculateLeafArea(segmented.count, warpWidth, warpHeight, paperWidthMm, paperHeightMm);
  const coverage = 100 * segmented.count / (warpWidth * warpHeight);
  renderResult(warped, segmented.mask, area, coverage);
  setStatus("Measurement complete.");
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function warpPaper(source, corners, width, height) {
  const destination = [
    { x: 0, y: 0 }, { x: width - 1, y: 0 },
    { x: width - 1, y: height - 1 }, { x: 0, y: height - 1 }
  ];
  const inverse = perspectiveTransform(destination, corners);
  const output = new ImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourcePoint = transformPoint(inverse, x, y);
      const sx = Math.max(0, Math.min(source.width - 1, Math.round(sourcePoint.x)));
      const sy = Math.max(0, Math.min(source.height - 1, Math.round(sourcePoint.y)));
      const sourceIndex = (sy * source.width + sx) * 4;
      const targetIndex = (y * width + x) * 4;
      output.data[targetIndex] = source.data[sourceIndex];
      output.data[targetIndex + 1] = source.data[sourceIndex + 1];
      output.data[targetIndex + 2] = source.data[sourceIndex + 2];
      output.data[targetIndex + 3] = 255;
    }
  }
  return output;
}

function segmentLeaf(imageData, sensitivity) {
  const { width, height, data } = imageData;
  const samples = [[], [], []];
  const band = Math.max(4, Math.round(Math.min(width, height) * 0.035));
  for (let y = 0; y < height; y += Math.max(1, Math.round(height / 120))) {
    for (let x = 0; x < width; x += Math.max(1, Math.round(width / 120))) {
      if (x < band || x >= width - band || y < band || y >= height - band) {
        const index = (y * width + x) * 4;
        samples[0].push(data[index]); samples[1].push(data[index + 1]); samples[2].push(data[index + 2]);
      }
    }
  }
  const background = samples.map((channel) => median(channel));
  const mask = new Uint8Array(width * height);
  const inset = Math.max(3, Math.round(Math.min(width, height) * 0.012));

  for (let y = inset; y < height - inset; y += 1) {
    for (let x = inset; x < width - inset; x += 1) {
      const pixel = y * width + x;
      const index = pixel * 4;
      const r = data[index]; const g = data[index + 1]; const b = data[index + 2];
      const distance = Math.hypot(r - background[0], g - background[1], b - background[2]);
      const chroma = Math.max(r, g, b) - Math.min(r, g, b);
      const darker = (background[0] + background[1] + background[2] - r - g - b) / 3;
      const score = distance + chroma * 0.35 + Math.max(0, darker) * 0.22;
      if (score > sensitivity) mask[pixel] = 1;
    }
  }
  return largestComponent(mask, width, height);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 255;
}

function renderResult(imageData, mask, area, coverage) {
  const output = new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
  for (let i = 0; i < mask.length; i += 1) {
    if (!mask[i]) continue;
    const index = i * 4;
    output.data[index] = Math.round(output.data[index] * 0.38 + 68 * 0.62);
    output.data[index + 1] = Math.round(output.data[index + 1] * 0.38 + 204 * 0.62);
    output.data[index + 2] = Math.round(output.data[index + 2] * 0.38 + 105 * 0.62);
  }
  elements.resultCanvas.width = imageData.width;
  elements.resultCanvas.height = imageData.height;
  elements.resultCanvas.getContext("2d").putImageData(output, 0, 0);
  elements.areaValue.textContent = area < 10 ? area.toFixed(2) : area.toFixed(1);
  elements.coverageValue.textContent = `${coverage.toFixed(2)}% of the paper area`;

  if (state.paperConfidence < 0.55) {
    elements.qualityTitle.textContent = "Check the paper outline";
    elements.qualityCopy.textContent = "The automatic outline had low confidence. Use manual corners if the green overlay looks wrong.";
  } else if (coverage > 60) {
    elements.qualityTitle.textContent = "Unusually large detection";
    elements.qualityCopy.textContent = "Part of the background or a shadow may be included. Try lowering sensitivity.";
  } else {
    elements.qualityTitle.textContent = "Review the green overlay";
    elements.qualityCopy.textContent = "Only the leaf should be highlighted. Adjust sensitivity if needed.";
  }
  elements.resultCard.hidden = false;
  elements.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setStatus(message, error = false) {
  elements.paperStatus.textContent = message;
  elements.paperStatus.style.color = error ? "#a13c2f" : "";
}
