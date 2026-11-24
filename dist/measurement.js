export const PAPER_SIZES = {
  a4: { widthMm: 210, heightMm: 297, label: "A4" },
  letter: { widthMm: 215.9, heightMm: 279.4, label: "US Letter" }
};

export function polygonArea(points) {
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    const next = points[(i + 1) % points.length];
    sum += points[i].x * next.y - next.x * points[i].y;
  }
  return Math.abs(sum) / 2;
}

export function solveLinearSystem(matrix, values) {
  const n = values.length;
  const augmented = matrix.map((row, index) => [...row, values[index]]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) < 1e-10) throw new Error("Could not calculate the paper perspective.");
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let j = column; j <= n; j += 1) augmented[column][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let j = column; j <= n; j += 1) augmented[row][j] -= factor * augmented[column][j];
    }
  }
  return augmented.map((row) => row[n]);
}

export function perspectiveTransform(from, to) {
  const matrix = [];
  const values = [];
  for (let i = 0; i < 4; i += 1) {
    const { x, y } = from[i];
    const target = to[i];
    matrix.push([x, y, 1, 0, 0, 0, -target.x * x, -target.x * y]);
    values.push(target.x);
    matrix.push([0, 0, 0, x, y, 1, -target.y * x, -target.y * y]);
    values.push(target.y);
  }
  const h = solveLinearSystem(matrix, values);
  return [...h, 1];
}

export function transformPoint(h, x, y) {
  const denominator = h[6] * x + h[7] * y + h[8];
  return {
    x: (h[0] * x + h[1] * y + h[2]) / denominator,
    y: (h[3] * x + h[4] * y + h[5]) / denominator
  };
}

export function largestComponent(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  let best = [];

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    const component = [];
    queue[tail++] = start;
    visited[start] = 1;

    while (head < tail) {
      const index = queue[head++];
      component.push(index);
      const x = index % width;
      const y = Math.floor(index / width);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            queue[tail++] = next;
          }
        }
      }
    }
    if (component.length > best.length) best = component;
  }

  const result = new Uint8Array(mask.length);
  for (const index of best) result[index] = 1;
  return { mask: result, count: best.length };
}

export function calculateLeafArea(leafPixels, imageWidth, imageHeight, paperWidthMm, paperHeightMm) {
  const squareMmPerPixel = (paperWidthMm * paperHeightMm) / (imageWidth * imageHeight);
  return (leafPixels * squareMmPerPixel) / 100;
}

export function orderCorners(points) {
  if (points.length < 4) throw new Error("Four paper corners are required.");
  return [
    points.reduce((best, point) => point.x + point.y < best.x + best.y ? point : best),
    points.reduce((best, point) => point.x - point.y > best.x - best.y ? point : best),
    points.reduce((best, point) => point.x + point.y > best.x + best.y ? point : best),
    points.reduce((best, point) => point.x - point.y < best.x - best.y ? point : best)
  ];
}
