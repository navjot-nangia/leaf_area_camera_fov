import test from "node:test";
import assert from "node:assert/strict";
import {
  calculateLeafArea,
  largestComponent,
  orderCorners,
  perspectiveTransform,
  polygonArea,
  transformPoint
} from "../dist/measurement.js";

test("polygonArea calculates a rectangle", () => {
  assert.equal(polygonArea([{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 3 }, { x: 0, y: 3 }]), 12);
});

test("perspective transform maps all four control points", () => {
  const from = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 20 }, { x: 0, y: 20 }];
  const to = [{ x: 5, y: 7 }, { x: 25, y: 9 }, { x: 23, y: 49 }, { x: 3, y: 45 }];
  const transform = perspectiveTransform(from, to);
  from.forEach((point, index) => {
    const mapped = transformPoint(transform, point.x, point.y);
    assert.ok(Math.abs(mapped.x - to[index].x) < 1e-7);
    assert.ok(Math.abs(mapped.y - to[index].y) < 1e-7);
  });
});

test("largestComponent keeps the biggest connected region", () => {
  const mask = new Uint8Array([
    1, 0, 0, 0,
    0, 0, 1, 1,
    0, 0, 1, 1
  ]);
  const result = largestComponent(mask, 4, 3);
  assert.equal(result.count, 4);
  assert.equal(result.mask[0], 0);
  assert.equal(result.mask[6], 1);
});

test("leaf area uses the paper as a physical scale", () => {
  const area = calculateLeafArea(5000, 100, 100, 200, 300);
  assert.equal(area, 300);
});

test("orderCorners returns clockwise order starting at top-left", () => {
  const points = [{ x: 10, y: 20 }, { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 0, y: 20 }];
  assert.deepEqual(orderCorners(points), [
    { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 20 }, { x: 0, y: 20 }
  ]);
});
