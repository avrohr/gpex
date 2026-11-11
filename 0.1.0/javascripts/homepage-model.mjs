// A small browser-side RBF example for the homepage, independent of the Python
// runtime. Box averages use the same eight-point midpoint rule as BoxAverage.
export const kinds = ["value", "slope", "average"];
export const radius = 0.45;
export const observationVariances = { value: 0.015, slope: 0.035, average: 0.01 };

function nodes(kind, x) {
  return kind === "average"
    ? Array.from({ length: 8 }, (_, i) => x + radius * (2 * (i + 0.5) / 8 - 1))
    : [x];
}

export function truth(kind, x) {
  if (kind === "slope") return 0.9 * Math.cos(0.9 * x) - 0.5 * Math.sin(2 * x);
  const points = nodes(kind, x);
  return points.reduce((sum, p) => sum + Math.sin(0.9 * p) + 0.25 * Math.cos(2 * p), 0) / points.length;
}

export function covariance(a, b) {
  const left = nodes(a.kind, a.x);
  const right = nodes(b.kind, b.x);
  let result = 0;
  for (const x of left) {
    for (const y of right) {
      const d = x - y;
      let k = Math.exp(-0.5 * d * d);
      if (a.kind === "slope" && b.kind === "slope") k *= 1 - d * d;
      else if (a.kind === "slope") k *= -d;
      else if (b.kind === "slope") k *= d;
      result += k;
    }
  }
  return result / (left.length * right.length);
}

function cholesky(matrix) {
  const lower = matrix.map(() => Array(matrix.length).fill(0));
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j <= i; j++) {
      let entry = matrix[i][j];
      for (let k = 0; k < j; k++) entry -= lower[i][k] * lower[j][k];
      if (i === j && !(entry > 0)) throw new Error("Observation covariance is not positive definite.");
      lower[i][j] = i === j ? Math.sqrt(entry) : entry / lower[j][j];
    }
  }
  return lower;
}

function whiten(lower, vector) {
  const result = [];
  for (let i = 0; i < vector.length; i++) {
    let entry = vector[i];
    for (let j = 0; j < i; j++) entry -= lower[i][j] * result[j];
    result.push(entry / lower[i][i]);
  }
  return result;
}

export function predict(observations, kind, xs) {
  const matrix = observations.map((a, i) => observations.map((b, j) =>
    covariance(a, b) + (i === j ? observationVariances[a.kind] : 0)));
  const lower = cholesky(matrix);
  const outcomes = whiten(lower, observations.map(o => truth(o.kind, o.x)));
  return xs.map(x => {
    const query = { kind, x };
    const cross = whiten(lower, observations.map(o => covariance(o, query)));
    const mean = cross.reduce((sum, v, i) => sum + v * outcomes[i], 0);
    const variance = covariance(query, query) - cross.reduce((sum, v) => sum + v * v, 0);
    if (!Number.isFinite(mean) || !(variance >= 0)) throw new Error("Prediction is not finite or has negative variance.");
    return { x, mean, sd: Math.sqrt(variance) };
  });
}
