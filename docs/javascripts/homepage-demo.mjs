import { kinds, observationVariances, predict, radius, truth } from "./homepage-model.mjs";

function initializeDemo() {
  const root = document.getElementById("gpex-demo");
  if (!root || root.dataset.ready) return;
  root.dataset.ready = "true";
  const get = id => root.querySelector(`#gpex-${id}`);
  const plots = [...root.querySelectorAll(".gpex-demo-plot")];
  const locations = root.querySelector(".gpex-demo-locations");
  for (const svg of [...plots, locations]) {
    svg.append(document.createElementNS("http://www.w3.org/2000/svg", "g"));
  }
  const names = { value: "Value", slope: "Slope", average: "Average" };
  const tracks = { value: "Values", slope: "Slopes", average: "Averages" };
  const xs = Array.from({ length: 141 }, (_, i) => -3.5 + i * 0.05);
  const checks = [...root.querySelectorAll('input[type="checkbox"]')];
  const limit = 12;
  let observations = [];
  let nextId = 0;
  let selectedId = 0;
  let dragId = null;
  let frame = null;
  const top = 16, bottom = 192, left = 31;
  const xPixel = (x, width) => left + (x + 3.5) / 7 * (width - left - 12);
  const yPixel = y => bottom - (y + 2.7) / 5.4 * (bottom - top);
  const selected = () => observations.find(o => o.id === selectedId);
  const enabled = () => new Set(checks.filter(c => c.checked).map(c => c.value));
  const active = () => observations.filter(o => enabled().has(o.kind));
  const curve = (points, value, width) => points.map((p, i) =>
    `${i ? "L" : "M"}${xPixel(p.x, width).toFixed(2)},${yPixel(value(p)).toFixed(2)}`).join(" ");

  function updatePosition() {
    const observation = selected();
    get("position").disabled = !observation;
    get("remove").disabled = !observation;
    get("location").textContent = observation ? observation.x.toFixed(2) : "—";
    if (observation) get("position").value = observation.x;
  }

  function updateEditor() {
    get("selected").replaceChildren(...observations.map(o => {
      const option = document.createElement("option");
      option.value = o.id;
      option.textContent = `${names[o.kind]} ${o.id + 1}`;
      return option;
    }));
    get("selected").value = selectedId;
    get("selected").disabled = !observations.length;
    get("add").disabled = observations.length >= limit;
    updatePosition();
  }

  function status() {
    const count = active().length;
    const observation = selected();
    get("demo-status").textContent = `${count ? `${count} active measurements` : "No active measurements · showing the prior"}`
      + ` · ${observations.length}/${limit} placed`
      + (observation ? ` · ${names[observation.kind]} ${observation.id + 1} at x = ${observation.x.toFixed(2)}` : "")
      + (observations.length === limit ? ". Remove a measurement to add another." : ".");
  }

  function draw() {
    frame = null;
    const used = active();
    try {
      for (const svg of plots) drawPrediction(svg, used);
    } catch (error) {
      plots.forEach(svg => svg.querySelector("g").replaceChildren());
      get("demo-status").textContent = `Unable to compute this prediction: ${error.message}`;
      return;
    }
    drawLocations();
    status();
  }

  function drawPrediction(svg, used) {
    const width = svg.clientWidth;
    if (!width) return;
    const kind = svg.dataset.kind;
    const prediction = predict(used, kind, xs);
    const x = value => xPixel(value, width);
    svg.setAttribute("viewBox", `0 0 ${width} 230`);
    svg.setAttribute("aria-label", `${names[kind]} prediction and uncertainty with ${used.length} active measurements`);
    const band = curve(prediction, p => p.mean + 2 * p.sd, width) + " "
      + curve([...prediction].reverse(), p => p.mean - 2 * p.sd, width).replace(/^M/, "L") + " Z";
    let markup = `<defs><clipPath id="gpex-clip-${kind}"><rect x="${left}" y="${top}" width="${width - left - 12}" height="${bottom - top}"/></clipPath></defs>`;
    for (const tick of [-2, 0, 2]) {
      markup += `<path class="gpex-axis" d="M${left},${yPixel(tick)}H${width - 12}"/>`
        + `<text x="${left - 10}" y="${yPixel(tick) + 4}" text-anchor="end">${tick}</text>`;
    }
    for (const tick of [-3, 0, 3]) {
      markup += `<text x="${x(tick)}" y="${bottom + 20}" text-anchor="middle">${tick}</text>`;
    }
    markup += `<text x="${width - 2}" y="${bottom + 20}" text-anchor="end">x</text>`;
    markup += `<g clip-path="url(#gpex-clip-${kind})"><path class="gpex-band" d="${band}"/>`
      + `<path class="gpex-truth" d="${curve(prediction, p => truth(kind, p.x), width)}"/>`
      + `<path class="gpex-mean" d="${curve(prediction, p => p.mean, width)}"/>`;
    // Only show outcomes with the same units as the selected prediction.
    for (const o of used.filter(o => o.kind === kind)) {
      const y = truth(o.kind, o.x), sd = Math.sqrt(observationVariances[o.kind]);
      markup += `<g class="gpex-${o.kind}"><path class="gpex-window" d="M${x(o.x)},${yPixel(y - 2 * sd)}V${yPixel(y + 2 * sd)}"/>`;
      if (kind === "average") markup += `<path class="gpex-window" d="M${x(o.x - radius)},${yPixel(y)}H${x(o.x + radius)}"/>`;
      markup += `<circle class="gpex-point" cx="${x(o.x)}" cy="${yPixel(y)}" r="4"/></g>`;
    }
    markup += `</g><rect class="gpex-hit-area gpex-add-area" x="${left}" y="${top}" width="${width - left - 12}" height="${bottom - top}"/>`;
    svg.querySelector("g").innerHTML = markup;
  }

  function drawLocations() {
    const width = locations.clientWidth;
    if (!width) return;
    locations.setAttribute("viewBox", `0 0 ${width} 132`);
    let markup = "";
    for (const [i, type] of kinds.entries()) {
      const y = 24 + i * 35;
      markup += `<text x="${left}" y="${y - 11}">${tracks[type]}</text>`
        + `<path class="gpex-axis" d="M${left},${y}H${width - 12}"/>`;
      for (const o of observations.filter(o => o.kind === type)) {
        const x = xPixel(o.x, width);
        const shape = type === "value" ? `<circle cx="${x}" cy="${y}" r="5"/>`
          : type === "slope" ? `<path d="M${x},${y - 6}l6,12h-12Z"/>`
          : `<rect x="${x - 5}" y="${y - 5}" width="10" height="10" rx="1"/>`;
        markup += `<g class="gpex-marker gpex-${type}" data-id="${o.id}" opacity="${enabled().has(type) ? 1 : 0.3}">`
          + `<title>${names[type]} ${o.id + 1} at x = ${o.x.toFixed(2)}</title>`
          + (type === "average" ? `<path class="gpex-window" d="M${xPixel(Math.max(-3.5, o.x - radius), width)},${y}H${xPixel(Math.min(3.5, o.x + radius), width)}"/>` : "")
          + `<g class="gpex-marker-shape">${shape}</g>`
          + (o.id === selectedId ? `<circle class="gpex-marker-selected" cx="${x}" cy="${y}" r="9"/>` : "")
          + `<circle class="gpex-hit-area" cx="${x}" cy="${y}" r="14"/></g>`;
      }
    }
    for (const tick of [-3, -2, -1, 0, 1, 2, 3]) {
      markup += `<text x="${xPixel(tick, width)}" y="124" text-anchor="middle">${tick}</text>`;
    }
    markup += `<text x="${width - 2}" y="124" text-anchor="end">x</text>`;
    locations.querySelector("g").innerHTML = markup;
  }

  function scheduleDraw() {
    if (frame === null) frame = requestAnimationFrame(draw);
  }

  function add(x) {
    if (observations.length >= limit) {
      status();
      return;
    }
    const kind = get("add-kind").value;
    checks.find(c => c.value === kind).checked = true;
    selectedId = nextId++;
    observations.push({ id: selectedId, kind, x });
    updateEditor();
    scheduleDraw();
  }

  function reset() {
    observations = [
      { kind: "value", x: -2.8 }, { kind: "value", x: -0.8 }, { kind: "value", x: 2.8 },
      { kind: "slope", x: -1.8 }, { kind: "slope", x: 1.1 },
      { kind: "average", x: -2.2 }, { kind: "average", x: 2.0 },
    ].map((o, id) => ({ ...o, id }));
    nextId = observations.length;
    selectedId = 0;
    checks.forEach(c => { c.checked = true; });
    get("add-kind").value = "value";
    updateEditor();
    scheduleDraw();
  }

  function pointerX(event) {
    const svg = event.currentTarget;
    const box = svg.getBoundingClientRect();
    const x = (event.clientX - box.left - left) / (svg.clientWidth - left - 12) * 7 - 3.5;
    return Math.round(Math.max(-3.5, Math.min(3.5, x)) * 100) / 100;
  }

  locations.addEventListener("pointerdown", event => {
    if (event.button !== 0 || !event.isPrimary) return;
    const marker = event.target.closest(".gpex-marker");
    if (marker) {
      event.preventDefault();
      dragId = selectedId = Number(marker.dataset.id);
      locations.setPointerCapture(event.pointerId);
      updateEditor();
      scheduleDraw();
    }
  });
  locations.addEventListener("pointermove", event => {
    if (dragId === null) return;
    observations.find(o => o.id === dragId).x = pointerX(event);
    updatePosition();
    scheduleDraw();
  });
  locations.addEventListener("pointerup", event => {
    if (event.button !== 0 || !event.isPrimary) return;
    if (dragId !== null) {
      dragId = null;
      locations.releasePointerCapture(event.pointerId);
    }
  });
  for (const svg of plots) {
    svg.addEventListener("click", event => {
      if (event.target.classList.contains("gpex-add-area")) {
        get("add-kind").value = svg.dataset.kind;
        add(pointerX(event));
      }
    });
  }
  locations.addEventListener("pointercancel", () => { dragId = null; });
  locations.addEventListener("lostpointercapture", () => { dragId = null; });
  checks.forEach(c => c.addEventListener("change", scheduleDraw));
  get("add").addEventListener("click", () => add(0));
  get("reset").addEventListener("click", reset);
  get("selected").addEventListener("change", event => {
    selectedId = Number(event.target.value);
    updatePosition();
    scheduleDraw();
  });
  get("position").addEventListener("input", event => {
    selected().x = Number(event.target.value);
    updatePosition();
    scheduleDraw();
  });
  get("remove").addEventListener("click", () => {
    observations = observations.filter(o => o.id !== selectedId);
    selectedId = observations[0]?.id;
    updateEditor();
    scheduleDraw();
  });
  root.querySelector(".gpex-demo-content").hidden = false;
  root.querySelector(".gpex-demo-fallback").hidden = true;
  new ResizeObserver(scheduleDraw).observe(plots[0]);
  reset();
}

initializeDemo();
if (typeof document$ !== "undefined") document$.subscribe(initializeDemo);
