// Keep optional plotting setup available without interrupting the tutorial.
function collapsePlottingSetup() {
  document.querySelectorAll(".jp-CodeCell.celltag_plotting-setup > .jp-Cell-inputWrapper").forEach((input) => {
    if (input.closest("details")) return;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Plotting setup (optional to read)";
    input.before(details);
    details.append(summary, input);
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(collapsePlottingSetup);
} else {
  document.addEventListener("DOMContentLoaded", collapsePlottingSetup);
}
