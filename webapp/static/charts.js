/**
 * Plain SVG Charting Library for Concept Lab (Offline, zero-dependencies)
 */

const Charts = {
  /**
   * Render Accuracy vs Number of Concepts Line Chart
   */
  renderAccuracyCurve(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!data || data.status === "not_run" || !data.series) {
      container.innerHTML = `
        <div style="height:100%; display:flex; align-items:center; justify-content:center; color:#94a3b8; font-size:0.9rem;">
          No concept curve data available for this dataset.
        </div>`;
      return;
    }

    const width = 560;
    const height = 280;
    const padLeft = 55;
    const padRight = 30;
    const padTop = 25;
    const padBottom = 45;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Collect all points to compute min/max
    const allPoints = [];
    const seriesKeys = ["greedy", "clip_topk", "random"];
    seriesKeys.forEach((key) => {
      if (data.series[key]) {
        allPoints.push(...data.series[key]);
      }
    });

    if (data.original_point) {
      allPoints.push(data.original_point);
    }

    if (allPoints.length === 0) {
      container.innerHTML = `
        <div style="height:100%; display:flex; align-items:center; justify-content:center; color:#94a3b8;">
          No points to plot.
        </div>`;
      return;
    }

    const xVals = allPoints.map((p) => p.n_concepts);
    const yVals = allPoints.map((p) => p.accuracy);
    const stdVals = allPoints.map((p) => p.std || 0);

    let minX = Math.min(...xVals);
    let maxX = Math.max(...xVals);
    if (minX === maxX) {
      minX = Math.max(0, minX - 5);
      maxX = maxX + 5;
    }

    let minY = Math.min(...yVals.map((y, i) => y - stdVals[i]));
    let maxY = Math.max(...yVals.map((y, i) => y + stdVals[i]));

    // Round nicely
    minY = Math.max(0, Math.floor(minY * 20) / 20 - 0.05);
    maxY = Math.min(1.0, Math.ceil(maxY * 20) / 20 + 0.05);

    const scaleX = (x) => padLeft + ((x - minX) / (maxX - minX)) * plotW;
    const scaleY = (y) => padTop + plotH - ((y - minY) / (maxY - minY)) * plotH;

    // Build SVG
    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width:100%; height:100%; font-family:inherit; overflow:visible;">`;

    // Grid lines & Y ticks (5 ticks)
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const yVal = minY + (i / yTicks) * (maxY - minY);
      const yPos = scaleY(yVal);
      svg += `
        <line x1="${padLeft}" y1="${yPos}" x2="${width - padRight}" y2="${yPos}" stroke="#E2E8F0" stroke-width="1" />
        <text x="${padLeft - 8}" y="${yPos + 4}" fill="#64748B" font-size="11" text-anchor="end">${yVal.toFixed(2)}</text>
      `;
    }

    // X ticks
    const xStep = Math.max(1, Math.round((maxX - minX) / 6));
    for (let x = Math.ceil(minX); x <= maxX; x += xStep) {
      const xPos = scaleX(x);
      svg += `
        <line x1="${xPos}" y1="${padTop}" x2="${xPos}" y2="${padTop + plotH}" stroke="#F1F5F9" stroke-width="1" />
        <text x="${xPos}" y="${padTop + plotH + 18}" fill="#64748B" font-size="11" text-anchor="middle">${x}</text>
      `;
    }

    // Axes
    svg += `
      <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" stroke="#94A3B8" stroke-width="1.5" />
      <line x1="${padLeft}" y1="${padTop + plotH}" x2="${width - padRight}" y2="${padTop + plotH}" stroke="#94A3B8" stroke-width="1.5" />
      <text x="${padLeft - 38}" y="${padTop + plotH / 2}" fill="#64748B" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90, ${padLeft - 38}, ${padTop + plotH / 2})">Accuracy</text>
      <text x="${padLeft + plotW / 2}" y="${height - 6}" fill="#64748B" font-size="11" font-weight="600" text-anchor="middle">Number of Concepts (k)</text>
    `;

    // Series definitions
    const seriesMeta = {
      random: { name: "Random", color: "#9CA3AF" },
      clip_topk: { name: "Top-k by CLIP activation", color: "#3B82F6" },
      greedy: { name: "Ours (Filtered)", color: "#EF4444" },
    };

    // Draw lines & error bars
    seriesKeys.forEach((key) => {
      const pts = data.series[key];
      if (!pts || pts.length === 0) return;
      const meta = seriesMeta[key];

      // Polyline path
      const pathD = pts
        .map((p, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(p.n_concepts)} ${scaleY(p.accuracy)}`)
        .join(" ");

      svg += `<path d="${pathD}" fill="none" stroke="${meta.color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;

      // Markers & error bars
      pts.forEach((p) => {
        const cx = scaleX(p.n_concepts);
        const cy = scaleY(p.accuracy);
        const std = p.std || 0;

        // Error bar
        if (std > 0) {
          const topY = scaleY(p.accuracy + std);
          const botY = scaleY(p.accuracy - std);
          svg += `
            <line x1="${cx}" y1="${botY}" x2="${cx}" y2="${topY}" stroke="${meta.color}" stroke-width="1.5" opacity="0.8" />
            <line x1="${cx - 3}" y1="${topY}" x2="${cx + 3}" y2="${topY}" stroke="${meta.color}" stroke-width="1.5" />
            <line x1="${cx - 3}" y1="${botY}" x2="${cx + 3}" y2="${botY}" stroke="${meta.color}" stroke-width="1.5" />
          `;
        }

        // Marker
        svg += `
          <circle cx="${cx}" cy="${cy}" r="4.5" fill="${meta.color}" stroke="#FFFFFF" stroke-width="1.5">
            <title>${meta.name}
Concepts: ${p.n_concepts}
Accuracy: ${p.accuracy.toFixed(3)}${std > 0 ? " ± " + std.toFixed(3) : ""}${p.n_seeds ? " (" + p.n_seeds + " seeds)" : ""}</title>
          </circle>
        `;
      });
    });

    // Original point (Star marker)
    if (data.original_point) {
      const op = data.original_point;
      const ox = scaleX(op.n_concepts);
      const oy = scaleY(op.accuracy);
      const starPath = `M ${ox} ${oy - 8} L ${ox + 2.5} ${oy - 2.5} L ${ox + 8} ${oy - 2.5} L ${ox + 3.5} ${oy + 1.5} L ${ox + 5.5} ${oy + 7.5} L ${ox} ${oy + 4} L ${ox - 5.5} ${oy + 7.5} L ${ox - 3.5} ${oy + 1.5} L ${ox - 8} ${oy - 2.5} L ${ox - 2.5} ${oy - 2.5} Z`;
      svg += `
        <path d="${starPath}" fill="#0F172A" stroke="#FFFFFF" stroke-width="1">
          <title>Original LF-CBM
Concepts: ${op.n_concepts}
Accuracy: ${op.accuracy.toFixed(3)}</title>
        </path>
      `;
    }

    svg += `</svg>`;
    container.innerHTML = svg;
  },

  /**
   * Render Grouped Bar Chart for Redundancy
   */
  renderRedundancyBars(containerId, datasetList) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!datasetList || datasetList.length === 0) {
      container.innerHTML = `<div style="height:100%; display:flex; align-items:center; justify-content:center; color:#94a3b8;">No redundancy data</div>`;
      return;
    }

    const width = 560;
    const height = 280;
    const padLeft = 45;
    const padRight = 20;
    const padTop = 25;
    const padBottom = 45;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Find max value
    const maxVal = Math.max(
      ...datasetList
        .filter((d) => d.status === "ok")
        .flatMap((d) => [d.original || 0, d.filtered || 0]),
      0.35
    );
    const maxY = Math.ceil(maxVal * 10) / 10 + 0.05;

    const scaleY = (val) => padTop + plotH - (val / maxY) * plotH;

    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width:100%; height:100%; font-family:inherit; overflow:visible;">`;

    // Horizontal grid lines
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const yVal = (i / yTicks) * maxY;
      const yPos = scaleY(yVal);
      svg += `
        <line x1="${padLeft}" y1="${yPos}" x2="${width - padRight}" y2="${yPos}" stroke="#E2E8F0" stroke-width="1" />
        <text x="${padLeft - 6}" y="${yPos + 4}" fill="#64748B" font-size="11" text-anchor="end">${yVal.toFixed(2)}</text>
      `;
    }

    // Axes
    svg += `
      <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" stroke="#94A3B8" stroke-width="1.5" />
      <line x1="${padLeft}" y1="${padTop + plotH}" x2="${width - padRight}" y2="${padTop + plotH}" stroke="#94A3B8" stroke-width="1.5" />
      <text x="${padLeft - 32}" y="${padTop + plotH / 2}" fill="#64748B" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90, ${padLeft - 32}, ${padTop + plotH / 2})">Redundancy</text>
    `;

    // Grouped bars
    const groupCount = datasetList.length;
    const groupWidth = plotW / groupCount;
    const barWidth = 22;
    const barGap = 4;

    datasetList.forEach((item, idx) => {
      const groupCenterX = padLeft + idx * groupWidth + groupWidth / 2;

      // Dataset label
      svg += `
        <text x="${groupCenterX}" y="${padTop + plotH + 18}" fill="#334155" font-size="11" font-weight="600" text-anchor="middle">
          ${item.display_name}
        </text>
      `;

      if (item.status === "not_run" || item.original === null) {
        // Draw "Not run" slot
        const slotW = 56;
        const slotH = 22;
        svg += `
          <rect x="${groupCenterX - slotW / 2}" y="${padTop + plotH - slotH - 4}" width="${slotW}" height="${slotH}" rx="4" fill="#F1F5F9" stroke="#CBD5E1" stroke-dasharray="3 3" />
          <text x="${groupCenterX}" y="${padTop + plotH - 10}" fill="#94A3B8" font-size="10" font-weight="600" text-anchor="middle">Not run</text>
        `;
      } else {
        // Original Bar
        const origX = groupCenterX - barWidth - barGap / 2;
        const origH = plotH - (scaleY(item.original) - padTop);
        const origY = scaleY(item.original);
        svg += `
          <rect x="${origX}" y="${origY}" width="${barWidth}" height="${origH}" rx="3" fill="#94A3B8">
            <title>${item.display_name} (Original): ${item.original.toFixed(2)}</title>
          </rect>
          <text x="${origX + barWidth / 2}" y="${origY - 4}" fill="#475569" font-size="10" font-weight="700" text-anchor="middle">
            ${item.original.toFixed(2)}
          </text>
        `;

        // Filtered Bar
        const filtX = groupCenterX + barGap / 2;
        const filtH = plotH - (scaleY(item.filtered) - padTop);
        const filtY = scaleY(item.filtered);
        svg += `
          <rect x="${filtX}" y="${filtY}" width="${barWidth}" height="${filtH}" rx="3" fill="#0D9488">
            <title>${item.display_name} (Filtered): ${item.filtered.toFixed(2)}</title>
          </rect>
          <text x="${filtX + barWidth / 2}" y="${filtY - 4}" fill="#0F766E" font-size="10" font-weight="700" text-anchor="middle">
            ${item.filtered.toFixed(2)}
          </text>
        `;
      }
    });

    svg += `</svg>`;
    container.innerHTML = svg;
  },
};
