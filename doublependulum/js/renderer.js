import { pointPositions } from "./physics.js";

function worldToScreenFactory(canvas, params) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  const totalLength = params.l1 + params.l2;
  const worldWidth = 2 * (params.segmentHalfLength + totalLength);
  const worldHeight = 2 * totalLength;
  const horizontalPadding = w < 720 ? 0.88 : 0.80;
  const verticalPadding = w < 720 ? 0.42 : 0.62;
  const scale = Math.min(w * horizontalPadding / worldWidth, h * verticalPadding / worldHeight);
  const originX = w * 0.5;
  const originY = w < 720 ? h * 0.35 : h * 0.48;

  return {
    scale,
    w,
    h,
    map(x, y) {
      return [originX + x * scale, originY + y * scale];
    }
  };
}

function drawLine(ctx, a, b, width = 2) {
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function drawCircle(ctx, p, r, fill = true) {
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  if (fill) ctx.fill();
  else ctx.stroke();
}


function drawHorizontalArrow(ctx, start, end, width = 2) {
  drawLine(ctx, start, end, width);
  const dx = end[0] - start[0];
  if (Math.abs(dx) < 1) return;
  const sign = Math.sign(dx);
  ctx.beginPath();
  ctx.moveTo(end[0], end[1]);
  ctx.lineTo(end[0] - sign * 10, end[1] - 6);
  ctx.lineTo(end[0] - sign * 10, end[1] + 6);
  ctx.closePath();
  ctx.fill();
}

function drawWindIndicator(ctx, width, params, windAcc) {
  const panelW = 168;
  const panelH = 58;
  const x = Math.max(18, width - panelW - 18);
  const y = 18;
  const amp = Math.max(0, params.windAmp || 0);
  const ratio = amp > 1e-6 ? Math.min(1, Math.abs(windAcc) / amp) : 0;
  const dir = Math.abs(windAcc) < 0.006 ? 0 : Math.sign(windAcc);
  const len = dir === 0 ? 0 : dir * (34 + 44 * ratio);
  const midX = x + panelW * 0.5;
  const midY = y + 34;

  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.90)';
  ctx.strokeStyle = 'rgba(37, 99, 235, 0.16)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, panelW, panelH, 18);
  ctx.fill();
  ctx.stroke();

  ctx.font = '600 12px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#1e3a8a';
  ctx.textAlign = 'center';
  ctx.fillText(dir === 0 ? 'Wind: calm' : (dir > 0 ? 'Wind →' : 'Wind ←'), midX, y + 18);

  ctx.strokeStyle = dir === 0 ? '#94a3b8' : '#0ea5e9';
  ctx.fillStyle = dir === 0 ? '#94a3b8' : '#0ea5e9';
  ctx.lineCap = 'round';
  if (dir === 0) {
    drawLine(ctx, [midX - 32, midY], [midX + 32, midY], 2);
    ctx.beginPath();
    ctx.arc(midX, midY, 3.5, 0, Math.PI * 2);
    ctx.fill();
  } else if (dir > 0) {
    drawHorizontalArrow(ctx, [midX - 42, midY], [midX - 42 + len, midY], 3);
  } else {
    drawHorizontalArrow(ctx, [midX + 42, midY], [midX + 42 + len, midY], 3);
  }

  ctx.font = '500 10px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#64748b';
  ctx.fillText(`${windAcc.toFixed(2)} m/s²`, midX, y + 51);
  ctx.restore();
}

function drawTargetGhost(ctx, xSupport, target, params, transform) {
  const ghost = {
    x: xSupport,
    vx: 0,
    th1: target.angles[0],
    th2: target.angles[1],
    om1: 0,
    om2: 0
  };
  const p = pointPositions(ghost, params);
  const s0 = transform.map(p.x0, p.y0);
  const s1 = transform.map(p.x1, p.y1);
  const s2 = transform.map(p.x2, p.y2);

  ctx.save();
  ctx.globalAlpha = 0.20;
  ctx.strokeStyle = "#1d4ed8";
  ctx.setLineDash([7, 9]);
  drawLine(ctx, s0, s1, 2);
  drawLine(ctx, s1, s2, 2);
  ctx.setLineDash([]);
  ctx.fillStyle = "#1d4ed8";
  drawCircle(ctx, s1, 5);
  drawCircle(ctx, s2, 6);
  ctx.restore();
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.floor(this.canvas.clientHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  clear() {
    const dpr = window.devicePixelRatio || 1;
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

  draw(state, params, target, commandAcc, windAcc, paused = false) {
    const ctx = this.ctx;
    const transform = worldToScreenFactory(this.canvas, params);
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.width / dpr;
    const height = this.canvas.height / dpr;
    this.clear();

    const points = pointPositions(state, params);
    const s0 = transform.map(points.x0, points.y0);
    const s1 = transform.map(points.x1, points.y1);
    const s2 = transform.map(points.x2, points.y2);
    const railLeft = transform.map(-params.segmentHalfLength, params.topY);
    const railRight = transform.map(params.segmentHalfLength, params.topY);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const railGradient = ctx.createLinearGradient(railLeft[0], railLeft[1], railRight[0], railRight[1]);
    railGradient.addColorStop(0, "#0f52ba");
    railGradient.addColorStop(0.52, "#38bdf8");
    railGradient.addColorStop(1, "#1e40af");
    ctx.strokeStyle = railGradient;
    drawLine(ctx, railLeft, railRight, 6);

    ctx.fillStyle = "#eff6ff";
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    drawCircle(ctx, railLeft, 7, true);
    drawCircle(ctx, railLeft, 7, false);
    drawCircle(ctx, railRight, 7, true);
    drawCircle(ctx, railRight, 7, false);

    drawTargetGhost(ctx, state.x, target, params, transform);

    ctx.strokeStyle = "#2563eb";
    drawLine(ctx, s0, s1, 5);
    ctx.strokeStyle = "#0ea5e9";
    drawLine(ctx, s1, s2, 5);

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 3;
    drawCircle(ctx, s0, 10, true);
    drawCircle(ctx, s0, 10, false);

    ctx.fillStyle = "#1d4ed8";
    drawCircle(ctx, s1, 12);
    ctx.fillStyle = "#0284c7";
    drawCircle(ctx, s2, 16);

    const arrowBase = [s0[0], s0[1] - 30];
    const arrowLength = Math.max(-48, Math.min(48, commandAcc * 1.9));
    ctx.strokeStyle = paused ? "#94a3b8" : "#2563eb";
    ctx.fillStyle = paused ? "#94a3b8" : "#2563eb";
    if (Math.abs(arrowLength) > 2) {
      drawHorizontalArrow(ctx, arrowBase, [arrowBase[0] + arrowLength, arrowBase[1]], 2);
      ctx.font = "500 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("AI push", arrowBase[0], arrowBase[1] - 8);
    }

    drawWindIndicator(ctx, width, params, windAcc);

    if (paused) {
      ctx.font = "600 13px Inter, system-ui, sans-serif";
      ctx.fillStyle = "#2563eb";
      ctx.textAlign = "center";
      ctx.fillText("Paused", width * 0.5, 28);
    }

    ctx.restore();
  }
}
