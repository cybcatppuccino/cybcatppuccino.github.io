const MAX_RENDER_DPR = 2;

import { pointPositionsInto, supportBounds } from "./physics.js";

function updateWorldToScreen(canvas, params, out) {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_RENDER_DPR);
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  const totalLength = params.l1 + params.l2;
  const rail = supportBounds(params);
  const worldWidth = (rail.right - rail.left) + 2 * totalLength;
  const worldHeight = 2 * totalLength;
  const horizontalPadding = w < 720 ? 0.88 : 0.80;
  const verticalPadding = w < 720 ? 0.42 : 0.62;
  const scale = Math.min(w * horizontalPadding / worldWidth, h * verticalPadding / worldHeight);
  const originX = w * 0.5;
  const originY = w < 720 ? h * 0.35 : h * 0.48;
  const worldCenterX = rail.center;

  out.scale = scale;
  out.w = w;
  out.h = h;
  out.originX = originX;
  out.originY = originY;
  out.worldCenterX = worldCenterX;
  out.railLeft = rail.left;
  out.railRight = rail.right;
  return out;
}

function mapWorldInto(transform, x, y, out) {
  out[0] = transform.originX + (x - transform.worldCenterX) * transform.scale;
  out[1] = transform.originY + y * transform.scale;
  return out;
}

function drawLine(ctx, a, b, width = 2) {
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function drawLineCoords(ctx, x0, y0, x1, y1, width = 2) {
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
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

function drawHorizontalArrowCoords(ctx, startX, startY, endX, endY, width = 2) {
  drawLineCoords(ctx, startX, startY, endX, endY, width);
  const dx = endX - startX;
  if (Math.abs(dx) < 1) return;
  const sign = Math.sign(dx);
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX - sign * 10, endY - 6);
  ctx.lineTo(endX - sign * 10, endY + 6);
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
    drawLineCoords(ctx, midX - 32, midY, midX + 32, midY, 2);
    ctx.beginPath();
    ctx.arc(midX, midY, 3.5, 0, Math.PI * 2);
    ctx.fill();
  } else if (dir > 0) {
    drawHorizontalArrowCoords(ctx, midX - 42, midY, midX - 42 + len, midY, 3);
  } else {
    drawHorizontalArrowCoords(ctx, midX + 42, midY, midX + 42 + len, midY, 3);
  }

  ctx.font = '500 10px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#64748b';
  ctx.fillText(`${windAcc.toFixed(2)} m/s²`, midX, y + 51);
  ctx.restore();
}

function drawTargetGhost(ctx, xSupport, target, params, transform, points) {
  const th1 = target.angles[0];
  const th2 = target.angles[1];
  const x0 = xSupport;
  const y0 = params.topY;
  const x1 = x0 + params.l1 * Math.sin(th1);
  const y1 = y0 + params.l1 * Math.cos(th1);
  const x2 = x1 + params.l2 * Math.sin(th2);
  const y2 = y1 + params.l2 * Math.cos(th2);
  const s0 = mapWorldInto(transform, x0, y0, points[0]);
  const s1 = mapWorldInto(transform, x1, y1, points[1]);
  const s2 = mapWorldInto(transform, x2, y2, points[2]);

  ctx.save();
  ctx.globalAlpha = 0.20;
  ctx.strokeStyle = "#1d4ed8";
  ctx.setLineDash([7, 9]);
  drawLine(ctx, s0, s1, 1.5);
  drawLine(ctx, s1, s2, 1.5);
  ctx.setLineDash([]);
  ctx.fillStyle = "#1d4ed8";
  drawCircle(ctx, s1, 2.2);
  drawCircle(ctx, s2, 2.8);
  ctx.restore();
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.dpr = 1;
    this.modelPoints = {};
    this.screenPoints = Array.from({ length: 7 }, () => [0, 0]);
    this.ghostPoints = [this.screenPoints[4], this.screenPoints[5], this.screenPoints[6]];
    this.transform = {};
    this.arrowBase = [0, 0];
    this.arrowEnd = [0, 0];
    this.railGradient = null;
    this.railGradientKey = "";
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, MAX_RENDER_DPR);
    this.canvas.width = Math.floor(this.canvas.clientWidth * this.dpr);
    this.canvas.height = Math.floor(this.canvas.clientHeight * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  clear() {
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

  draw(state, params, target, commandAcc, windAcc, paused = false, aiDisabled = false) {
    const ctx = this.ctx;
    const transform = updateWorldToScreen(this.canvas, params, this.transform);
    const width = transform.w;
    this.clear();

    const points = pointPositionsInto(state, params, this.modelPoints);
    const s0 = mapWorldInto(transform, points.x0, points.y0, this.screenPoints[0]);
    const s1 = mapWorldInto(transform, points.x1, points.y1, this.screenPoints[1]);
    const s2 = mapWorldInto(transform, points.x2, points.y2, this.screenPoints[2]);
    const railLeft = mapWorldInto(transform, transform.railLeft, params.topY, this.screenPoints[3]);
    const railRight = mapWorldInto(transform, transform.railRight, params.topY, this.screenPoints[4]);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const railGradientKey = `${railLeft[0]}:${railLeft[1]}:${railRight[0]}:${railRight[1]}`;
    if (railGradientKey !== this.railGradientKey) {
      this.railGradientKey = railGradientKey;
      this.railGradient = ctx.createLinearGradient(railLeft[0], railLeft[1], railRight[0], railRight[1]);
      this.railGradient.addColorStop(0, "#0f52ba");
      this.railGradient.addColorStop(0.52, "#38bdf8");
      this.railGradient.addColorStop(1, "#1e40af");
    }
    ctx.strokeStyle = this.railGradient;
    drawLine(ctx, railLeft, railRight, 6);

    ctx.fillStyle = "#eff6ff";
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    drawCircle(ctx, railLeft, 7, true);
    drawCircle(ctx, railLeft, 7, false);
    drawCircle(ctx, railRight, 7, true);
    drawCircle(ctx, railRight, 7, false);

    drawTargetGhost(ctx, state.x, target, params, transform, this.ghostPoints);

    ctx.strokeStyle = "#2563eb";
    drawLine(ctx, s0, s1, 3.2);
    ctx.strokeStyle = "#0ea5e9";
    drawLine(ctx, s1, s2, 3.2);

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 3;
    drawCircle(ctx, s0, 8, true);
    drawCircle(ctx, s0, 8, false);

    ctx.fillStyle = "#1d4ed8";
    drawCircle(ctx, s1, 4.6);
    ctx.fillStyle = "#0284c7";
    drawCircle(ctx, s2, 6.1);

    const arrowBase = this.arrowBase;
    const arrowEnd = this.arrowEnd;
    arrowBase[0] = s0[0];
    arrowBase[1] = s0[1] - 30;
    const arrowLength = Math.max(-48, Math.min(48, commandAcc * 1.9));
    ctx.strokeStyle = paused || aiDisabled ? "#94a3b8" : "#2563eb";
    ctx.fillStyle = paused || aiDisabled ? "#94a3b8" : "#2563eb";
    if (Math.abs(arrowLength) > 2) {
      arrowEnd[0] = arrowBase[0] + arrowLength;
      arrowEnd[1] = arrowBase[1];
      drawHorizontalArrow(ctx, arrowBase, arrowEnd, 2);
      ctx.font = "500 10px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("AI push", arrowBase[0], arrowBase[1] - 8);
    }

    drawWindIndicator(ctx, width, params, windAcc);

    if (paused || aiDisabled) {
      ctx.font = "600 13px Inter, system-ui, sans-serif";
      ctx.fillStyle = "#2563eb";
      ctx.textAlign = "center";
      ctx.fillText(paused ? "Paused" : "AI off · physics only", width * 0.5, 28);
    }

    ctx.restore();
  }
}
