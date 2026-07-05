import {
  TARGETS,
  angleToPhaseCoord,
  clamp,
  phaseCoordToAngle,
  wrapUnit
} from "./physics.js";

const TRAIL_TTL_MS = 4600;
const MAX_PHASE_DPR = 2;
const MAX_TRAIL_POINTS = 520;
const MAX_DRAWN_TRAIL_SEGMENTS = 260;

function stateToPhasePoint(state) {
  return {
    u: angleToPhaseCoord(state.th1),
    v: angleToPhaseCoord(state.th2)
  };
}

function screenToCanvas(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.clientX ?? (event.touches && event.touches[0]?.clientX) ?? 0;
  const clientY = event.clientY ?? (event.touches && event.touches[0]?.clientY) ?? 0;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

export class PhasePortrait {
  constructor(canvas, callbacks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true });
    this.staticCanvas = document.createElement("canvas");
    this.staticCtx = this.staticCanvas.getContext("2d", { alpha: true });
    this.callbacks = callbacks;
    this.trail = [];
    this.dragging = false;
    this.lastDragPoint = null;
    this.resize();
    this.attachEvents();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_PHASE_DPR);
    this.canvas.width = Math.floor(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.floor(this.canvas.clientHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.staticCanvas.width = this.canvas.width;
    this.staticCanvas.height = this.canvas.height;
    this.staticCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawStaticLayer();
  }

  attachEvents() {
    this.canvas.addEventListener("pointerdown", event => {
      event.preventDefault();
      this.dragging = true;
      this.canvas.setPointerCapture(event.pointerId);
      this.callbacks.onDragStart?.();
      this.updateDraggedState(event);
    });

    this.canvas.addEventListener("pointermove", event => {
      if (!this.dragging) return;
      event.preventDefault();
      this.updateDraggedState(event);
    });

    const endDrag = event => {
      if (!this.dragging) return;
      event.preventDefault();
      this.dragging = false;
      this.canvas.releasePointerCapture?.(event.pointerId);
      this.callbacks.onDragEnd?.();
    };

    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);
    this.canvas.addEventListener("lostpointercapture", () => {
      if (!this.dragging) return;
      this.dragging = false;
      this.callbacks.onDragEnd?.();
    });
  }

  plotRect() {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_PHASE_DPR);
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    const pad = Math.max(30, Math.min(w, h) * 0.13);
    const size = Math.min(w, h) - 2 * pad;
    return {
      left: (w - size) * 0.5,
      top: (h - size) * 0.5,
      size
    };
  }

  phaseToScreen(u, v, rect, horizontalOffset = 0) {
    return {
      x: rect.left + (u + horizontalOffset) * rect.size,
      y: rect.top + (1 - v) * rect.size
    };
  }

  pointerToPhase(event) {
    const rect = this.plotRect();
    const p = screenToCanvas(this.canvas, event);
    const u = wrapUnit((p.x - rect.left) / rect.size);
    const v = clamp(1 - (p.y - rect.top) / rect.size, 0, 1);
    return { u, v };
  }

  updateDraggedState(event) {
    const p = this.pointerToPhase(event);
    this.lastDragPoint = p;
    this.callbacks.onAnglesChange?.(phaseCoordToAngle(p.u), phaseCoordToAngle(p.v));
  }

  resetTrail() {
    this.trail = [];
  }

  addTrailPoint(state, now) {
    const p = stateToPhasePoint(state);
    const last = this.trail[this.trail.length - 1];
    const moved = !last || Math.abs(last.u - p.u) + Math.abs(last.v - p.v) > 0.0015;
    if (moved) this.trail.push({ ...p, t: now });

    const cutoff = now - TRAIL_TTL_MS;
    while (this.trail.length && this.trail[0].t < cutoff) this.trail.shift();
    if (this.trail.length > MAX_TRAIL_POINTS) {
      this.trail.splice(0, this.trail.length - MAX_TRAIL_POINTS);
    }
  }

  drawStateLabels(ctx, rect) {
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 12px Inter, system-ui, sans-serif";
    for (const target of TARGETS) {
      const u = angleToPhaseCoord(target.angles[0]);
      const v = angleToPhaseCoord(target.angles[1]);
      const p = this.phaseToScreen(u, v, rect);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = target.id === 0 ? "#dbeafe" : "#e0f2fe";
      ctx.fill();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.fillStyle = "#1e3a8a";
      ctx.fillText(String(target.id), p.x, p.y + 0.4);
    }
    ctx.restore();
  }

  drawTrail(ctx, rect, now) {
    if (this.trail.length < 2) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const stride = Math.max(1, Math.ceil(this.trail.length / MAX_DRAWN_TRAIL_SEGMENTS));
    for (let i = stride; i < this.trail.length; i += stride) {
      const a = this.trail[i - stride];
      const b = this.trail[i];
      const age = now - b.t;
      const alpha = clamp(1 - age / TRAIL_TTL_MS, 0, 1);
      if (alpha <= 0) continue;

      const du = b.u - a.u;
      const dv = b.v - a.v;

      // The phase map is periodic in both coordinates. A jump across an edge
      // should be interpreted as wrap-around, not as a long straight segment
      // through the plot. Therefore we simply break the polyline at both the
      // horizontal seam and the vertical seam.
      if (Math.abs(du) > 0.5 || Math.abs(dv) > 0.5) continue;

      ctx.strokeStyle = `rgba(37, 99, 235, ${0.10 + 0.62 * alpha})`;
      ctx.lineWidth = 1.2 + 2.0 * alpha;
      const s0 = this.phaseToScreen(a.u, a.v, rect, 0);
      const s1 = this.phaseToScreen(b.u, b.v, rect, 0);
      ctx.beginPath();
      ctx.moveTo(s0.x, s0.y);
      ctx.lineTo(s1.x, s1.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  drawCurrentPoint(ctx, state, rect) {
    const p = stateToPhasePoint(state);
    const s = this.phaseToScreen(p.u, p.v, rect, 0);
    ctx.save();
    ctx.beginPath();
    ctx.arc(s.x, s.y, this.dragging ? 8.5 : 7, 0, Math.PI * 2);
    ctx.fillStyle = "#0f172a";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.4;
    ctx.stroke();
    ctx.restore();
  }

  drawStaticLayer() {
    const ctx = this.staticCtx;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_PHASE_DPR);
    const w = this.staticCanvas.width / dpr;
    const h = this.staticCanvas.height / dpr;
    const rect = this.plotRect();

    ctx.clearRect(0, 0, w, h);
    ctx.save();

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    ctx.font = "700 12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#1e3a8a";
    ctx.fillText("Angle phase map", 2, 0);

    ctx.font = "500 10px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText("θ1 →   θ2 ↑   ", 2, 16);

    const fill = ctx.createLinearGradient(rect.left, rect.top, rect.left + rect.size, rect.top + rect.size);
    fill.addColorStop(0, "rgba(239,246,255,0.65)");
    fill.addColorStop(1, "rgba(224,242,254,0.35)");
    ctx.fillStyle = fill;
    ctx.fillRect(rect.left, rect.top, rect.size, rect.size);

    ctx.strokeStyle = "#bfdbfe";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 8]);
    for (const v of [0.25, 0.5, 0.75]) {
      const x = rect.left + v * rect.size;
      const y = rect.top + (1 - v) * rect.size;
      ctx.beginPath();
      ctx.moveTo(x, rect.top);
      ctx.lineTo(x, rect.top + rect.size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(rect.left, y);
      ctx.lineTo(rect.left + rect.size, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    ctx.strokeRect(rect.left, rect.top, rect.size, rect.size);

    this.drawStateLabels(ctx, rect);
    ctx.restore();
  }

  draw(state, now, paused = false) {
    const ctx = this.ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_PHASE_DPR);
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    const rect = this.plotRect();

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.drawImage(this.staticCanvas, 0, 0, w, h);

    this.drawTrail(ctx, rect, now);
    this.drawCurrentPoint(ctx, state, rect);

    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = paused ? "#64748b" : "#2563eb";
    ctx.fillText(paused ? "dragging / paused" : "drag point to set angles", rect.left + rect.size * 0.5, rect.top + rect.size + 18);

    ctx.restore();
  }
}
