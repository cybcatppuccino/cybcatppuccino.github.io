import { PendulumController } from "./controller.js";
import { PhasePortrait } from "./phase.js";
import { Renderer } from "./renderer.js";
import {
  DEFAULT_PARAMS,
  TARGETS,
  makeInitialState,
  stepRK4,
  windAcceleration,
  wrapAngle
} from "./physics.js";
import { UI } from "./ui.js";

const params = { ...DEFAULT_PARAMS };
let state = makeInitialState();
let simulationTime = 0;
let lastTimestamp = performance.now();
let accumulator = 0;
let paused = false;
let pauseReason = "";
let pausedBeforePhaseDrag = false;
let aiManualDisabled = false;
let commandAcc = 0;

// Small fixed time step gives visibly better energy behavior than frame-time integration.
const fixedDt = 1 / 360;
const maxCatchUpSteps = 8;

const renderer = new Renderer(document.querySelector("#scene"));
const controller = new PendulumController(params);
const phase = new PhasePortrait(document.querySelector("#phase"), {
  onDragStart: () => {
    pausedBeforePhaseDrag = paused;
    paused = true;
    pauseReason = "Dragging phase point";
    accumulator = 0;
  },
  onAnglesChange: (th1, th2) => {
    state.th1 = wrapAngle(th1);
    state.th2 = wrapAngle(th2);
    // Intentionally preserve angular velocities, support position and support velocity.
    controller.plan = [];
    phase.resetTrail();
    phase.addTrailPoint(state, performance.now());
  },
  onDragEnd: () => {
    paused = pausedBeforePhaseDrag;
    pauseReason = paused ? "Paused" : "";
    lastTimestamp = performance.now();
    accumulator = 0;
  }
});

const ui = new UI(params, {
  onTarget: id => selectTarget(id),
  onPauseToggle: () => {
    paused = !paused;
    pauseReason = paused ? "Paused" : "";
    lastTimestamp = performance.now();
    accumulator = 0;
  }
});

function selectTarget(id) {
  const normalizedId = Number(id);
  if (!Number.isInteger(normalizedId) || normalizedId < 0 || normalizedId > 3) return;
  ui.setActiveTarget(normalizedId);
  controller.setTarget(normalizedId, state);
}

function setAiManualDisabled(disabled) {
  if (aiManualDisabled === disabled) return;
  aiManualDisabled = disabled;
  commandAcc = 0;
  if (disabled) {
    controller.suspendForPhysicalMode?.();
  } else {
    controller.resumeFromPhysicalMode?.(state);
    lastTimestamp = performance.now();
    accumulator = 0;
  }
}

window.addEventListener("keydown", event => {
  if (event.code === "Numpad1") {
    event.preventDefault();
    if (!event.repeat) setAiManualDisabled(true);
    return;
  }

  if (!event.altKey && !event.ctrlKey && !event.metaKey && /^Digit[0-3]$/.test(event.code)) {
    event.preventDefault();
    selectTarget(Number(event.code.slice(5)));
  }
});

window.addEventListener("keyup", event => {
  if (event.code === "Numpad1") {
    event.preventDefault();
    setAiManualDisabled(false);
  }
});

window.addEventListener("blur", () => {
  setAiManualDisabled(false);
});

function resetIfNumericallyInvalid() {
  const values = [state.x, state.vx, state.th1, state.th2, state.om1, state.om2];
  if (values.some(v => !Number.isFinite(v))) {
    state = makeInitialState();
    simulationTime = 0;
    commandAcc = 0;
    controller.commandAcc = 0;
    controller.plan = [];
    phase.resetTrail();
  }
}

function stepSimulation(dt) {
  if (aiManualDisabled) {
    commandAcc = 0;
    state = stepRK4(state, 0, params, simulationTime, dt, true);
  } else {
    commandAcc = controller.update(state, params, dt, simulationTime);
    state = stepRK4(state, commandAcc, params, simulationTime, dt, true);
  }
  simulationTime += dt;
}

function frame(timestamp) {
  const elapsed = Math.min(0.05, (timestamp - lastTimestamp) / 1000);
  lastTimestamp = timestamp;

  if (!paused) {
    accumulator += elapsed;
    let steps = 0;
    while (accumulator >= fixedDt && steps < maxCatchUpSteps) {
      stepSimulation(fixedDt);
      accumulator -= fixedDt;
      steps += 1;
    }
    if (steps >= maxCatchUpSteps) accumulator = 0;
  } else {
    accumulator = 0;
  }

  resetIfNumericallyInvalid();
  if (!paused) phase.addTrailPoint(state, timestamp);
  const windAcc = windAcceleration(simulationTime, params);
  renderer.draw(state, params, controller.target || TARGETS[0], commandAcc, windAcc, paused, aiManualDisabled);
  phase.draw(state, timestamp, paused);
  ui.updateReadout(state, commandAcc, windAcc, paused, pauseReason);
  requestAnimationFrame(frame);
}

phase.addTrailPoint(state, performance.now());
requestAnimationFrame(frame);
