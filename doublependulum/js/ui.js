import { TARGETS } from "./physics.js";

function numberText(value, digits = 2) {
  return Number(value).toFixed(digits);
}

export class UI {
  constructor(params, callbacks) {
    this.params = params;
    this.callbacks = callbacks;
    this.maxAcc = document.querySelector("#maxAcc");
    this.gravity = document.querySelector("#gravity");
    this.wind = document.querySelector("#wind");
    this.friction = document.querySelector("#friction");
    this.pauseToggle = document.querySelector("#pauseToggle");
    this.maxAccValue = document.querySelector("#maxAccValue");
    this.gravityValue = document.querySelector("#gravityValue");
    this.windValue = document.querySelector("#windValue");
    this.frictionValue = document.querySelector("#frictionValue");
    this.buttons = Array.from(document.querySelectorAll(".target-button"));

    this.attachEvents();
    this.syncParamsFromControls();
    this.setActiveTarget(0);
    this.setPaused(false);
  }

  attachEvents() {
    this.buttons.forEach(button => {
      button.addEventListener("click", () => {
        const id = Number(button.dataset.target);
        this.setActiveTarget(id);
        this.callbacks.onTarget?.(id);
      });
    });

    this.pauseToggle.addEventListener("click", () => {
      this.callbacks.onPauseToggle?.();
    });

    [this.maxAcc, this.gravity, this.wind, this.friction].forEach(input => {
      input.addEventListener("input", () => this.syncParamsFromControls());
    });
  }

  syncParamsFromControls() {
    this.params.maxAcc = Number(this.maxAcc.value);
    this.params.g = Number(this.gravity.value);
    this.params.windAmp = Number(this.wind.value);
    this.params.friction = Number(this.friction.value);
    this.maxAccValue.textContent = `${numberText(this.params.maxAcc, 1)} m/s²`;
    this.gravityValue.textContent = `${numberText(this.params.g, 2)} m/s²`;
    this.windValue.textContent = `${numberText(this.params.windAmp, 2)} m/s²`;
    this.frictionValue.textContent = `${numberText(this.params.friction, 2)}`;
  }

  setActiveTarget(id) {
    this.buttons.forEach(button => button.classList.toggle("active", Number(button.dataset.target) === id));
    const target = TARGETS[id] || TARGETS[0];
    this.currentTargetName = target.name;
  }

  setPaused(paused, reason = "") {
    this.pauseToggle.textContent = paused ? "Resume" : "Pause";
    this.pauseToggle.setAttribute("aria-pressed", paused ? "true" : "false");
    this.pauseToggle.title = paused ? (reason || "Paused") : "Running";
  }

  updateReadout(state, commandAcc, windAcc, paused = false, reason = "") {
    this.setPaused(paused, reason);
  }
}
