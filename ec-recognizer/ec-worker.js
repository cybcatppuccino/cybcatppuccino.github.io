importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyReady = null;

async function initPyodideCore() {
  const pyodide = await loadPyodide();
  const coreCode = await fetch("./py/ec_core.py", { cache: "no-store" }).then((r) => {
    if (!r.ok) throw new Error(`Cannot load ec_core.py: ${r.status}`);
    return r.text();
  });
  await pyodide.runPythonAsync(coreCode);
  return pyodide;
}

pyReady = initPyodideCore();

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === "ping") {
    try {
      await pyReady;
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", error: String(err && err.message ? err.message : err) });
    }
    return;
  }
  if (msg.type !== "identify") return;
  try {
    const pyodide = await pyReady;
    pyodide.globals.set("EC_INPUT", String(msg.input || ""));
    const jsonText = await pyodide.runPythonAsync("identify_cubic_json(EC_INPUT)");
    self.postMessage({ type: "result", requestId: msg.requestId, result: JSON.parse(jsonText) });
  } catch (err) {
    self.postMessage({ type: "result", requestId: msg.requestId, result: { ok: false, error: String(err && err.message ? err.message : err) } });
  }
};
