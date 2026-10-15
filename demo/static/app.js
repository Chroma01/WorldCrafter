import { localizedText, initLanguage } from "./i18n.js?v=gravity-1";
const $ = (id) => document.getElementById(id);
const PLAYBACK_FPS = 10;
const ENCODED_FPS = 16;
let ws = null,
  session = null,
  generation = 0,
  uploadId = null,
  presetId = null,
  captionStyle = null,
  captionReady = false,
  inputBusy = false,
  editingLocked = false,
  allPresets = [],
  currentState = "idle";
const held = new Set();
// Use the same server snapshot as the six-step progress bar. Physical keyup
// only clears pending input; it must not extinguish the chunk being generated.
function renderActiveKeys(action, paused = false) {
  let activeKey = null;
  if (action?.forward) activeKey = action.forward > 0 ? "w" : "s";
  else if (action?.right) activeKey = action.right > 0 ? "d" : "a";
  else if (action?.up) activeKey = action.up > 0 ? "q" : "e";
  else if (action?.yaw) activeKey = action.orbit
    ? (action.yaw > 0 ? "j" : "l") : (action.yaw > 0 ? "arrowright" : "arrowleft");
  else if (action?.pitch) activeKey = action.orbit
    ? (action.pitch > 0 ? "k" : "i") : (action.pitch > 0 ? "arrowup" : "arrowdown");
  document.querySelectorAll("[data-control-key]").forEach((key) => {
    key.classList.toggle("active-chunk", key.dataset.controlKey === activeKey ||
      (key.dataset.controlKey === "space" && paused));
  });
}
const controlKeys = new Set([
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "i", "j", "k", "l",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
]);
const labels = {
  waiting_connection: "等待浏览器连接",
  loading_model: "正在加载常驻模型",
  initializing: "正在编码首帧与描述",
  calibrating: "正在校准重力方向",
  queued: "等待下一段",
  waiting_action: "等待动作输入 · WASD / QE / 方向键 / IJKL",
  generating: "正在生成",
  pausing: "本段完成后暂停",
  paused: "已暂停",
  stopped: "已停止",
  complete: "探索完成",
  disconnected: "连接已断开",
  error: "生成出错",
};
function error(message) {
  $("error").hidden = !message;
  localizedText($("error"), message || "");
}
async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw Error(e.detail);
  }
  return r.json();
}
function send(value) {
  if (ws?.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ ...value, generation }));
}
function refreshInputs() {
  const disabled = editingLocked || inputBusy;
  for (const id of ["upload", "seed", "maxChunks", "gravityAlign", "firstPerson", "thirdPerson"])
    $(id).disabled = disabled;
  $("prompt").disabled = disabled || (uploadId !== null && !captionReady);
  document.querySelector('label[for="upload"]').setAttribute("aria-disabled", String(disabled));
  for (const button of $("presetGallery").children) {
    button.disabled = disabled;
    button.setAttribute("aria-pressed", String(!uploadId && button.dataset.preset === presetId));
  }
  $("firstPerson").setAttribute("aria-pressed", String(captionStyle === "first-person"));
  $("thirdPerson").setAttribute("aria-pressed", String(captionStyle === "third-person"));
  $("start").disabled = disabled || !$("prompt").value.trim() || (uploadId !== null && !captionReady);
}
function actionText(a) {
  if (!a) return "等待下一段";
  const parts = [];
  if (a.forward)
    parts.push(
      `${a.forward > 0 ? "前进" : "后退"} ${(Math.abs(a.forward) * a.speed).toFixed(1)} m`,
    );
  if (a.right)
    parts.push(
      `${a.right > 0 ? "右移" : "左移"} ${(Math.abs(a.right) * a.speed).toFixed(1)} m`,
    );
  if (a.up)
    parts.push(
      `${a.up > 0 ? "上浮" : "下降"} ${(Math.abs(a.up) * a.speed).toFixed(1)} m`,
    );
  if (a.yaw)
    parts.push(`${a.orbit ? (a.yaw > 0 ? "向左环绕" : "向右环绕") : (a.yaw > 0 ? "右转" : "左转")} ${Math.abs(a.yaw).toFixed(0)}°`);
  if (a.pitch)
    parts.push(
      `${a.orbit ? (a.pitch > 0 ? "向下环绕" : "向上环绕") : (a.pitch > 0 ? "抬头" : "低头")} ${Math.abs(a.pitch).toFixed(0)}°`,
    );
  if (a.orbit) parts.push(`Orbit · 半径 ${a.orbit_radius.toFixed(1)} m`);
  return parts.join(" · ") || "静止";
}
class Player {
  constructor() {
    this.videos = [$("video0"), $("video1")];
    this.clear();
    this.videos.forEach((v) =>
      v.addEventListener("ended", () => {
        if (v === this.videos[this.front]) {
          this.waiting = true;
          this.advance();
        }
      }),
    );
  }
  clear() {
    this.token = (this.token || 0) + 1;
    this.queue = [];
    this.front = 0;
    this.waiting = true;
    this.loaded = null;
    this.loading = false;
    this.seen = new Set();
    this.videos.forEach((v) => {
      v.pause();
      v.removeAttribute("src");
      v.load();
      v.style.visibility = "hidden";
      v.style.zIndex = "1";
    });
    $("waitLabel").style.display = "none";
    localizedText($("playbackLabel"), "首帧预览");
  }
  add(chunk) {
    if (this.seen.has(chunk.chunk_index)) return;
    this.seen.add(chunk.chunk_index);
    this.queue.push(chunk);
    this.queue.sort((a, b) => a.chunk_index - b.chunk_index);
    this.preload();
  }
  preload() {
    if (this.loading || this.loaded || !this.queue.length) return;
    this.loading = true;
    const token = this.token;
    const chunk = this.queue.shift();
    const v = this.videos[1 - this.front];
    v.src = chunk.url;
    v.defaultPlaybackRate = PLAYBACK_FPS / ENCODED_FPS;
    v.playbackRate = PLAYBACK_FPS / ENCODED_FPS;
    const ready = () => {
      v.removeEventListener("canplay", ready);
      if (token !== this.token) return;
      this.loading = false;
      this.loaded = { v, chunk };
      this.advance();
    };
    v.addEventListener("canplay", ready);
    v.onerror = () => {
      if (token === this.token) {
        this.loading = false;
        error("视频加载失败，请检查连接。");
      }
    };
    v.load();
  }
  async advance() {
    if (!this.waiting) return;
    if (!this.loaded) {
      $("waitLabel").style.display = session ? "block" : "none";
      this.preload();
      return;
    }
    const { v, chunk } = this.loaded;
    v.playbackRate = PLAYBACK_FPS / ENCODED_FPS;
    this.loaded = null;
    this.front = this.videos.indexOf(v);
    this.waiting = false;
    v.style.visibility = "visible";
    v.style.zIndex = "3";
    this.videos[1 - this.front].style.zIndex = "2";
    $("waitLabel").style.display = "none";
    localizedText($("playbackLabel"), `播放第 ${chunk.chunk_index + 1} 段 · ${PLAYBACK_FPS} fps`);
    try {
      await v.play();
      send({
        type: "playback_started",
        chunk_index: chunk.chunk_index,
        client_time: Date.now() / 1000,
      });
    } catch (e) {
      error("浏览器暂停了播放，请点击画面继续。");
      this.waiting = true;
    }
    this.preload();
  }
}
initLanguage();
const player = new Player();
function state(s) {
  currentState = s.state;
  localizedText($("state"), labels[s.state] || s.state);
  localizedText($("progressText"), `${s.chunks} / ${s.max_chunks} 段`);
  localizedText($("currentAction"), actionText(s.active));
  const completed = Math.max(0, Math.min(6, Number(s.completed_steps) || 0));
  localizedText($("stepCount"), `${completed} / 6 步`);
  $("stepProgress").setAttribute("aria-valuenow", String(completed));
  [...$("stepProgress").children].forEach((segment, i) =>
    segment.classList.toggle("complete", i < completed),
  );
  localizedText($("nextAction"), actionText(s.pending) === "静止"
    ? "等待动作输入" : actionText(s.pending));
  if (s.generation_s != null) {
    $("seconds").innerHTML = `${s.generation_s.toFixed(2)}<small> s</small>`;
    $("fps").innerHTML = `${s.generation_fps.toFixed(2)}<small> FPS</small>`;
  }
  const terminal = ["stopped", "complete", "disconnected", "error"].includes(
    s.state,
  );
  renderActiveKeys(terminal ? null : s.active, ["paused", "pausing"].includes(s.state));
  editingLocked = !terminal;
  $("pause").disabled = terminal || ["paused", "pausing"].includes(s.state);
  $("resume").disabled = !["paused", "pausing"].includes(s.state);
  $("reset").disabled = ["stopped", "disconnected", "error"].includes(s.state);
  $("stop").disabled = terminal;
  $("download").setAttribute("aria-disabled", String(s.chunks === 0));
  $("download").href =
    `/api/sessions/${session}/recording?generation=${generation}`;
  refreshInputs();
  if (s.error) error(s.error);
}
function changePreset() {
  const p = allPresets.find((p) => p.id === presetId);
  uploadId = null;
  captionStyle = null;
  captionReady = false;
  $("captionModes").hidden = true;
  $("upload").value = "";
  player.clear();
  $("prompt").value = p.prompt;
  $("sceneName").textContent = p.name;
  $("preview").src = p.image;
  $("preview").style.visibility = "visible";
  $("emptyState").style.display = "none";
  refreshInputs();
  error("");
}
$("prompt").addEventListener("input", refreshInputs);
$("upload").addEventListener("change", async () => {
  const f = $("upload").files[0];
  if (!f) return;
  inputBusy = true;
  refreshInputs();
  try {
    const r = await api("/api/upload", { method: "POST", body: f });
    uploadId = r.upload_id;
    captionStyle = null;
    captionReady = false;
    $("prompt").value = "";
    $("captionModes").hidden = false;
    localizedText($("captionStatus"), "选择视角，自动生成世界描述");
    player.clear();
    $("preview").src = r.preview;
    $("preview").style.visibility = "visible";
    $("sceneName").textContent = f.name;
    $("emptyState").style.display = "none";
    error("");
  } catch (e) {
    error(e.message);
  } finally {
    inputBusy = false;
    $("upload").value = "";
    refreshInputs();
  }
});
async function generatePrompt(style) {
  if (!uploadId || inputBusy || editingLocked) return;
  const image = uploadId;
  captionStyle = style;
  captionReady = false;
  inputBusy = true;
  $("prompt").value = "";
  localizedText($("captionStatus"), "正在生成世界描述…");
  refreshInputs();
  error("");
  try {
    const result = await api("/api/caption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: image, style }),
    });
    if (uploadId !== image) return;
    $("prompt").value = result.prompt;
    captionReady = true;
    localizedText($("captionStatus"), "已生成，可以编辑描述后开始探索");
  } catch (e) {
    localizedText($("captionStatus"), "生成失败，请重新选择视角重试");
    error(e.message);
  } finally {
    inputBusy = false;
    refreshInputs();
  }
}
$("firstPerson").addEventListener("click", () => generatePrompt("first-person"));
$("thirdPerson").addEventListener("click", () => generatePrompt("third-person"));
$("start").addEventListener("click", async () => {
  try {
    error("");
    editingLocked = true;
    refreshInputs();
    if (ws) ws.close();
    ws = null;
    const s = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset: presetId,
        upload_id: uploadId,
        prompt: $("prompt").value,
        seed: Number($("seed").value),
        max_chunks: Number($("maxChunks").value),
        gravity_align: $("gravityAlign").checked,
      }),
    });
    session = s.session_id;
    generation = s.generation;
    player.clear();
    state(s);
    $("seedFooter").textContent = $("seed").value;
    const connection = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/${session}`,
    );
    ws = connection;
    ws.onopen = () => {
      if (ws !== connection) return;
      localizedText($("connection"), "已连接");
      $("connectionDot").classList.add("online");
      send({ type: "speed", value: Number($("speed").value) });
      send({ type: "vertical_speed", value: Number($("verticalSpeed").value) });
      send({ type: "rotation_angle", value: Number($("rotationAngle").value) });
      send({ type: "orbit_radius", value: Number($("orbitRadius").value) });
    };
    ws.onmessage = (e) => {
      if (ws !== connection) return;
      const m = JSON.parse(e.data);
      if (m.type === "error") {
        error(m.message);
        return;
      }
      if (m.generation < generation) return;
      if (m.generation > generation) {
        generation = m.generation;
        player.clear();
        clearKeys(false);
      }
      if (m.type === "status") state(m);
      if (m.type === "chunk_ready") player.add(m);
    };
    ws.onclose = () => {
      if (ws !== connection) return;
      clearKeys(false);
      localizedText($("connection"), "连接断开 · 生成已停止");
      $("connectionDot").classList.remove("online");
      renderActiveKeys(null);
      currentState = "disconnected";
      editingLocked = false;
      refreshInputs();
    };
  } catch (e) {
    error(e.message);
    editingLocked = false;
    refreshInputs();
  }
});
for (const type of ["pause", "resume", "reset", "stop"])
  $(type).addEventListener("click", () => {
    if (type === "reset" || type === "stop") clearKeys();
    send({ type });
  });
$("orbitRadius").addEventListener("input", () => {
  const value = Number($("orbitRadius").value);
  $("orbitRadiusLabel").textContent = value.toFixed(1);
  send({ type: "orbit_radius", value });
});
for (const [id, type, label, angle] of [
  ["speed", "speed", "speedLabel", false],
  ["verticalSpeed", "vertical_speed", "verticalSpeedLabel", false],
  ["rotationAngle", "rotation_angle", "rotationAngleLabel", true],
]) {
  $(id).addEventListener("input", () => {
    const value = Number($(id).value);
    localizedText($(label), angle
      ? `${value.toFixed(0)}° / 段`
      : `${value.toFixed(1)} 米 / 段`);
    send({ type, value });
  });
}
function clearKeys(notify = true) {
  held.clear();
  if (notify) send({ type: "blur" });
}
function editingText(target) {
  return target.tagName === "TEXTAREA" || target.isContentEditable ||
    (target.tagName === "INPUT" && !["range", "file"].includes(target.type));
}
// Pointer changes return control immediately. Keyboard users can still adjust
// sliders/selects with arrows; WASD works while these controls have focus.
for (const control of document.querySelectorAll('input[type="range"], select')) {
  control.addEventListener(control.tagName === "SELECT" ? "change" : "pointerup", () => {
    if (session && !editingLocked) return;
    if (session) $("viewport").focus({ preventScroll: true });
  });
}
document.addEventListener("focusin", (e) => {
  if (editingText(e.target)) clearKeys();
});
document.addEventListener("keydown", (e) => {
  if (editingText(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
  if ((e.target.tagName === "SELECT" || e.target.type === "range") &&
      ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(e.key)) return;
  if (
    !session ||
    ["stopped", "complete", "disconnected", "error"].includes(currentState)
  )
    return;
  if (e.key === "Escape") {
    clearKeys();
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    if (!e.repeat)
      send({
        type: ["paused", "pausing"].includes(currentState) ? "resume" : "pause",
      });
    return;
  }
  const key = e.key.toLowerCase();
  if (!controlKeys.has(key)) return;
  e.preventDefault();
  // OS auto-repeat must never make an older held key override a newer key.
  if (!e.repeat && !held.has(key)) {
    held.add(key);
    send({ type: "key", key, down: true });
  }
});
document.addEventListener("keyup", (e) => {
  const key = e.key.toLowerCase();
  if (held.delete(key)) send({ type: "key", key, down: false });
});
window.addEventListener("blur", () => clearKeys());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearKeys();
});
$("viewport").addEventListener("click", () => {
  $("viewport").focus();
  if (session && player.waiting) player.advance();
});
window.addEventListener("beforeunload", () => ws?.close());
(async () => {
  try {
    const r = await api("/api/presets");
    allPresets = r.presets;
    $("presetCount").textContent = `01 — ${String(allPresets.length).padStart(2, "0")}`;
    for (const p of allPresets) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "preset-card";
      button.dataset.preset = p.id;
      const image = document.createElement("img");
      image.loading = "lazy";
      image.decoding = "async";
      image.width = 320;
      image.height = 192;
      image.addEventListener("error", () => { image.src = p.image; }, { once: true });
      image.src = `/static/thumbnails/${encodeURIComponent(p.id)}.webp`;
      image.alt = p.name;
      button.setAttribute("aria-label", p.name);
      button.append(image);
      button.addEventListener("click", () => {
        presetId = p.id;
        changePreset();
      });
      $("presetGallery").append(button);
    }
    presetId = r.default;
    changePreset();
    const h = await api("/api/health");
    localizedText($("connection"), h.error
      ? "模型加载失败"
      : h.mock
        ? "模拟模式"
        : h.ready
          ? "模型已就绪"
          : "模型加载中");
    $("connectionDot").classList.add("online");
    if (h.error) error(h.error);
  } catch (e) {
    error(e.message);
  }
})();
