import { Reactor } from "@reactor-team/js-sdk";
import "./style.css";

const actions = [
  ["forward", "Forward"],
  ["backward", "Backward"],
  ["left", "Move left"],
  ["right", "Move right"],
  ["up", "Move up"],
  ["down", "Move down"],
  ["yaw_left", "Look left"],
  ["yaw_right", "Look right"],
  ["pitch_up", "Look up"],
  ["pitch_down", "Look down"],
  ["orbit_left", "Orbit left"],
  ["orbit_right", "Orbit right"],
  ["orbit_up", "Orbit up"],
  ["orbit_down", "Orbit down"],
] as const;

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<header><a class="brand" href="https://reactor.inc" target="_blank" rel="noreferrer"><img src="/reactor-symbol.svg" alt="Reactor" width="32" height="23"><strong>WorldCrafter</strong><span>on Reactor</span></a>
<div class="connection"><label for="endpoint">Runtime</label><input id="endpoint" type="url" value="http://localhost:8090" spellcheck="false"><span id="status" role="status">disconnected</span><button id="connect">Connect</button></div></header>
<main><aside><p class="eyebrow">01 / BUILD YOUR WORLD</p><h1>A scene.<br>A new perspective.</h1><p class="hint">Start from an image, then explore with camera motion and persistent scene memory.</p>
<label for="image">Reference image</label><input id="image" type="file" accept="image/png,image/jpeg,image/webp"><img id="preview" alt="Selected reference" hidden>
<label for="prompt">Scene prompt</label><textarea id="prompt" rows="6" maxlength="4096" placeholder="Describe the scene and first- or third-person perspective…"></textarea><p class="hint">Image and prompt apply when you start a new world. Selecting a file does not generate video.</p>
<label for="seed">Seed</label><input id="seed" type="number" min="0" max="2147483647" step="1" value="0">
<label class="check"><input id="gravity" type="checkbox" checked> Align camera to gravity</label>
<button id="start" class="primary" disabled>Start world</button><p id="error" role="alert" hidden></p>
<details><summary>How to explore</summary><p class="hint">The first chunk is stationary. Each camera click then generates one 33-frame chunk. Wait for it to finish before the next action. Changes to the prompt require Start new world.</p></details></aside>
<section><div class="heading"><div><p class="eyebrow">02 / LIVE OUTPUT</p><h2>Explore beyond the image.</h2></div><span class="tag">640 × 384 · Adaptive playback</span></div>
<div class="screen"><video id="video" muted playsinline aria-label="Generated WorldCrafter video"></video><div id="placeholder"><span>◇</span><h3 id="waiting">Your world starts here</h3><p>Connect, choose an image and prompt, then Start world.</p></div></div>
<div class="player-bar"><span id="progress" role="status">No world generated yet.</span><div><button id="play" hidden>Play video</button><button id="pause" disabled>Pause</button><button id="reset" disabled>Reset world</button><button id="fullscreen" disabled>Fullscreen</button></div></div>
<div class="controls"><div class="heading"><h3>Camera controls</h3><span id="control-status">Connect to explore</span></div><div class="amounts"><label for="distance">Distance / chunk<input id="distance" type="number" min="0.1" max="5" step="0.1" value="0.2"></label><label for="angle">Degrees / chunk<input id="angle" type="number" min="0.1" max="45" step="0.1" value="5"></label></div><div class="actions">${actions.map(([action, label]) => `<button data-action="${action}" disabled>${label}</button>`).join("")}</div><p class="hint">One click, one chunk. No held-key repeats or automatic action queue. Pause affects future generation; buffered frames can still play.</p></div>
<footer><span>WorldCrafter-Fast · Tencent ARC</span><a href="https://reactor.inc" target="_blank" rel="noreferrer">Built on Reactor ↗</a></footer></section></main>`;

const el = <T extends HTMLElement>(id: string) =>
  document.getElementById(id)! as T;
const btn = (id: string) => el<HTMLButtonElement>(id);
const inp = (id: string) => el<HTMLInputElement>(id);
const video = el<HTMLVideoElement>("video");
const prompt = el<HTMLTextAreaElement>("prompt");
let client: Reactor | undefined;
let ready = false,
  busy = false,
  active = false,
  paused = true;
let world = 0,
  completed = 0,
  queued = 0,
  limit = 50,
  awaiting = 0;
let stream = new MediaStream();
let preview: string | undefined;
let frameCallback: number | undefined;
let playing = false;
let duration = "";

function error(message: string) {
  el("error").textContent = message;
  el("error").hidden = !message;
}
function clearVideo() {
  if (frameCallback !== undefined)
    video.cancelVideoFrameCallback(frameCallback);
  frameCallback = undefined;
  video.pause();
  video.srcObject = null;
  playing = false;
  btn("play").hidden = true;
}
function refresh() {
  btn("connect").disabled = busy;
  inp("endpoint").disabled = !!client || busy;
  btn("start").disabled =
    !ready || busy || !inp("image").files?.length || !prompt.value.trim();
  btn("start").textContent = active ? "Start new world" : "Start world";
  btn("pause").disabled = !ready || busy || !active || completed >= limit;
  btn("pause").textContent = paused ? "Resume" : "Pause";
  btn("reset").disabled = !ready || busy || !active;
  btn("fullscreen").disabled = !playing;
  for (const id of ["image", "seed", "gravity"]) inp(id).disabled = busy;
  prompt.disabled = busy;
  const pending = completed < awaiting || queued > 0;
  document
    .querySelectorAll<HTMLButtonElement>("[data-action]")
    .forEach((button) => {
      button.disabled =
        !ready ||
        busy ||
        !active ||
        paused ||
        completed < 1 ||
        completed >= limit ||
        pending;
    });
  el("placeholder").hidden = playing;
  el("waiting").textContent = active
    ? "Generating your first view…"
    : ready
      ? "Ready when you are"
      : "Your world starts here";
  el("progress").textContent = active
    ? `World ${world} · ${completed}/${limit} chunks · ${queued} queued${duration}`
    : "No world generated yet.";
  el("control-status").textContent = !ready
    ? "Connect to explore"
    : !active
      ? "Start a world first"
      : completed >= limit
        ? "World complete · start or reset"
        : paused
          ? "Paused"
          : pending || completed === 0
            ? "Rendering camera movement…"
            : "Choose your next move";
}
async function play() {
  try {
    await video.play();
    btn("play").hidden = true;
  } catch {
    btn("play").hidden = false;
  }
}
function attach() {
  if (
    !ready ||
    !active ||
    completed < 1 ||
    !stream.getVideoTracks().length ||
    video.srcObject === stream
  )
    return;
  video.srcObject = stream;
  frameCallback = video.requestVideoFrameCallback(() => {
    frameCallback = undefined;
    playing = true;
    refresh();
  });
  void play();
}
function disconnected() {
  ready = active = false;
  paused = true;
  world = completed = queued = awaiting = 0;
  duration = "";
  clearVideo();
  stream = new MediaStream();
  el("status").textContent = "disconnected";
  btn("connect").textContent = "Connect";
  refresh();
}
async function act(fn: () => Promise<void>) {
  if (busy) return;
  busy = true;
  error("");
  refresh();
  try {
    await fn();
  } catch (cause) {
    error(cause instanceof Error ? cause.message : String(cause));
  } finally {
    busy = false;
    refresh();
  }
}
async function command(name: string, data: Record<string, unknown> = {}) {
  if (!client || !ready) throw new Error("Connect before sending commands.");
  const reply = await client.sendCommand(name, data);
  if (!reply)
    throw new Error(el("error").textContent || "The command was refused.");
  return reply;
}
btn("connect").onclick = () =>
  void act(async () => {
    if (client) {
      const previous = client;
      client = undefined;
      disconnected();
      await previous.disconnect();
      return;
    }
    const endpoint = inp("endpoint").value.trim();
    if (!["http:", "https:"].includes(new URL(endpoint).protocol))
      throw new Error("Use an HTTP or HTTPS Runtime endpoint.");
    const connection = new Reactor({
      modelName: "worldcrafter-fast",
      apiUrl: endpoint,
      local: true,
      controlRequestTimeoutMs: 180000,
      modelTracks: [
        { name: "main_video", kind: "video", direction: "recvonly" },
      ],
    });
    client = connection;
    connection.on("statusChanged", (status) => {
      if (client !== connection) return;
      ready = status === "ready";
      el("status").textContent = status;
      btn("connect").textContent =
        status === "disconnected" ? "Connect" : "Disconnect";
      if (status === "disconnected") {
        client = undefined;
        disconnected();
      }
      refresh();
      attach();
    });
    connection.on("error", (cause) => {
      if (client === connection) error(cause.message);
    });
    connection.on("trackReceived", (_name, track) => {
      if (client !== connection || track.kind !== "video") return;
      if (!stream.getTracks().some((t) => t.id === track.id))
        stream.addTrack(track);
      attach();
    });
    connection.on("message", (message) => {
      if (client !== connection) return;
      const data = message.data as Record<string, unknown>;
      if (message.type === "state_update") {
        if (world !== Number(data.world_id)) {
          clearVideo();
          awaiting = 0;
          duration = "";
        }
        world = Number(data.world_id);
        active = Boolean(data.image_selected);
        paused = Boolean(data.paused);
        completed = Number(data.completed_chunks);
        queued = Number(data.queued_chunks);
        limit = Number(data.max_chunks);
        refresh();
        attach();
      } else if (
        message.type === "chunk_generated" &&
        Number(data.world_id) === world
      ) {
        duration = ` · ${Number(data.generation_seconds).toFixed(2)} s / chunk`;
        refresh();
      }
    });
    try {
      await connection.connect();
    } catch (cause) {
      if (client === connection) client = undefined;
      try {
        await connection.disconnect();
      } finally {
        disconnected();
      }
      throw cause;
    }
  });
inp("image").onchange = () => {
  if (preview) URL.revokeObjectURL(preview);
  const file = inp("image").files?.[0];
  preview = file ? URL.createObjectURL(file) : undefined;
  const image = el<HTMLImageElement>("preview");
  image.hidden = !preview;
  if (preview) image.src = preview;
  else image.removeAttribute("src");
  refresh();
};
prompt.oninput = refresh;
btn("start").onclick = () =>
  void act(async () => {
    const file = inp("image").files?.[0];
    if (!file || !client) throw new Error("Select a reference image first.");
    if (file.size > 20 * 1024 * 1024)
      throw new Error("Choose an image smaller than 20 MiB.");
    const seed = Number(inp("seed").value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647)
      throw new Error("Seed must be an integer between 0 and 2147483647.");
    const upload = await client.uploadFile(file);
    await command("start_world", {
      image: upload,
      prompt: prompt.value.trim(),
      seed,
      gravity_align: inp("gravity").checked,
    });
  });
document
  .querySelectorAll<HTMLButtonElement>("[data-action]")
  .forEach((button) => {
    button.onclick = () =>
      void act(async () => {
        const action = button.dataset.action!;
        const rotation = action.includes("_");
        const amount = Number(inp(rotation ? "angle" : "distance").value);
        if (
          !Number.isFinite(amount) ||
          amount < 0.1 ||
          amount > (rotation ? 45 : 5)
        )
          throw new Error(
            "Choose a distance of 0.1–5 or an angle of 0.1–45 degrees.",
          );
        const target = completed + 1;
        await command("move", { action, amount, chunks: 1 });
        awaiting = target;
      });
  });
btn("pause").onclick = () =>
  void act(async () => {
    await command(paused ? "resume" : "pause");
  });
btn("reset").onclick = () =>
  void act(async () => {
    await command("reset");
  });
btn("fullscreen").onclick = () =>
  void act(async () => {
    await video.requestFullscreen();
  });
btn("play").onclick = () => void play();
refresh();
