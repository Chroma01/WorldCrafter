// Interface language is independent of the English inference prompt.
const messages = {
  "重力适配": "Gravity alignment",
  "正在校准重力方向": "Calibrating gravity",
  "移动 / 升降": "Move / Elevate",
  "原地转头": "Look",
  "环绕": "Orbit",
  "暂停 / 继续": "Pause / Resume",
  "方向键原地转头；I/J/K/L 向上/左/下/右环绕前方虚拟中心。": "Arrow keys turn in place; I/J/K/L orbit up/left/down/right around a virtual pivot ahead.",
  "正在连接服务": "Connecting",
  "交互画面：WASD 移动，Q 上浮 E 下降，方向键转动视角，IJKL 环绕": "Interactive view: WASD to move, Q/E to move up/down, arrow keys to turn, IJKL to orbit",
  "实际输入首帧，640 × 384": "Input image, 640 × 384",
  "从一帧，走进世界": "Explore a world from one image",
  "选择场景，开始探索": "Choose a scene and start exploring",
  "首帧预览": "Input preview",
  "等待下一段 · 保持当前画面": "Waiting for the next chunk",
  "移动 ·": "Move ·",
  "升降": "Up/down",
  "转向": "Turn",
  "Space 暂停/继续": "Space: pause/resume",
  "开始探索": "Explore",
  "暂停": "Pause",
  "继续": "Resume",
  "重新开始": "Restart",
  "停止": "Stop",
  "↓ 下载录像": "↓ Download",
  "准备就绪": "Ready",
  "本段生成耗时": "Chunk generation time",
  "实际生成速度": "Generation speed",
  "播放帧率": "Playback rate",
  "当前段生成步数": "Current chunk generation steps",
  "当前生成动作": "Current action",
  "等待开始": "Waiting to start",
  "下一段待执行 · 新按键覆盖": "Next action · latest key wins",
  "等待动作输入": "Waiting for input",
  "输入动作后才生成，每段 33 帧，以 10 fps 顺序播放。下载录像保持 16 fps； 播放帧率不代表实际生成速度。": "Generation starts when you enter an action. Each chunk has 33 frames and plays at 10 fps. Downloads use 16 fps. Playback rate is independent of generation speed.",
  "随机种子": "Seed",
  "最大段数": "Max chunks",
  "WASD 移动速度": "WASD movement",
  "慢速探索": "Slower",
  "快速移动": "Faster",
  "Q/E 升降速度": "Q/E vertical movement",
  "向右环绕": "Orbit right",
  "向左环绕": "Orbit left",
  "向上环绕": "Orbit up",
  "向下环绕": "Orbit down",
  "Orbit 半径": "Orbit radius",
  "旋转角度（方向键 / IJKL）": "Turn angle (arrows / IJKL)",
  "精细调整 · 10°": "Fine · 10°",
  "大幅转向 · 45°": "Wide · 45°",
  "让视角跟随你": "Camera controls",
  "前进": "Forward",
  "后退": "Backward",
  "左移": "Left",
  "右移": "Right",
  "上浮": "Up",
  "下降": "Down",
  "抬头": "Look up",
  "低头": "Look down",
  "左转": "Turn left",
  "右转": "Turn right",
  "一段只执行一个按键动作。下一段开始前，新按键随时覆盖旧输入；开始后锁定本段动作。 短按生效一段，长按连续生效，没有待执行动作时停止生成。被覆盖的旧键不会自动恢复。": "Each chunk uses one key action, fixed when the chunk starts. Before then, the latest key replaces the pending action. Tap for one chunk or hold to repeat. Generation waits when no action is pending; replaced keys do not resume automatically.",
  "Q/E 沿世界竖直方向升降。Esc 或切换窗口清空待执行动作。暂停保留当前进度。": "Q/E moves along the world vertical axis. Press Esc or switch windows to clear pending input. Pausing preserves progress.",
  "图片驱动 · 实时相机控制": "Image input · live camera control",
  "选择场景或上传图片，然后使用方向键探索": "Choose a scene or upload an image, then explore with the keyboard",
  "输入图像与世界描述": "Input image and world description",
  "场景预设": "Scene presets",
  "选择输入图像": "Choose an input image",
  "＋ 上传自己的首帧": "＋ Upload your image",
  "自动中心裁剪为 640 × 384": "Center-cropped to 640 × 384",
  "世界描述": "World description",
  "自动描述视角": "Automatic prompt perspective",
  "第一人称": "First person",
  "第三人称": "Third person",
  "选择视角，自动生成世界描述": "Choose a perspective to generate an English description",
  "正在生成世界描述…": "Generating description…",
  "已生成，可以编辑描述后开始探索": "Ready — edit the description, then start exploring",
  "生成失败，请重新选择视角重试": "Generation failed. Select a perspective to retry",
  "等待浏览器连接": "Waiting for browser connection",
  "正在加载常驻模型": "Loading model",
  "正在编码首帧与描述": "Encoding image and description",
  "等待下一段": "Waiting for the next chunk",
  "等待动作输入 · WASD / QE / 方向键 / IJKL": "Waiting for input · WASD / QE / arrow keys / IJKL",
  "正在生成": "Generating",
  "本段完成后暂停": "Pausing after this chunk",
  "已暂停": "Paused",
  "已停止": "Stopped",
  "探索完成": "Exploration complete",
  "连接已断开": "Disconnected",
  "生成出错": "Generation failed",
  "静止": "Stationary",
  "视频加载失败，请检查连接。": "Video could not load. Check your connection.",
  "浏览器暂停了播放，请点击画面继续。": "Playback paused by the browser. Click the view to resume.",
  "已连接": "Connected",
  "连接断开 · 生成已停止": "Disconnected · generation stopped",
  "模型加载失败": "Model loading failed",
  "模拟模式": "Mock mode",
  "模型已就绪": "Model ready",
  "模型加载中": "Loading model",
  "正在生成世界描述，请稍候": "Generating a description. Please wait.",
  "模型尚未就绪，请稍后重试": "The model is not ready. Please retry shortly.",
  "请先停止当前探索，再生成世界描述": "Stop the current exploration before generating a description.",
  "服务已停止": "Service stopped",
};

// These templates cover numerical telemetry without translating prompt contents.
const templates = [
  [/^(\d+) \/ (\d+) 段$/, "$1 / $2 chunks"],
  [/^(\d+) \/ (\d+) 步$/, "$1 / $2 steps"],
  [/^([\d.]+) 米 \/ 段$/, "$1 m / chunk"],
  [/^([\d.]+)° \/ 段$/, "$1° / chunk"],
  [/^播放第 (\d+) 段 · (\d+) fps$/, "Playing chunk $1 · $2 fps"],
  [/^Orbit · 半径 ([\d.]+) m$/, "Orbit · radius $1 m"],
  [/^半径 ([\d.]+) m$/, "radius $1 m"],

];
let language = "en";
try {
  if (localStorage.getItem("worldcrafter-language") === "zh") language = "zh";
} catch { /* Storage may be disabled in a private browser. */ }
const bindings = new Map();
const normalize = (value) => value.trim().replace(/\s+/g, " ");

export function translate(source) {
  if (language === "zh") return source;
  const key = normalize(source);
  if (messages[key]) return messages[key];
  const captionError = key.match(/^自动描述生成失败：(.*)$/s);
  if (captionError) return `Description generation failed: ${translate(captionError[1])}`;
  for (const [pattern, replacement] of templates)
    if (pattern.test(key)) return key.replace(pattern, replacement);
  if (key.includes(" · ")) return key.split(" · ").map(translate).join(" · ");
  const action = key.match(/^(前进|后退|左移|右移|上浮|下降|抬头|低头|左转|右转|向右环绕|向左环绕|向上环绕|向下环绕) (.*)$/);
  if (action) return `${messages[action[1]]} ${action[2]}`;
  return source;
}

export function localizedText(element, source) {
  bindings.set(element, { source });
  element.textContent = translate(source);
}

function applyLanguage() {
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  for (const [element, { source, attribute }] of bindings) {
    if (attribute) element.setAttribute(attribute, translate(source));
    else element.textContent = translate(source);
  }
  document.getElementById("language").value = language;
}

export function initLanguage() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    bindings.set(el, { source: el.dataset.i18n });
  });
  for (const attribute of ["aria-label", "alt"]) {
    document.querySelectorAll(`[data-i18n-${attribute}]`).forEach((el) => {
      bindings.set(el, { source: el.getAttribute(`data-i18n-${attribute}`), attribute });
    });
  }
  document.getElementById("language").addEventListener("change", (event) => {
    language = event.target.value === "zh" ? "zh" : "en";
    try { localStorage.setItem("worldcrafter-language", language); } catch {}
    applyLanguage();
  });
  applyLanguage();
}
