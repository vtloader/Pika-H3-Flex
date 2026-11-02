/* Pika H3 Flex WebUI — hand-written, no build step, no eval. */
(() => {
  "use strict";

  const MAX_SEED = 9223372036854775807n;
  const SEED_STORE = "pika-h3-seed";
  const FORM_STORE = "pika-h3-form";
  const seedApi = window.PikaSeed || null;
  const IMAGE_SUFFIXES = ["png", "jpg", "jpeg", "webp", "bmp"];
  const VIDEO_SUFFIXES = ["mp4", "mov", "m4v", "webm", "mkv"];
  const AUDIO_SUFFIXES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"];
  const LIMITS = {
    image_bytes: 20 * 1024 * 1024,
    video_bytes: 256 * 1024 * 1024,
    audio_bytes: 64 * 1024 * 1024,
    request_bytes: 512 * 1024 * 1024,
    images: 9, videos: 3, audio: 3, total: 12,
    clip_min: 2, clip_max: 15, clip_total: 15,
    prompt: 8000,
  };

  const $ = (id) => document.getElementById(id);
  const statusLabels = { queued: "排队中", running: "生成中", done: "已完成", error: "失败", canceled: "已取消" };
  const stageLabels = {
    queued: "等待执行", prepare: "准备", loading: "加载模型", conditioning: "条件编码",
    sampling: "采样去噪", decode: "解码音视频", encode: "封装 MP4", done: "完成",
    error: "失败", canceled: "已取消",
  };
  const modeLabels = { text: "文本", keyframes: "关键帧", firstlast: "首尾帧", references: "多模态参考" };

  // ---------- helpers ----------
  function bytes(n) {
    if (!Number.isFinite(n)) return "";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / 1024 / 1024).toFixed(1)} MiB`;
  }
  function digits(text) {
    return String(text == null ? "" : text).replace(/[^0-9]/g, "");
  }
  function bigintFrom(text) {
    const clean = digits(text);
    if (!clean) return null;
    try { return BigInt(clean); } catch (_) { return null; }
  }
  function randomSeedString() {
    if (seedApi) return seedApi.randomSeed(window.crypto);
    const buffer = new Uint8Array(8);
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      window.crypto.getRandomValues(buffer);
    } else {
      for (let i = 0; i < buffer.length; i += 1) buffer[i] = Math.floor(Math.random() * 256);
    }
    let value = 0n;
    for (const byte of buffer) value = (value << 8n) | BigInt(byte);
    return (value % (MAX_SEED + 1n)).toString();
  }
  function seedToBigInt(text) {
    const value = bigintFrom(text);
    return value === null ? null : value;
  }

  function clampSeed(value) {
    if (value < 0n) return 0n;
    if (value > MAX_SEED) return MAX_SEED;
    return value;
  }
  function decimal(input) {
    const text = String(input.value == null ? "" : input.value).trim();
    if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)) return NaN;
    const value = Number(text);
    return Number.isFinite(value) ? value : NaN;
  }
  // ---------- seed control ----------
  const seedInput = $("seed");
  const seedError = $("seed-error");
  const seedModeRoot = $("seed-mode");
  let seedMode = "random";

  function loadSeedState() {
    try {
      const raw = localStorage.getItem(SEED_STORE);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (parsed && (parsed.mode === "fixed" || parsed.mode === "random")) return parsed;
    } catch (_) { /* ignore corrupt storage */ }
    return null;
  }
  function saveSeedState() {
    try {
      localStorage.setItem(SEED_STORE, JSON.stringify({
        mode: seedMode, value: seedInput.value.trim(),
      }));
    } catch (_) { /* storage may be blocked */ }
  }
  function setSeedMode(mode, { keepValue = true } = {}) {
    seedMode = mode === "fixed" ? "fixed" : "random";
    for (const button of seedModeRoot.querySelectorAll("button")) {
      button.setAttribute("aria-selected", String(button.dataset.seedMode === seedMode));
    }
    if (seedMode === "random" && !keepValue) seedInput.value = randomSeedString();
    seedInput.classList.toggle("fixed-seed", seedMode === "fixed");
    $("seed-hint").textContent = seedMode === "fixed"
      ? "固定模式：当前种子会在每次提交时重复使用；切回“随机”或点“换一个”可重新抽取。"
      : "随机模式：每次提交后自动换新的种子；手动输入或 +1/−1 会切换为“固定”。";
    validateSeed();
    saveSeedState();
  }
  function validateSeed() {
    seedInput.setCustomValidity("");
    const value = seedToBigInt(seedInput.value);
    let message = "";
    if (value === null) message = "种子必须是 0 到 2^63−1 之间的整数。";
    else if (value < 0n) message = "种子不能为负数。";
    else if (value > MAX_SEED) message = "种子必须小于 2^63（9223372036854775808）。";
    seedError.textContent = message;
    if (message) seedInput.setCustomValidity(message);
    return !message;
  }
  function stepSeed(delta) {
    if (seedApi) {
      seedInput.value = seedApi.stepSeedText(seedInput.value, delta);
    } else {
      const current = seedToBigInt(seedInput.value) ?? 0n;
      seedInput.value = clampSeed(current + BigInt(delta)).toString();
    }
    setSeedMode("fixed");
  }
  seedInput.addEventListener("input", () => {
    // Manual typing is an explicit choice: keep it and switch to fixed.
    seedInput.value = digits(seedInput.value).slice(0, 19);
    setSeedMode("fixed");
  });
  $("seed-plus").addEventListener("click", () => stepSeed(1));
  $("seed-minus").addEventListener("click", () => stepSeed(-1));
  $("seed-reroll").addEventListener("click", () => {
    seedInput.value = randomSeedString();
    setSeedMode("random", { keepValue: true });
  });
  for (const button of seedModeRoot.querySelectorAll("button")) {
    button.addEventListener("click", () => {
      const mode = button.dataset.seedMode;
      if (mode === "random" && seedMode !== "random") {
        // Switching to random draws a fresh value immediately.
        seedInput.value = randomSeedString();
        setSeedMode("random", { keepValue: true });
      } else {
        setSeedMode(mode);
      }
    });
  }

  function initSeed() {
    const stored = loadSeedState();
    if (stored && stored.mode === "fixed") {
      const value = bigintFrom(stored.value);
      seedInput.value = (value === null ? randomSeedString() : value.toString());
      setSeedMode("fixed");
    } else {
      seedInput.value = randomSeedString();
      setSeedMode("random", { keepValue: true });
    }
  }

  // ---------- uploads ----------
  const uploads = ["first", "last", "refs", "video_refs", "audio_refs"].map((name) => ({
    name,
    input: $(name),
    preview: $(`${name}-preview`),
    clear: name === "refs" || name === "video_refs" || name === "audio_refs" ? $(`clear-${name}`) : null,
    files: [],
    entries: [],
    replaceIndex: null,
  }));
  const accumulating = new Set(["refs", "video_refs", "audio_refs"]);
  const metaCache = new Map();

  function uploadByName(name) {
    return uploads.find((upload) => upload.name === name);
  }
  function syncInput(upload) {
    if (typeof DataTransfer === "undefined") return;
    try {
      const transfer = new DataTransfer();
      for (const file of upload.files) transfer.items.add(file);
      upload.input.files = transfer.files;
    } catch (_) { /* browsers without DataTransfer still post the FileList below */ }
  }
  function suffixOf(file) {
    const name = file && file.name ? file.name : "";
    const index = name.lastIndexOf(".");
    return index >= 0 ? name.slice(index + 1).toLowerCase() : "";
  }
  function validateFile(kind, file) {
    const suffix = suffixOf(file);
    if (kind === "image") {
      if (!IMAGE_SUFFIXES.includes(suffix)) return `${file.name}：仅支持 png/jpg/webp/bmp`;
      if (file.size > LIMITS.image_bytes) return `${file.name}：图片超过 20 MiB`;
    } else if (kind === "video") {
      if (!VIDEO_SUFFIXES.includes(suffix)) return `${file.name}：仅支持 mp4/mov/m4v/webm/mkv`;
      if (file.size > LIMITS.video_bytes) return `${file.name}：视频超过 256 MiB`;
    } else {
      if (!AUDIO_SUFFIXES.includes(suffix)) return `${file.name}：仅支持 wav/mp3/m4a/aac/flac/ogg/opus`;
      if (file.size > LIMITS.audio_bytes) return `${file.name}：音频超过 64 MiB`;
    }
    return "";
  }
  function kindOf(upload) {
    if (upload.name === "video_refs") return "video";
    if (upload.name === "audio_refs") return "audio";
    return "image";
  }
  function refOrderLabel(upload, index) {
    if (upload.name === "refs") return `Picture ${index + 1}`;
    if (upload.name === "video_refs") return `Video ${index + 1}`;
    if (upload.name === "audio_refs") return `Audio ${index + 1}`;
    return upload.name === "first" ? "首帧" : "尾帧";
  }
  function releasePreview(entry) {
    entry.active = false;
    try { URL.revokeObjectURL(entry.url); } catch (_) { /* ignore */ }
    if (entry.media) entry.media.removeAttribute("src");
  }
  function releasePreviews(upload) {
    for (const entry of upload.entries) releasePreview(entry);
    upload.entries = [];
    if (upload.preview) upload.preview.replaceChildren();
  }
  function mediaDuration(file) {
    const cached = metaCache.get(file);
    if (cached) return Promise.resolve(cached);
    return new Promise((resolve) => {
      const isAudio = (file.type || "").startsWith("audio/") || AUDIO_SUFFIXES.includes(suffixOf(file));
      const element = document.createElement(isAudio ? "audio" : "video");
      element.preload = "metadata";
      const url = URL.createObjectURL(file);
      const finish = (value) => {
        URL.revokeObjectURL(url);
        const result = { duration: Number.isFinite(value) ? value : null, width: element.videoWidth || null, height: element.videoHeight || null };
        if (result.duration !== null) metaCache.set(file, result);
        resolve(result);
      };
      element.addEventListener("loadedmetadata", () => finish(element.duration));
      element.addEventListener("error", () => finish(NaN));
      element.src = url;
      window.setTimeout(() => finish(NaN), 8000);
    });
  }
  function renderUploads() {
    for (const upload of uploads) {
      releasePreviews(upload);
      const files = upload.files;
      if (upload.clear) upload.clear.disabled = files.length === 0;
      files.forEach((file, index) => {
        const figure = document.createElement("figure");
        figure.className = "upload-preview";
        const thumb = document.createElement("div");
        thumb.className = "thumb";
        const kind = kindOf(upload);
        const media = document.createElement(kind === "video" ? "video" : kind === "audio" ? "audio" : "img");
        if (media.tagName === "IMG") media.alt = file.name;
        else { media.controls = true; media.preload = "metadata"; }
        const entry = { media, url: "", active: true };
        try {
          entry.url = URL.createObjectURL(file);
          media.src = entry.url;
        } catch (_) { /* preview only */ }
        upload.entries.push(entry);

        const order = document.createElement("span");
        order.className = "order";
        order.textContent = refOrderLabel(upload, index);
        thumb.append(media, order);

        const meta = document.createElement("div");
        meta.className = "meta";
        const title = document.createElement("strong");
        title.textContent = file.name;
        title.title = file.name;
        const info = document.createElement("small");
        info.textContent = bytes(file.size);
        meta.append(title, info);

        if (kind !== "image") {
          mediaDuration(file).then((result) => {
            if (!entry.active) return;
            if (result.duration === null) {
              info.textContent = `${bytes(file.size)} · 无法读取时长`;
              return;
            }
            const total = result.duration;
            const durationText = `${total.toFixed(2)}s`;
            info.textContent = `${bytes(file.size)} · ${durationText}`;
            if (total < LIMITS.clip_min - 0.05 || total > LIMITS.clip_max + 0.05) {
              info.style.color = "var(--error)";
              info.textContent += `（须 2–15 秒）`;
            }
            entry.duration = total;
            syncReferenceSummary();
          });
        }

        const actions = document.createElement("div");
        actions.className = "actions";
        const replace = document.createElement("button");
        replace.type = "button";
        replace.className = "secondary-button";
        replace.textContent = "替换";
        replace.addEventListener("click", () => {
          upload.replaceIndex = index;
          upload.input.value = "";
          upload.input.click();
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary-button";
        remove.textContent = "删除";
        remove.addEventListener("click", () => {
          upload.files.splice(index, 1);
          syncInput(upload);
          renderUploads();
          persistForm();
        });
        actions.append(replace, remove);
        figure.append(thumb, meta, actions);
        upload.preview.append(figure);
      });
    }
    syncReferenceSummary();
  }
  function totalDuration(upload) {
    return upload.entries.reduce((sum, entry) => sum + (Number.isFinite(entry.duration) ? entry.duration : 0), 0);
  }
  function syncReferenceSummary() {
    const images = uploadByName("refs").files.length;
    const videoUpload = uploadByName("video_refs");
    const audioUpload = uploadByName("audio_refs");
    const videos = videoUpload.files.length;
    const audios = audioUpload.files.length;
    $("ref-counts").textContent = `图片 ${images}/${LIMITS.images} · 视频 ${videos}/${LIMITS.videos} · 音频 ${audios}/${LIMITS.audio} · 合计 ${images + videos + audios}/${LIMITS.total}`;
    const parts = [];
    const videoSeconds = totalDuration(videoUpload);
    const audioSeconds = totalDuration(audioUpload);
    if (videos) parts.push(`参考视频合计 ${videoSeconds.toFixed(2)}s / 15s`);
    if (audios) parts.push(`独立音频合计 ${audioSeconds.toFixed(2)}s / 15s`);
    parts.push("参考顺应对应 <Picture n>、<Video n>、<Audio n>；视频自带音轨不单独计入音频。");
    $("ref-hint").textContent = parts.join(" · ");
    let warn = "";
    if (images > LIMITS.images) warn = "参考图片超过 9 张";
    else if (videos > LIMITS.videos) warn = "参考视频超过 3 个";
    else if (audios > LIMITS.audio) warn = "参考音频超过 3 个";
    else if (images + videos + audios > LIMITS.total) warn = "参考文件超过 12 个";
    else if (videoSeconds > LIMITS.clip_total + 0.05) warn = "参考视频总时长超过 15 秒";
    else if (audioSeconds > LIMITS.clip_total + 0.05) warn = "独立音频总时长超过 15 秒";
    if (warn) $("ref-hint").textContent = `⚠ ${warn}。` + $("ref-hint").textContent;
  }
  for (const upload of uploads) {
    upload.input.addEventListener("change", () => {
      const selected = Array.from(upload.input.files || []);
      if (upload.replaceIndex !== null) {
        const target = upload.replaceIndex;
        upload.replaceIndex = null;
        if (selected.length) {
          const kind = kindOf(upload);
          const problem = validateFile(kind, selected[0]);
          if (problem) { formError(problem); return; }
          upload.files[target] = selected[0];
        }
        upload.input.value = "";
        syncInput(upload);
        renderUploads();
        persistForm();
        return;
      }
      const accepted = [];
      for (const file of selected) {
        const problem = validateFile(kindOf(upload), file);
        if (problem) { formError(problem); continue; }
        accepted.push(file);
      }
      upload.files = accumulating.has(upload.name) ? upload.files.concat(accepted) : accepted.slice(0, 1);
      const cap = upload.name === "refs" ? LIMITS.images
        : upload.name === "video_refs" ? LIMITS.videos
        : upload.name === "audio_refs" ? LIMITS.audio : 1;
      if (upload.files.length > cap) {
        formError(`${refOrderLabel(upload, 0).split(" ")[0]}最多 ${cap} 个，已忽略多余文件。`);
        upload.files = upload.files.slice(0, cap);
      }
      upload.input.value = "";
      syncInput(upload);
      renderUploads();
      persistForm();
    });
    if (upload.clear) {
      upload.clear.addEventListener("click", () => {
        upload.input.value = "";
        upload.files = [];
        upload.replaceIndex = null;
        syncInput(upload);
        renderUploads();
        persistForm();
      });
    }
  }
  window.addEventListener("pagehide", () => uploads.forEach(releasePreviews));

  function mediaRefs() {
    const list = [];
    for (const upload of uploads) {
      if (!accumulating.has(upload.name)) continue;
      upload.files.forEach((file, index) => {
        list.push({ upload, file, index });
      });
    }
    return list;
  }
  function validateReferenceLimits() {
    const images = uploadByName("refs").files.length;
    const videos = uploadByName("video_refs").files.length;
    const audios = uploadByName("audio_refs").files.length;
    if (images > LIMITS.images) return `参考图片最多 ${LIMITS.images} 张。`;
    if (videos > LIMITS.videos) return `参考视频最多 ${LIMITS.videos} 个。`;
    if (audios > LIMITS.audio) return `参考音频最多 ${LIMITS.audio} 个。`;
    if (images + videos + audios > LIMITS.total) return `参考文件合计最多 ${LIMITS.total} 个。`;
    if (images + videos === 0) return "参考模式至少需要一张参考图片或一个参考视频。";
    const videoTotal = totalDuration(uploadByName("video_refs"));
    const audioTotal = totalDuration(uploadByName("audio_refs"));
    if (videoTotal > LIMITS.clip_total + 0.05) return "参考视频合计时长超过 15 秒。";
    if (audioTotal > LIMITS.clip_total + 0.05) return "独立参考音频合计时长超过 15 秒。";
    for (const upload of [uploadByName("video_refs"), uploadByName("audio_refs")]) {
      for (const entry of upload.entries) {
        if (Number.isFinite(entry.duration)
          && (entry.duration < LIMITS.clip_min - 0.05 || entry.duration > LIMITS.clip_max + 0.05)) {
          return `${upload.name === "video_refs" ? "视频" : "音频"}须为 2–15 秒（${(entry.duration).toFixed(2)}s）。`;
        }
      }
    }
    return "";
  }

  // ---------- modes ----------
  let mode = "text";
  const modeCards = $("mode-cards");
  const modeHints = {
    text: "仅使用提示词，不需要上传素材。",
    keyframes: "上传首帧、尾帧或两者：只放首帧 → i2va，只放尾帧 → l2va，两者都有 → fl2va。",
    firstlast: "同时上传首帧与尾帧，生成从首帧过渡到尾帧的视频（fl2va）。",
    references: "按顺序上传参考图 / 视频 / 音频，顺序映射 <Picture n>、<Video n>、<Audio n>；至少一张图或一个视频。",
  };
  function selectMode(next) {
    mode = next;
    for (const card of modeCards.querySelectorAll(".mode-card")) {
      card.setAttribute("aria-checked", String(card.dataset.mode === next));
    }
    $("mode-hint").textContent = modeHints[next] || "";
    $("keyframe-fields").classList.toggle("hidden", !(next === "keyframes" || next === "firstlast"));
    $("reference-fields").classList.toggle("hidden", next !== "references");
    uploadByName("first").input.disabled = !(next === "keyframes" || next === "firstlast");
    uploadByName("last").input.disabled = !(next === "keyframes" || next === "firstlast");
    for (const upload of uploads) {
      if (accumulating.has(upload.name)) upload.input.disabled = next !== "references";
    }
    persistForm();
  }
  modeCards.addEventListener("click", (event) => {
    const card = event.target.closest(".mode-card");
    if (!card) return;
    formError("");
    selectMode(card.dataset.mode);
    updatePlan();
  });

  // ---------- canvas mode / plan ----------
  let canvasMode = "mp";
  let planTimer = null;
  function selectCanvasMode(next) {
    canvasMode = next === "preset" ? "preset" : "mp";
    for (const button of $("canvas-mode").querySelectorAll("button")) {
      button.setAttribute("aria-selected", String(button.dataset.canvasMode === canvasMode));
    }
    $("mp-panel").classList.toggle("hidden", canvasMode !== "mp");
    $("preset-panel").classList.toggle("hidden", canvasMode !== "preset");
    updatePlan();
  }
  $("canvas-mode").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-canvas-mode]");
    if (!button) return;
    selectCanvasMode(button.dataset.canvasMode);
  });

  function planFields() {
    const fields = {
      canvas_mode: canvasMode,
      aspect: $("aspect").value,
      budget: $("budget").value,
      override: $("override").checked ? "true" : "false",
    };
    if (canvasMode === "mp") {
      fields.mp = $("mp").value;
      fields.duration = $("mp-duration").value;
    } else {
      fields.resolution = $("resolution").value;
      fields.duration = $("preset-duration").value;
    }
    if ($("override").checked) {
      fields.override_width = $("override-width").value.trim();
      fields.override_height = $("override-height").value.trim();
      fields.override_duration = $("override-duration").value.trim();
    }
    return fields;
  }

  function renderPlan(plan) {
    const main = $("plan-summary").querySelector(".plan-main");
    const detail = $("plan-summary").querySelector(".plan-detail");
    const notes = $("plan-summary").querySelector(".plan-notes");
    main.textContent = `${plan.width} × ${plan.height} · ${plan.frames} 帧 · ${plan.seconds.toFixed(3)}s`;
    const chips = [
      `${plan.mp.toFixed(3)} MP`,
      `${plan.aspect_ratio.toFixed(3)} 宽高比（目标 ${plan.aspect_target.toFixed(3)}）`,
      `预算 ${plan.budget_used.toFixed(1)} / ${plan.budget} MP·帧`,
      plan.auto_duration
        ? `时长自动 ${plan.duration_seconds}s（${plan.frames} 帧 = ${plan.seconds.toFixed(3)}s）`
        : `时长 ${plan.duration_seconds}s（${plan.frames} 帧 = ${plan.seconds.toFixed(3)}s）`,
    ];
    detail.textContent = chips.join(" · ");
    notes.replaceChildren();
    for (const note of plan.notes || []) {
      const item = document.createElement("li");
      item.textContent = note;
      notes.append(item);
    }
    persistForm();
  }
  function renderPlanError(message) {
    $("plan-summary").querySelector(".plan-main").textContent = "参数需要修正";
    $("plan-summary").querySelector(".plan-detail").textContent = message;
    $("plan-summary").querySelector(".plan-notes").replaceChildren();
  }
  async function updatePlan() {
    const fields = planFields();
    // Local quick validation first so the UI stays responsive without the server.
    if (canvasMode === "mp") {
      const mp = decimal($("mp"));
      if (!Number.isFinite(mp) || mp < 0.05 || mp > 8) {
        renderPlanError("MP 必须是 0.05–8 之间的数字。");
        return;
      }
    }
    const budget = decimal($("budget"));
    if (!Number.isFinite(budget) || budget < 20 || budget > 1200) {
      renderPlanError("预算必须是 20–1200 MP·帧 之间的数字。");
      return;
    }
    try {
      const response = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      });
      const data = await response.json();
      if (!response.ok) { renderPlanError(data.error || `HTTP ${response.status}`); return; }
      renderPlan(data.plan);
    } catch (_) {
      renderPlanError("无法连接服务，稍后会重试。");
    }
  }
  function schedulePlan() {
    if (planTimer !== null) window.clearTimeout(planTimer);
    planTimer = window.setTimeout(() => { planTimer = null; updatePlan(); }, 180);
  }
  for (const id of ["mp", "budget", "aspect", "resolution", "preset-duration",
    "override-width", "override-height", "override-duration"]) {
    $(id).addEventListener("input", schedulePlan);
    $(id).addEventListener("change", schedulePlan);
  }
  $("override").addEventListener("change", () => {
    $("override-inputs").classList.toggle("hidden", !$("override").checked);
    updatePlan();
  });

  // ---------- misc form state ----------
  function formError(message) {
    $("form-error").textContent = message || "";
  }
  function promptMeter() {
    const length = $("prompt").value.length;
    $("prompt-meter").textContent = `${length} / ${LIMITS.prompt} 字符`;
    $("prompt-meter").style.color = length > LIMITS.prompt ? "var(--error)" : "";
  }
  $("prompt").addEventListener("input", () => { promptMeter(); persistForm(); });

  const examples = {
    text: "A red ball bounces on a wooden table in warm afternoon light. The camera slowly pushes in; ambient room tone and a soft bounce sound.",
    keyframes: "The camera follows the scene naturally from the opening frame, with gentle motion and matching ambience.",
    firstlast: "A continuous shot moving smoothly from the opening state to the ending state, with consistent lighting and ambience.",
    references: "Use <Picture 1> as the main subject and match the lighting of <Video 1>; keep the rhythm of <Audio 1>.",
  };
  $("load-example").addEventListener("click", () => {
    $("prompt").value = examples[mode] || examples.text;
    promptMeter();
    persistForm();
  });

  function formSnapshot() {
    const fields = {};
    for (const element of $("task-form").elements) {
      if (!element.name || element.type === "file" || element.type === "submit") continue;
      // The seed has its own persistent state; the form snapshot must never
      // resurrect the previous value over a freshly drawn random seed.
      if (element.name === "seed") continue;
      if (element.type === "checkbox") fields[element.name] = element.checked;
      else fields[element.name] = element.value;
    }
    return { fields, mode, canvasMode };
  }
  let persistTimer = null;
  function persistForm() {
    if (persistTimer !== null) return;
    persistTimer = window.setTimeout(() => {
      persistTimer = null;
      try { sessionStorage.setItem(FORM_STORE, JSON.stringify(formSnapshot())); } catch (_) { /* ignore */ }
    }, 250);
  }
  function restoreForm() {
    let stored = null;
    try { stored = JSON.parse(sessionStorage.getItem(FORM_STORE) || "null"); } catch (_) { stored = null; }
    if (!stored || typeof stored !== "object" || !stored.fields) return;
    for (const [name, value] of Object.entries(stored.fields)) {
      const element = $("task-form").elements.namedItem(name);
      if (!element || element.type === "file" || name === "seed") continue;
      if (element.type === "checkbox") element.checked = Boolean(value);
      else element.value = String(value);
    }
    if (stored.mode && modeLabels[stored.mode]) mode = stored.mode;
    if (stored.canvasMode === "preset" || stored.canvasMode === "mp") canvasMode = stored.canvasMode;
  }
  $("task-form").addEventListener("input", persistForm);
  $("task-form").addEventListener("change", persistForm);

  // ---------- config bootstrap ----------
  let config = null;
  async function loadConfig() {
    try {
      const response = await fetch("/api/config", { cache: "no-store" });
      if (!response.ok) return;
      config = await response.json();
      const defaults = config.defaults || {};
      // Only fill empty fields so a restored session's overrides survive.
      for (const [id, key] of [["base-path", "base_path"], ["adapter-path", "adapter_path"],
        ["model-repo", "model_repo"], ["revision", "revision"]]) {
        if (!$(id).value.trim()) $(id).value = defaults[key] || "";
      }
      if (!$("budget").value.trim()) $("budget").value = String(config.budget ? config.budget.default : 260);
      if (config.dry_run) {
        document.body.classList.add("dry-run");
        $("queue-hint").textContent = "试运行模式：不会加载模型。";
      }
      updatePlan();
    } catch (_) { /* server unavailable; plan will show an error */ }
  }

  // ---------- task polling ----------
  let activeTaskId = null;
  let pollTimer = null;
  let logFetchedFor = null;
  let lastLogLength = 0;
  let polling = false;

  function taskMetadata(task) {
    const plan = task.plan || {};
    const parts = [];
    if (plan.width && plan.height) parts.push(`${plan.width}×${plan.height}`);
    if (plan.frames) parts.push(`${plan.frames} 帧`);
    if (plan.seconds) parts.push(`${Number(plan.seconds).toFixed(3)}s`);
    if (task.tensor_parallel) parts.push("TP×2");
    if (task.low_vram) parts.push("低显存");
    return parts;
  }
  function statusBadge(task) {
    const span = document.createElement("span");
    span.className = `badge ${task.status}`;
    span.textContent = statusLabels[task.status] || task.status;
    return span;
  }
  function renderCurrent(task) {
    const empty = $("current-empty");
    const panel = $("current-task");
    if (!task) {
      empty.classList.remove("hidden");
      panel.classList.add("hidden");
      $("current-log-link").classList.add("hidden");
      return;
    }
    empty.classList.add("hidden");
    panel.classList.remove("hidden");
    $("current-log-link").classList.remove("hidden");
    $("current-log-link").href = `/logs/${encodeURIComponent(task.id)}`;
    const badge = statusLabels[task.status] || task.status;
    $("current-status").textContent = `${badge} · ${task.mode_label || task.mode} · ${task.task}`;
    let stage = stageLabels[task.stage] || task.stage || "";
    if (task.status === "queued" && task.queue_position) stage = `排队第 ${task.queue_position} 位`;
    $("current-stage").textContent = stage;
    const progress = Math.max(0, Math.min(100, Number(task.progress) || 0));
    const bar = $("current-progress-bar");
    bar.style.width = `${progress}%`;
    bar.setAttribute("aria-valuenow", String(progress));
    // The video is served before finalization; only reveal it once the task is done.
    const playable = task.status === "done" && task.output_ready && !task.dry_run;
    const parts = [`进度 ${progress}%`];
    if (task.elapsed_seconds) parts.push(`耗时 ${task.elapsed_seconds.toFixed(1)}s`);
    parts.push(...taskMetadata(task));
    if (task.dry_run) parts.push("试运行");
    $("current-meta").textContent = parts.join(" · ");
    $("current-seed").textContent = task.seed !== undefined && task.seed !== null
      ? `本次种子: ${task.seed}` : "";
    $("current-error").textContent = task.error || "";
    const active = task.status === "queued" || task.status === "running";
    $("cancel-button").classList.toggle("hidden", !active);
    // The placeholder dry-run MP4 is downloadable but is not presented as the result.
    const downloadable = task.status === "done" && task.output_ready;
    $("current-download").classList.toggle("hidden", !downloadable);
    $("current-json").classList.toggle("hidden", !(task.status === "done" && task.metadata_ready));
    if (downloadable) {
      $("current-download").href = `/api/tasks/${encodeURIComponent(task.id)}/video`;
      $("current-download").setAttribute("download", `${task.id}.mp4`);
    }
    if (task.metadata_ready) {
      $("current-json").href = `/api/tasks/${encodeURIComponent(task.id)}/metadata`;
      $("current-json").setAttribute("download", `${task.id}.json`);
    }
    const videoPanel = $("current-video");
    const player = $("current-player");
    videoPanel.classList.toggle("hidden", !playable);
    if (playable) {
      const source = `/api/tasks/${encodeURIComponent(task.id)}/video`;
      if (player.getAttribute("src") !== source) player.src = source;
    } else {
      player.removeAttribute("src");
    }
  }
  function renderHistory(tasks) {
    const root = $("history");
    root.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "暂无历史任务。";
      root.append(empty);
      return;
    }
    for (const task of tasks) {
      const item = document.createElement("div");
      item.className = "history-item";
      const info = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${task.mode_label || task.mode} · ${task.task} `;
      const badge = statusBadge(task);
      title.append(" ");
      title.append(badge);
      const meta = document.createElement("small");
      const plan = task.plan || {};
      const bits = [task.id, plan.width && plan.height ? `${plan.width}×${plan.height}` : "",
        plan.frames ? `${plan.frames} 帧` : "", `种子 ${task.seed}`];
      meta.textContent = bits.filter(Boolean).join(" · ");
      info.append(title, meta);
      if (task.error) {
        const err = document.createElement("small");
        err.style.color = "var(--error)";
        err.textContent = task.error;
        info.append(err);
      }
      const links = document.createElement("div");
      links.style.display = "flex";
      links.style.gap = "10px";
      const logLink = document.createElement("a");
      logLink.href = `/logs/${encodeURIComponent(task.id)}`;
      logLink.textContent = "日志";
      links.append(logLink);
      if (task.status === "done" && task.output_ready) {
        const videoLink = document.createElement("a");
        videoLink.href = `/api/tasks/${encodeURIComponent(task.id)}/video`;
        videoLink.textContent = "视频";
        links.append(videoLink);
      }
      item.append(info, links);
      root.append(item);
    }
  }
  async function refreshTasks() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch("/api/tasks", { cache: "no-store" });
      if (!response.ok) throw new Error("任务列表请求失败");
      const data = await response.json();
      const tasks = Array.isArray(data.tasks) ? data.tasks : [];
      renderHistory(tasks);
      const current = activeTaskId ? tasks.find((task) => task.id === activeTaskId) : null;
      renderCurrent(current || tasks[0] || null);
      const shown = current || tasks[0];
      if (shown) {
        if (activeTaskId === null) activeTaskId = shown.id;
        await refreshLog(shown.id, shown.status);
      }
    } catch (_) {
      // Transient poll failure: keep the last known view and retry on the next tick.
    } finally {
      polling = false;
    }
  }
  async function refreshLog(taskId, status) {
    if (logFetchedFor !== taskId) {
      logFetchedFor = taskId;
      lastLogLength = 0;
    }
    try {
      const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/log`, { cache: "no-store" });
      if (!response.ok) return;
      const text = await response.text();
      const box = $("current-log");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
      if (text.length !== lastLogLength || box.textContent.length === 0) {
        box.textContent = text || "等待日志…";
        lastLogLength = text.length;
        if (atBottom) box.scrollTop = box.scrollHeight;
      }
      if (status === "done" || status === "error" || status === "canceled") activeTaskId = taskId;
    } catch (_) { /* transient */ }
  }
  $("cancel-button").addEventListener("click", async () => {
    if (!activeTaskId) return;
    if (!window.confirm("确定要取消当前任务吗？进程组会被终止。")) return;
    try {
      const response = await fetch(`/api/tasks/${encodeURIComponent(activeTaskId)}/cancel`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) formError(data.error || "取消失败");
    } catch (_) { formError("取消失败：无法连接服务。"); }
    refreshTasks();
  });

  // ---------- device status ----------
  async function refreshDevice() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      const device = data.device || {};
      const queue = data.queue || {};
      const bars = $("device-bars");
      bars.replaceChildren();
      if (!device.available) {
        $("device-text").textContent = `未检测到 NPU：${device.reason || "未知原因"}`;
      } else {
        $("device-text").textContent = device.devices.map((item) =>
          `${item.label || `NPU ${item.id}`}`).join(" · ");
        for (const item of device.devices) {
          const row = document.createElement("div");
          row.className = "device-bar";
          if ((item.hbm_percent || 0) >= 80) row.classList.add("high");
          const label = document.createElement("span");
          label.textContent = `${item.label || `NPU ${item.id}`} · HBM ${item.hbm_used_gib} / ${item.hbm_total_gib} GiB`;
          const track = document.createElement("div");
          track.className = "bar-track";
          const fill = document.createElement("div");
          fill.className = "bar-fill";
          fill.style.width = `${item.hbm_percent || 0}%`;
          track.append(fill);
          row.append(label, track);
          bars.append(row);
        }
      }
      const queueText = queue.pending > 0
        ? `队列 ${queue.pending} 个等待 · 1 个任务同时运行`
        : "空闲 · 同一时间只运行 1 个任务";
      $("queue-hint").textContent = data.dry_run ? "试运行模式" : queueText;
    } catch (_) {
      $("device-text").textContent = "设备状态暂不可用";
    }
  }

  // ---------- submission ----------
  let submitting = false;
  $("task-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitting) return;
    formError("");
    if (!validateSeed()) {
      seedInput.focus();
      // The inline seed error is the primary message; keep the summary generic.
      return;
    }
    if (!$("prompt").value.trim()) {
      formError("请输入提示词。");
      $("prompt").focus();
      return;
    }
    if (mode === "keyframes" && !uploadByName("first").files.length && !uploadByName("last").files.length) {
      formError("关键帧模式需要上传首帧、尾帧或两者。");
      return;
    }
    if (mode === "firstlast" && (!uploadByName("first").files.length || !uploadByName("last").files.length)) {
      formError("首尾帧模式需要同时上传首帧和尾帧。");
      return;
    }
    if (mode === "references") {
      const problem = validateReferenceLimits();
      if (problem) { formError(problem); return; }
    }
    let total = 0;
    for (const { file } of mediaRefs()) total += file.size;
    for (const name of ["first", "last"]) total += uploadByName(name).files.reduce((sum, file) => sum + file.size, 0);
    if (total > LIMITS.request_bytes) {
      formError("本次上传总量超过 512 MiB 限制。");
      return;
    }
    if ($("override").checked) {
      const w = $("override-width").value.trim();
      const h = $("override-height").value.trim();
      if ((w && !h) || (!w && h)) {
        formError("手动覆盖画布时请同时填写宽和高。");
        return;
      }
      for (const [label, value] of [["宽", w], ["高", h]]) {
        if (!value) continue;
        const number = Number(value);
        if (!Number.isInteger(number) || number <= 0 || number % 32 !== 0) {
          formError(`${label}必须是 32 的倍数。`);
          return;
        }
      }
    }
    const seedUsed = seedInput.value.trim();
    const body = new FormData();
    body.set("mode", mode);
    body.set("prompt", $("prompt").value);
    body.set("seed", seedUsed);
    for (const [key, value] of Object.entries(planFields())) body.set(key, value);
    body.set("low_vram", $("low-vram").checked ? "true" : "false");
    body.set("tensor_parallel", $("tensor-parallel").checked ? "true" : "false");
    body.set("vae_decode_dtype", $("vae-decode-dtype").value);
    body.set("base_path", $("base-path").value.trim());
    body.set("adapter_path", $("adapter-path").value.trim());
    body.set("model_repo", $("model-repo").value.trim());
    body.set("revision", $("revision").value.trim());
    for (const upload of uploads) {
      const disabled = upload.input.disabled;
      if (disabled) continue;
      for (const file of upload.files) {
        if (upload.name === "first") body.append("first", file, file.name);
        else if (upload.name === "last") body.append("last", file, file.name);
        else body.append(upload.name, file, file.name);
      }
    }
    // Files are only appended from the ordered `uploads` list above.
    submitting = true;
    $("submit-button").disabled = true;
    $("submit-button").textContent = "正在提交…";
    try {
      let response;
      try {
        response = await fetch("/api/tasks", { method: "POST", body });
      } catch (_) {
        throw new Error("无法连接服务，请确认 python -m webui.server 正在运行。");
      }
      let data = null;
      try { data = await response.json(); } catch (_) { data = null; }
      if (!response.ok) throw new Error((data && data.error) || `提交失败：HTTP ${response.status}`);
      if (!data || !data.id) throw new Error("提交失败：服务未返回任务编号。");
      activeTaskId = data.id;
      logFetchedFor = null;
      lastLogLength = 0;
      $("current-log").textContent = "等待日志…";
      if (seedMode === "random") {
        // Refresh immediately on POST success so a second submit never reuses this seed.
        seedInput.value = randomSeedString();
        setSeedMode("random", { keepValue: true });
      } else {
        saveSeedState();
      }
      persistForm();
      await refreshTasks();
      await refreshDevice();
    } catch (error) {
      formError(error.message || String(error));
    } finally {
      submitting = false;
      $("submit-button").disabled = false;
      $("submit-button").textContent = "开始生成";
    }
  });

  // ---------- boot ----------
  function boot() {
    initSeed();
    restoreForm();
    selectMode(mode);
    promptMeter();
    $("override-inputs").classList.toggle("hidden", !$("override").checked);
    selectCanvasMode(canvasMode);
    loadConfig();
    refreshTasks();
    refreshDevice();
    pollTimer = window.setInterval(() => { refreshTasks(); refreshDevice(); }, 1200);
  }
  window.addEventListener("beforeunload", () => {
    if (pollTimer !== null) window.clearInterval(pollTimer);
  });
  boot();
})();
