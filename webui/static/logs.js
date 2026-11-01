(() => {
  "use strict";
  const id = window.WEBUI_TASK_ID;
  const log = document.getElementById("task-log");
  const videoPanel = document.getElementById("task-video");
  const video = document.getElementById("video-player");
  const statusLabels = { queued: "排队中", running: "生成中", done: "已完成", error: "失败", canceled: "已取消" };
  const stageLabels = {
    queued: "等待执行", prepare: "准备", loading: "加载模型", conditioning: "条件编码",
    sampling: "采样去噪", decode: "解码音视频", encode: "封装 MP4", done: "完成",
    error: "失败", canceled: "已取消",
  };
  let polling = false;
  let lastLength = 0;
  let hasStatus = false;
  let currentStatus = "queued";

  async function refreshStatus() {
    const response = await fetch(`/api/tasks/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!response.ok) throw new Error("任务状态请求失败");
    const task = await response.json();
    hasStatus = true;
    currentStatus = task.status;
    document.getElementById("task-status").textContent =
      `${statusLabels[task.status] || task.status} · ${task.mode_label || task.mode || ""} · ${task.task || ""}`;
    document.getElementById("task-stage").textContent =
      task.status === "queued" && task.queue_position
        ? `排队第 ${task.queue_position} 位`
        : stageLabels[task.stage] || task.stage || "";
    const plan = task.plan || {};
    const progress = Math.max(0, Math.min(100, Number(task.progress) || 0));
    const bar = document.getElementById("task-progress-bar");
    bar.style.width = `${progress}%`;
    bar.setAttribute("aria-valuenow", String(progress));
    const bits = [`进度 ${progress}%`];
    if (plan.width && plan.height) bits.push(`${plan.width} × ${plan.height}`);
    if (plan.frames) bits.push(`${plan.frames} 帧`);
    if (plan.seconds) bits.push(`${Number(plan.seconds).toFixed(3)} 秒`);
    if (task.elapsed_seconds) bits.push(`耗时 ${Number(task.elapsed_seconds).toFixed(1)}s`);
    if (task.dry_run) bits.push("试运行");
    document.getElementById("task-meta").textContent = [task.id, ...bits].join(" · ");
    document.getElementById("task-plan").textContent = plan.width
      ? `画布 ${plan.width}×${plan.height} · ${plan.frames} 帧 · 实际 ${Number(plan.seconds).toFixed(3)} 秒 · 预算 ${plan.budget_used}/${plan.budget} MP·帧`
      : "";
    document.getElementById("task-seed").textContent =
      task.seed !== undefined && task.seed !== null ? `本次种子: ${task.seed}` : "";
    document.getElementById("task-error").textContent = task.error || "";
    const active = task.status === "queued" || task.status === "running";
    document.getElementById("cancel-task").classList.toggle("hidden", !active);
    const done = task.status === "done" && task.output_ready;
    const videoLink = document.getElementById("download-video");
    videoLink.classList.toggle("hidden", !done);
    if (done) {
      const source = `/api/tasks/${encodeURIComponent(id)}/video`;
      videoLink.href = source;
      videoLink.setAttribute("download", `${id}.mp4`);
    }
    const jsonLink = document.getElementById("download-json");
    jsonLink.classList.toggle("hidden", !task.metadata_ready);
    if (task.metadata_ready) {
      jsonLink.href = `/api/tasks/${encodeURIComponent(id)}/metadata`;
      jsonLink.setAttribute("download", `${id}.json`);
    }
    const playable = done && !task.dry_run;
    videoPanel.classList.toggle("hidden", !playable);
    if (playable) {
      const source = `/api/tasks/${encodeURIComponent(id)}/video`;
      if (video.getAttribute("src") !== source) video.src = source;
    } else {
      video.removeAttribute("src");
    }
  }

  async function refreshLog() {
    const response = await fetch(`/api/tasks/${encodeURIComponent(id)}/log`, { cache: "no-store" });
    if (!response.ok) throw new Error("日志请求失败");
    const text = await response.text();
    if (text.length === lastLength && log.textContent.length > 0) return;
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
    log.textContent = text || "暂无日志，等待任务开始…";
    lastLength = text.length;
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      const [statusResult, logResult] = await Promise.allSettled([refreshStatus(), refreshLog()]);
      const errors = [];
      if (statusResult.status === "rejected") {
        errors.push("任务状态暂时无法获取，将自动重试。");
        if (!hasStatus) document.getElementById("task-status").textContent = "状态暂不可用";
      }
      if (logResult.status === "rejected") {
        errors.push("日志暂时无法获取，将自动重试。");
        if (log.textContent.length === 0) log.textContent = "日志暂不可用，正在重试…";
      }
      document.getElementById("task-refresh-error").textContent = errors.join(" ");
    } finally {
      polling = false;
    }
  }

  document.getElementById("cancel-task").addEventListener("click", async () => {
    if (!window.confirm("确定要取消当前任务吗？")) return;
    try {
      const response = await fetch(`/api/tasks/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      if (!response.ok) {
        const data = await response.json();
        document.getElementById("task-refresh-error").textContent = data.error || "取消失败";
      }
    } catch (_) {
      document.getElementById("task-refresh-error").textContent = "取消失败：无法连接服务。";
    }
    refresh();
  });

  refresh();
  const timer = window.setInterval(() => {
    if (currentStatus === "done" || currentStatus === "error" || currentStatus === "canceled") {
      // Still poll a few more times so the last log lines arrive, then stop.
      if (lastLength > 0) { window.clearInterval(timer); }
    }
    refresh();
  }, 1200);
})();
