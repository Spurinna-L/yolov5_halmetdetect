(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // =========================================================================
  // Global State
  // =========================================================================
  let selectedImageFile = null;
  let imageOriginObjectUrl = null;
  let selectedVideoObjectUrl = null;

  const streamState = {
    ws: null,
    running: false,
    paused: false,
    inFlight: false,
    sentAtMs: 0,
    processedFrames: 0,
    consecutiveAlarmFrames: 0,
    lastSnapshotAtMs: 0,
    snapshots: [],
    offscreenCanvas: document.createElement("canvas"),
  };

  // =========================================================================
  // Utilities
  // =========================================================================
  function formatTimecode(seconds) {
    const safeSec = Math.max(0, Number(seconds) || 0);
    const mins = Math.floor(safeSec / 60);
    const secs = safeSec % 60;
    return `${String(mins).padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
  }

  function setStatusBar(leftText, perfText) {
    if (leftText !== null && leftText !== undefined) {
      $("statusBarLeft").textContent = leftText;
    }
    if (perfText !== null && perfText !== undefined) {
      $("statusBarPerf").textContent = perfText;
    }
  }

  function updateKpiPanel(prefix, data) {
    const totalEl = $(`${prefix}KpiTotal`);
    const helmetEl = $(`${prefix}KpiHelmet`);
    const headEl = $(`${prefix}KpiHead`);
    const rateEl = $(`${prefix}KpiRate`);

    if (!data) {
      totalEl.textContent = "--";
      helmetEl.textContent = "--";
      headEl.textContent = "--";
      rateEl.textContent = "--";
      return;
    }

    totalEl.textContent = String(data.total ?? 0);
    helmetEl.textContent = String(data.helmet_count ?? 0);
    headEl.textContent = String(data.head_count ?? 0);
    rateEl.textContent =
      data.compliance_rate === null || data.compliance_rate === undefined
        ? "N/A"
        : `${data.compliance_rate}%`;
  }

  function setAlarmBanner(bannerEl, mode, title, desc) {
    bannerEl.classList.remove("alarm-idle", "alarm-safe", "alarm-danger");
    bannerEl.classList.add(mode);
    bannerEl.querySelector(".alarm-title").textContent = title;
    bannerEl.querySelector(".alarm-desc").textContent = desc;
  }

  function openLightbox(src, title) {
    if (!src) return;
    $("lightboxImg").src = src;
    $("lightboxTitle").textContent = title || "预览";
    $("lightboxModal").classList.remove("hidden");
  }

  function closeLightbox() {
    $("lightboxModal").classList.add("hidden");
    $("lightboxImg").src = "";
  }

  // =========================================================================
  // Backend Health Check
  // =========================================================================
  async function checkBackendStatus() {
    const pill = $("backendStatusPill");
    const textEl = $("backendStatusText");
    try {
      const resp = await fetch("/api/status");
      const data = await resp.json();
      pill.classList.remove("status-loading", "status-ok", "status-err");
      if (data.ready) {
        pill.classList.add("status-ok");
        const provider = (data.providers && data.providers[0]) || "CPU";
        const shortProv = provider.replace("ExecutionProvider", "");
        textEl.textContent = `${data.model_name} (${shortProv})`;
        setStatusBar("就绪 · ONNX 模型已加载", `${data.input_size}×${data.input_size}`);
      } else {
        pill.classList.add("status-err");
        textEl.textContent = `未就绪: ${data.error || "检查 best.onnx"}`;
        setStatusBar("模型未就绪", "--");
      }
    } catch (_err) {
      pill.classList.remove("status-loading", "status-ok");
      pill.classList.add("status-err");
      textEl.textContent = "后端离线";
      setStatusBar("无法连接 WSL 后端服务", "--");
    }
  }

  // =========================================================================
  // Tab Navigation
  // =========================================================================
  function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        tabBtns.forEach((b) => {
          const isActive = b === btn;
          b.classList.toggle("active", isActive);
          b.setAttribute("aria-selected", String(isActive));
        });
        tabPanels.forEach((panel) => {
          panel.classList.toggle("active", panel.id === `tab-${target}`);
        });

        if (target !== "video" && streamState.running) {
          stopVideoStream("已切换页面并停止流式检测");
        }
        if (target === "image") {
          setStatusBar("图片检测模式", $("imgPerfBadge").textContent);
        } else {
          setStatusBar("流式视频检测模式", $("vidStreamBadge").textContent);
        }
      });
    });
  }

  // =========================================================================
  // DropZone Helper
  // =========================================================================
  function bindDropZone(zoneEl, inputEl, onFileSelected) {
    zoneEl.addEventListener("click", () => inputEl.click());
    zoneEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        inputEl.click();
      }
    });
    zoneEl.addEventListener("dragover", (e) => {
      e.preventDefault();
      zoneEl.classList.add("drag-over");
    });
    zoneEl.addEventListener("dragleave", () => zoneEl.classList.remove("drag-over"));
    zoneEl.addEventListener("drop", (e) => {
      e.preventDefault();
      zoneEl.classList.remove("drag-over");
      const file = e.dataTransfer?.files?.[0];
      if (file) onFileSelected(file);
    });
    inputEl.addEventListener("change", () => {
      const file = inputEl.files?.[0];
      if (file) onFileSelected(file);
    });
  }

  // =========================================================================
  // Tab 1: Image Detection
  // =========================================================================
  function handleImageSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("请选择有效的图片文件（JPG / PNG / BMP / WEBP）");
      return;
    }
    selectedImageFile = file;
    $("imgFileName").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    if (imageOriginObjectUrl) URL.revokeObjectURL(imageOriginObjectUrl);
    imageOriginObjectUrl = URL.createObjectURL(file);

    const originImg = $("imgOriginPreview");
    originImg.src = imageOriginObjectUrl;
    originImg.classList.remove("hidden");
    $("imgOriginEmpty").classList.add("hidden");

    $("imgResultPreview").classList.add("hidden");
    $("imgResultEmpty").classList.remove("hidden");
    $("imgDownloadBtn").classList.add("disabled");
    $("imgDownloadBtn").removeAttribute("href");
    $("imgDetectBtn").disabled = false;
    $("imgPerfBadge").textContent = "待推理";
    setStatusBar(`已加载图片: ${file.name}`, "待推理");
  }

  function renderImageDetectionsList(detections) {
    const listEl = $("imgDetList");
    $("imgDetCount").textContent = `${detections.length} 个目标`;

    if (!detections.length) {
      listEl.innerHTML = '<div class="empty-list">当前阈值下未检测到头部或安全帽目标</div>';
      return;
    }

    listEl.innerHTML = detections
      .map((det, idx) => {
        const cls = det.class_name;
        const chipCls = cls === "helmet" ? "tag-helmet" : cls === "head" ? "tag-head" : "tag-person";
        const pct = (det.confidence * 100).toFixed(1);
        return `
          <div class="det-item">
            <div class="det-item-top">
              <span class="tag-chip ${chipCls}">#${idx + 1} ${det.label_cn}</span>
              <span class="mono-val">${pct}%</span>
            </div>
            <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
          </div>
        `;
      })
      .join("");
  }

  async function runImageDetection() {
    if (!selectedImageFile) return;

    const btn = $("imgDetectBtn");
    btn.disabled = true;
    btn.textContent = "推理中...";
    $("imgPerfBadge").textContent = "推理中...";
    setStatusBar("正在执行 ONNX 图像推理...", "处理中");

    const formData = new FormData();
    formData.append("file", selectedImageFile);
    formData.append("conf", $("imgConfSlider").value);
    formData.append("iou", $("imgIouSlider").value);
    formData.append("draw_hud", String($("imgHudCheckbox").checked));

    try {
      const resp = await fetch("/api/detect/image", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "图片推理失败");
      }

      const resultImg = $("imgResultPreview");
      resultImg.src = `${data.output_url}?t=${Date.now()}`;
      resultImg.classList.remove("hidden");
      $("imgResultEmpty").classList.add("hidden");

      const dlBtn = $("imgDownloadBtn");
      dlBtn.href = data.output_url;
      dlBtn.download = data.filename || "helmet_detected.jpg";
      dlBtn.classList.remove("disabled");

      const perfStr = `${data.width}×${data.height} | ${data.infer_ms} ms | ${data.fps} FPS`;
      $("imgPerfBadge").textContent = perfStr;
      setStatusBar("图片检测完成", perfStr);
      updateKpiPanel("img", data);
      renderImageDetectionsList(data.detections || []);

      const banner = $("imgAlarmBanner");
      if (data.is_alarm) {
        setAlarmBanner(
          banner,
          "alarm-danger",
          `未佩戴告警 (${data.head_count} 人)`,
          `安全帽佩戴率 ${data.compliance_rate ?? 0}%`
        );
      } else if (data.total > 0) {
        setAlarmBanner(
          banner,
          "alarm-safe",
          "全员佩戴合规",
          `已规范佩戴 ${data.helmet_count} 人`
        );
      } else {
        setAlarmBanner(
          banner,
          "alarm-idle",
          "未检测到目标",
          "可调低 Conf 阈值后重试"
        );
      }
    } catch (err) {
      $("imgPerfBadge").textContent = "推理失败";
      setStatusBar(`检测失败: ${err.message}`, "--");
      alert(`检测失败: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "开始检测";
    }
  }

  function initImageTab() {
    bindDropZone($("imgDropZone"), $("imgFileInput"), handleImageSelected);

    $("imgConfSlider").addEventListener("input", (e) => {
      $("imgConfVal").textContent = Number(e.target.value).toFixed(2);
    });
    $("imgIouSlider").addEventListener("input", (e) => {
      $("imgIouVal").textContent = Number(e.target.value).toFixed(2);
    });
    $("imgDetectBtn").addEventListener("click", runImageDetection);

    $("imgOriginPreview").addEventListener("click", () =>
      openLightbox($("imgOriginPreview").src, "原始输入图像")
    );
    $("imgResultPreview").addEventListener("click", () =>
      openLightbox($("imgResultPreview").src, "ONNX 检测结果图")
    );
  }

  // =========================================================================
  // Tab 2: Streaming Video Detection (WebSocket Backpressure Loop)
  // =========================================================================
  function handleVideoSelected(file) {
    if (!file.type.startsWith("video/")) {
      alert("请选择有效的视频文件（MP4 / MOV / WEBM）");
      return;
    }

    stopVideoStream();
    $("vidFileName").textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;

    if (selectedVideoObjectUrl) URL.revokeObjectURL(selectedVideoObjectUrl);
    selectedVideoObjectUrl = URL.createObjectURL(file);

    const videoEl = $("vidOriginPlayer");
    videoEl.src = selectedVideoObjectUrl;
    videoEl.playbackRate = Number($("vidSpeedSelect").value) || 1.0;
    videoEl.classList.remove("hidden");
    $("vidOriginEmpty").classList.add("hidden");

    $("vidStartBtn").disabled = false;
    $("vidStreamBadge").textContent = "视频已加载";
    setStatusBar(`已加载视频: ${file.name}`, "待推流");
  }

  function renderSnapshotsList() {
    const listEl = $("vidSnapshotList");
    $("vidSnapshotCount").textContent = `${streamState.snapshots.length} 次告警`;

    if (!streamState.snapshots.length) {
      listEl.innerHTML =
        '<div class="empty-list">暂无违规抓拍记录（点击抓拍卡片可跳转对应时间点复查）</div>';
      return;
    }

    listEl.innerHTML = streamState.snapshots
      .map(
        (snap, idx) => `
        <div class="snapshot-card" data-snap-index="${idx}" title="点击跳转至 ${snap.timeStr} 查看">
          <img class="snapshot-thumb" src="${snap.thumbnail}" alt="违规抓拍 ${snap.timeStr}" />
          <div class="snapshot-meta">
            <div class="snapshot-time">${snap.timeStr} · 未戴 ${snap.headCount}</div>
            <div class="snapshot-detail">已佩戴 ${snap.helmetCount} · 点击复查</div>
          </div>
        </div>
      `
      )
      .join("");

    listEl.querySelectorAll(".snapshot-card").forEach((card) => {
      card.addEventListener("click", () => {
        const snap = streamState.snapshots[Number(card.dataset.snapIndex)];
        if (snap) reviewSnapshot(snap);
      });
    });
  }

  function reviewSnapshot(snap) {
    const videoEl = $("vidOriginPlayer");
    if (streamState.running && !streamState.paused) {
      togglePauseVideoStream();
    }
    videoEl.currentTime = snap.timestampSec;
    setStatusBar(`回看违规抓拍 (${snap.timeStr})`, `时间 ${snap.timeStr}`);

    const canvas = $("vidResultCanvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      canvas.classList.remove("hidden");
      $("vidResultEmpty").classList.add("hidden");
    };
    img.src = snap.thumbnail;
  }

  function drawCanvasAndHandleResult(msg) {
    const canvas = $("vidResultCanvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      canvas.classList.remove("hidden");
      $("vidResultEmpty").classList.add("hidden");
    };
    img.src = msg.annotated_image;

    streamState.processedFrames += 1;
    const roundTripMs = Math.max(1, performance.now() - streamState.sentAtMs);
    const liveFps = 1000 / roundTripMs;
    const timeStr = formatTimecode(msg.timestamp);
    const perfLine = `帧 #${streamState.processedFrames} | ${liveFps.toFixed(1)} FPS | ${Math.round(roundTripMs)} ms | ${timeStr}`;

    $("vidStreamBadge").textContent = `${liveFps.toFixed(1)} FPS · ${Math.round(roundTripMs)} ms`;
    setStatusBar("WS 实时推流推理中", perfLine);
    updateKpiPanel("vid", msg);

    const alarmThreshold = Number($("vidAlarmFramesSlider").value) || 3;
    const cooldownMs = (Number($("vidCooldownSlider").value) || 3) * 1000;
    const banner = $("vidAlarmBanner");

    if (msg.head_count > 0) {
      streamState.consecutiveAlarmFrames += 1;
      if (streamState.consecutiveAlarmFrames >= alarmThreshold) {
        setAlarmBanner(
          banner,
          "alarm-danger",
          `违规告警 · 未戴 ${msg.head_count} 人`,
          `连续 ${streamState.consecutiveAlarmFrames} 帧确认 · 佩戴率 ${msg.compliance_rate ?? 0}%`
        );

        const nowMs = Date.now();
        if (nowMs - streamState.lastSnapshotAtMs >= cooldownMs) {
          streamState.lastSnapshotAtMs = nowMs;
          streamState.snapshots.unshift({
            timestampSec: Number(msg.timestamp) || 0,
            timeStr,
            headCount: msg.head_count,
            helmetCount: msg.helmet_count,
            thumbnail: msg.annotated_image,
          });
          if (streamState.snapshots.length > 30) {
            streamState.snapshots.pop();
          }
          renderSnapshotsList();
        }
      } else {
        setAlarmBanner(
          banner,
          "alarm-idle",
          `防误报确认 (${streamState.consecutiveAlarmFrames}/${alarmThreshold})`,
          "过滤瞬时帧抖动"
        );
      }
    } else {
      streamState.consecutiveAlarmFrames = 0;
      if (msg.total > 0) {
        setAlarmBanner(
          banner,
          "alarm-safe",
          "监测合规 · 全员佩戴",
          `当前画面共 ${msg.helmet_count} 人已佩戴`
        );
      } else {
        setAlarmBanner(
          banner,
          "alarm-idle",
          "监测中 · 暂无目标",
          "画面持续推理中"
        );
      }
    }
  }

  function scheduleNextFrame() {
    if (!streamState.running || streamState.paused || streamState.inFlight) return;
    requestAnimationFrame(captureAndSendFrame);
  }

  function captureAndSendFrame() {
    if (!streamState.running || streamState.paused || streamState.inFlight) return;
    const ws = streamState.ws;
    const videoEl = $("vidOriginPlayer");
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (videoEl.ended) {
      stopVideoStream("流式检测已完成");
      return;
    }

    if (videoEl.readyState < 2 || videoEl.videoWidth === 0) {
      requestAnimationFrame(captureAndSendFrame);
      return;
    }

    const maxWidth = 960;
    const scale = Math.min(1, maxWidth / videoEl.videoWidth);
    const targetW = Math.max(1, Math.round(videoEl.videoWidth * scale));
    const targetH = Math.max(1, Math.round(videoEl.videoHeight * scale));

    const offscreen = streamState.offscreenCanvas;
    offscreen.width = targetW;
    offscreen.height = targetH;
    const ctx = offscreen.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, targetW, targetH);

    const frameDataUrl = offscreen.toDataURL("image/jpeg", 0.78);
    streamState.inFlight = true;
    streamState.sentAtMs = performance.now();

    ws.send(
      JSON.stringify({
        image: frameDataUrl,
        conf: Number($("vidConfSlider").value),
        iou: Number($("vidIouSlider").value),
        draw_hud: $("vidHudCheckbox").checked,
        timestamp: videoEl.currentTime,
      })
    );
  }

  function startVideoStream() {
    const videoEl = $("vidOriginPlayer");
    if (!videoEl.src) return;

    stopVideoStream();
    streamState.running = true;
    streamState.paused = false;
    streamState.inFlight = false;
    streamState.processedFrames = 0;
    streamState.consecutiveAlarmFrames = 0;

    $("vidStartBtn").disabled = true;
    $("vidPauseBtn").disabled = false;
    $("vidPauseBtn").textContent = "暂停";
    $("vidStopBtn").disabled = false;
    $("vidStreamBadge").textContent = "连接 WS...";
    setStatusBar("正在建立 WebSocket 推流连接...", "--");

    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProtocol}//${location.host}/ws/stream`);
    streamState.ws = ws;

    ws.onopen = () => {
      $("vidStreamBadge").textContent = "推流中";
      setStatusBar("WS 实时推流推理中", "准备接收首帧...");
      videoEl.playbackRate = Number($("vidSpeedSelect").value) || 1.0;
      if (videoEl.ended) videoEl.currentTime = 0;
      videoEl.play().catch(() => {});
      scheduleNextFrame();
    };

    ws.onmessage = (event) => {
      let msg = null;
      try {
        msg = JSON.parse(event.data);
      } catch (_err) {
        streamState.inFlight = false;
        scheduleNextFrame();
        return;
      }

      streamState.inFlight = false;
      if (msg.type === "error") {
        $("vidStreamBadge").textContent = msg.message;
      } else if (msg.type === "result") {
        drawCanvasAndHandleResult(msg);
      }
      scheduleNextFrame();
    };

    ws.onerror = () => {
      $("vidStreamBadge").textContent = "WS 异常";
      stopVideoStream("WebSocket 连接中断");
    };

    ws.onclose = () => {
      if (streamState.running) {
        stopVideoStream("WebSocket 连接已关闭");
      }
    };
  }

  function togglePauseVideoStream() {
    if (!streamState.running) return;
    const videoEl = $("vidOriginPlayer");
    streamState.paused = !streamState.paused;
    if (streamState.paused) {
      videoEl.pause();
      $("vidPauseBtn").textContent = "继续";
      $("vidStreamBadge").textContent = "已暂停";
      setStatusBar("流式检测已暂停", `帧 #${streamState.processedFrames}`);
    } else {
      videoEl.play().catch(() => {});
      $("vidPauseBtn").textContent = "暂停";
      $("vidStreamBadge").textContent = "推流中";
      setStatusBar("WS 实时推流推理中", `帧 #${streamState.processedFrames}`);
      scheduleNextFrame();
    }
  }

  function stopVideoStream(statusLabel) {
    const videoEl = $("vidOriginPlayer");
    streamState.running = false;
    streamState.paused = false;
    streamState.inFlight = false;
    streamState.consecutiveAlarmFrames = 0;

    if (videoEl && !videoEl.paused) {
      videoEl.pause();
    }

    if (streamState.ws) {
      const oldWs = streamState.ws;
      streamState.ws = null;
      if (oldWs.readyState === WebSocket.OPEN || oldWs.readyState === WebSocket.CONNECTING) {
        oldWs.close();
      }
    }

    $("vidStartBtn").disabled = !videoEl.src;
    $("vidPauseBtn").disabled = true;
    $("vidPauseBtn").textContent = "暂停";
    $("vidStopBtn").disabled = true;
    if (statusLabel) {
      $("vidStreamBadge").textContent = statusLabel;
      setStatusBar(statusLabel, `共处理 ${streamState.processedFrames} 帧`);
    } else if (videoEl.src) {
      $("vidStreamBadge").textContent = "已停止";
    }
  }

  function initVideoTab() {
    bindDropZone($("vidDropZone"), $("vidFileInput"), handleVideoSelected);

    $("vidConfSlider").addEventListener("input", (e) => {
      $("vidConfVal").textContent = Number(e.target.value).toFixed(2);
    });
    $("vidIouSlider").addEventListener("input", (e) => {
      $("vidIouVal").textContent = Number(e.target.value).toFixed(2);
    });
    $("vidAlarmFramesSlider").addEventListener("input", (e) => {
      $("vidAlarmFramesVal").textContent = `${e.target.value} 帧`;
    });
    $("vidCooldownSlider").addEventListener("input", (e) => {
      $("vidCooldownVal").textContent = `${e.target.value} 秒`;
    });
    $("vidSpeedSelect").addEventListener("change", (e) => {
      $("vidOriginPlayer").playbackRate = Number(e.target.value) || 1.0;
    });

    $("vidStartBtn").addEventListener("click", startVideoStream);
    $("vidPauseBtn").addEventListener("click", togglePauseVideoStream);
    $("vidStopBtn").addEventListener("click", () => stopVideoStream("已手动停止检测"));
    $("vidClearSnapshotsBtn").addEventListener("click", () => {
      streamState.snapshots = [];
      renderSnapshotsList();
    });
  }

  // =========================================================================
  // Bootstrap
  // =========================================================================
  function initLightbox() {
    $("lightboxCloseBtn").addEventListener("click", closeLightbox);
    document.querySelector(".lightbox-backdrop").addEventListener("click", closeLightbox);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeLightbox();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initImageTab();
    initVideoTab();
    initLightbox();
    checkBackendStatus();
  });
})();
