(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // =========================================================================
  // Global State
  // =========================================================================
  let selectedImageFile = null;
  let imageOriginObjectUrl = null;
  let selectedVideoObjectUrl = null;
  let trainingLoaded = false;

  const streamState = {
    ws: null,
    running: false,
    paused: false,
    inFlight: false,
    sentAtMs: 0,
    processedFrames: 0,
    consecutiveAlarmFrames: 0,
    lastSnapshotAtMs: 0,
    alarmTriggered: false,
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
    rateEl.textContent = data.compliance_rate === null || data.compliance_rate === undefined
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
    $("lightboxTitle").textContent = title || "图片预览";
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
        textEl.textContent = `后端在线 · ${data.model_name} (${shortProv})`;
      } else {
        pill.classList.add("status-err");
        textEl.textContent = `模型未就绪: ${data.error || "检查 best.onnx"}`;
      }
    } catch (_err) {
      pill.classList.remove("status-loading", "status-ok");
      pill.classList.add("status-err");
      textEl.textContent = "无法连接后端服务";
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

        // Cleanly pause/stop active streaming when leaving the video tab.
        if (target !== "video" && streamState.running) {
          stopVideoStream();
        }
        if (target === "training" && !trainingLoaded) {
          loadTrainingReport();
        }
      });
    });
  }

  // =========================================================================
  // Tab 1: Image Detection
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
    $("imgPerfBadge").textContent = "图片已就绪，点击「开始检测」";
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
    $("imgPerfBadge").textContent = "正在通过 WSL ONNX Runtime 推理...";

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

      $("imgPerfBadge").textContent = `${data.width}×${data.height} · ${data.infer_ms} ms (${data.fps} FPS)`;
      updateKpiPanel("img", data);
      renderImageDetectionsList(data.detections || []);

      const banner = $("imgAlarmBanner");
      if (data.is_alarm) {
        setAlarmBanner(
          banner,
          "alarm-danger",
          `发现未佩戴安全帽告警 (${data.head_count} 人)`,
          `当前安全帽佩戴率 ${data.compliance_rate ?? 0}% ，请及时督促整改`
        );
      } else if (data.total > 0) {
        setAlarmBanner(
          banner,
          "alarm-safe",
          "安全合规 · 全员已佩戴安全帽",
          `共检测到 ${data.helmet_count} 人全部规范佩戴安全帽`
        );
      } else {
        setAlarmBanner(
          banner,
          "alarm-idle",
          "未检测到人员头部目标",
          "可尝试适当调低左侧置信度阈值后重新检测"
        );
      }
    } catch (err) {
      $("imgPerfBadge").textContent = `检测失败: ${err.message}`;
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
    $("vidStreamBadge").textContent = "视频已加载，点击「开始流式检测」";
  }

  function renderSnapshotsList() {
    const listEl = $("vidSnapshotList");
    $("vidSnapshotCount").textContent = `${streamState.snapshots.length} 次告警`;

    if (!streamState.snapshots.length) {
      listEl.innerHTML =
        '<div class="empty-list">暂无违规抓拍记录（点击抓拍卡片可跳转视频对应时间点复查）</div>';
      return;
    }

    listEl.innerHTML = streamState.snapshots
      .map(
        (snap, idx) => `
        <div class="snapshot-card" data-snap-index="${idx}" title="点击跳转至 ${snap.timeStr} 查看">
          <img class="snapshot-thumb" src="${snap.thumbnail}" alt="违规抓拍 ${snap.timeStr}" />
          <div class="snapshot-meta">
            <div class="snapshot-time">⏱ ${snap.timeStr} · 未戴 ${snap.headCount} 人</div>
            <div class="snapshot-detail">已佩戴 ${snap.helmetCount} 人 · 点击回看该帧</div>
          </div>
        </div>
      `
      )
      .join("");

    listEl.querySelectorAll(".snapshot-card").forEach((card) => {
      card.addEventListener("click", () => {
        const snap = streamState.snapshots[Number(card.dataset.snapIndex)];
        if (!snap) return;
        reviewSnapshot(snap);
      });
    });
  }

  function reviewSnapshot(snap) {
    const videoEl = $("vidOriginPlayer");
    if (streamState.running && !streamState.paused) {
      togglePauseVideoStream();
    }
    videoEl.currentTime = snap.timestampSec;
    $("vidCurrentTime").textContent = snap.timeStr;

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

    $("vidProcessedFrames").textContent = String(streamState.processedFrames);
    $("vidLiveFps").textContent = `${liveFps.toFixed(1)} FPS`;
    $("vidLiveLatency").textContent = `${Math.round(roundTripMs)} ms`;
    $("vidCurrentTime").textContent = formatTimecode(msg.timestamp);

    updateKpiPanel("vid", msg);

    const alarmThreshold = Number($("vidAlarmFramesSlider").value) || 3;
    const cooldownMs = (Number($("vidCooldownSlider").value) || 3) * 1000;
    const banner = $("vidAlarmBanner");

    if (msg.head_count > 0) {
      streamState.consecutiveAlarmFrames += 1;
      if (streamState.consecutiveAlarmFrames >= alarmThreshold) {
        streamState.alarmTriggered = true;
        setAlarmBanner(
          banner,
          "alarm-danger",
          `持续违规告警 · 检测到 ${msg.head_count} 人未戴安全帽`,
          `已连续 ${streamState.consecutiveAlarmFrames} 帧触发告警，合规率 ${msg.compliance_rate ?? 0}%`
        );

        const nowMs = Date.now();
        if (nowMs - streamState.lastSnapshotAtMs >= cooldownMs) {
          streamState.lastSnapshotAtMs = nowMs;
          streamState.snapshots.unshift({
            timestampSec: Number(msg.timestamp) || 0,
            timeStr: formatTimecode(msg.timestamp),
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
          `疑似未佩戴 (${streamState.consecutiveAlarmFrames}/${alarmThreshold} 帧确认中)`,
          "正在通过连续帧防误报窗口过滤瞬时抖动"
        );
      }
    } else {
      streamState.consecutiveAlarmFrames = 0;
      streamState.alarmTriggered = false;
      if (msg.total > 0) {
        setAlarmBanner(
          banner,
          "alarm-safe",
          "实时监测合规 · 全员佩戴安全帽",
          `当前画面共 ${msg.helmet_count} 人均已佩戴安全帽`
        );
      } else {
        setAlarmBanner(
          banner,
          "alarm-idle",
          "实时监测中 · 当前帧暂无头部目标",
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
      stopVideoStream("视频播放与流式检测已完成");
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
    $("vidStreamBadge").textContent = "正在建立 WebSocket 推流连接...";

    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProtocol}//${location.host}/ws/stream`);
    streamState.ws = ws;

    ws.onopen = () => {
      $("vidStreamBadge").textContent = "WS 实时推流推理中";
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
        $("vidStreamBadge").textContent = `推理提示: ${msg.message}`;
      } else if (msg.type === "result") {
        drawCanvasAndHandleResult(msg);
      }
      scheduleNextFrame();
    };

    ws.onerror = () => {
      $("vidStreamBadge").textContent = "WebSocket 连接异常";
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
      $("vidStreamBadge").textContent = "流式检测已暂停";
    } else {
      videoEl.play().catch(() => {});
      $("vidPauseBtn").textContent = "暂停";
      $("vidStreamBadge").textContent = "WS 实时推流推理中";
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
  // Tab 3: Training Report
  // =========================================================================
  async function loadTrainingReport() {
    try {
      const resp = await fetch("/api/training-report");
      const data = await resp.json();
      trainingLoaded = true;

      const s = data.summary || {};
      $("trMap50").textContent = `${((s.final_map50 || 0) * 100).toFixed(1)}%`;
      $("trHelmetMap50").textContent = `${((s.helmet_map50 || 0) * 100).toFixed(1)}%`;
      $("trHeadMap50").textContent = `${((s.head_map50 || 0) * 100).toFixed(1)}%`;
      $("trPR").textContent = `${((s.final_precision || 0) * 100).toFixed(1)}% / ${((s.final_recall || 0) * 100).toFixed(1)}%`;
      $("trMap5095").textContent = `${((s.final_map50_95 || 0) * 100).toFixed(1)}%`;

      const cfg = data.config || {};
      const badgeItems = [
        cfg.weights ? `预训练权重: ${cfg.weights}` : null,
        cfg.epochs ? `Epochs: ${cfg.epochs}` : null,
        cfg.imgsz ? `输入尺寸: ${cfg.imgsz}×${cfg.imgsz}` : null,
        cfg.optimizer ? `优化器: ${cfg.optimizer}` : null,
        cfg.freeze ? `冻结主干: 前 ${cfg.freeze} 层` : null,
      ].filter(Boolean);

      $("trainConfigBadges").innerHTML = badgeItems
        .map((txt) => `<span class="badge">${txt}</span>`)
        .join("");

      const charts = data.charts || [];
      $("trainChartsGrid").innerHTML = charts.length
        ? charts
            .map(
              (c) => `
              <div class="chart-card" data-chart-url="${c.url}" data-chart-title="${c.title}">
                <img src="${c.url}" alt="${c.title}" loading="lazy" />
                <div class="chart-info">
                  <h4>${c.title}</h4>
                  <p>${c.description}</p>
                </div>
              </div>
            `
            )
            .join("")
        : '<div class="empty-list">未在 project/runs/ 目录找到训练图表</div>';

      $("trainChartsGrid").querySelectorAll(".chart-card").forEach((card) => {
        card.addEventListener("click", () => {
          openLightbox(card.dataset.chartUrl, card.dataset.chartTitle);
        });
      });

      const epochs = data.epochs || [];
      $("trainEpochCount").textContent = `${epochs.length} Epochs`;
      $("trainEpochsBody").innerHTML = epochs
        .map(
          (row) => `
          <tr>
            <td>${row.epoch}</td>
            <td>${row.train_box_loss.toFixed(4)}</td>
            <td>${row.train_obj_loss.toFixed(4)}</td>
            <td>${row.val_box_loss.toFixed(4)}</td>
            <td>${(row.precision * 100).toFixed(2)}%</td>
            <td>${(row.recall * 100).toFixed(2)}%</td>
            <td>${(row.map50 * 100).toFixed(2)}%</td>
            <td>${(row.map50_95 * 100).toFixed(2)}%</td>
          </tr>
        `
        )
        .join("");
    } catch (_err) {
      $("trainChartsGrid").innerHTML = '<div class="empty-list">加载训练报告失败，请确认后端已启动</div>';
    }
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
