/* 语义化控件与三维部件互通；阶段与手动选中互相独立。 */
(() => {
  const $ = (id) => document.getElementById(id);
  let current = null,
    pinned = false,
    ready = false,
    lastPhase = -1;
  function inspect(id, manual = false) {
    if (!current) return;
    const p = STRUCTURES[current.id].parts.find((p) => p.id === id);
    if (!p) return;
    if (manual) {
      pinned = true;
      if (PipelineAnim.getState().playing) PipelineAnim.toggle();
    }
    $("part-title").textContent = p.label;
    $("part-shape").textContent = p.shape;
    $("part-idea").textContent = p.idea;
    $("p-code").textContent = p.code;
    $("selection-note").textContent = pinned
      ? "已锁定部件 · 播放不改变选择"
      : "跟随当前阶段";
    document
      .querySelectorAll("#parts button")
      .forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.part === id)),
      );
    Museum.selectPart(id);
    if (manual) {
      $("part-content").scrollIntoView({
        block: "nearest",
        behavior: "instant",
      });
    }
  }
  function renderState(s) {
    if (!current || !s.phase) return;
    $("phase-count").textContent =
      `阶段 ${String(s.index + 1).padStart(2, "0")} / ${String(s.total).padStart(2, "0")}`;
    $("phase-title").textContent = s.phase.title;
    $("phase-text").textContent = s.phase.text;
    $("run").textContent = s.playing
      ? "Ⅱ 暂停"
      : s.index === s.total - 1
        ? "↺ 重播"
        : "▶ 继续";
    $("run").setAttribute(
      "aria-label",
      s.playing
        ? "暂停演示"
        : s.index === s.total - 1
          ? "重播演示"
          : "继续演示",
    );
    $("step").disabled = s.index === s.total - 1;
    $("run-state").textContent = s.playing
      ? "运行中 · 点击部件暂停并查看"
      : s.index === s.total - 1
        ? "演示完成 · 可重播或回看阶段"
        : "已暂停 · 单步前进 / 空格继续";
    document.querySelectorAll("#timeline button").forEach((b, i) => {
      if (i === s.index) b.setAttribute("aria-current", "step");
      else b.removeAttribute("aria-current");
    });
    document
      .querySelectorAll("#parts button")
      .forEach((b) =>
        b.classList.toggle(
          "phase-active",
          s.phase.ids.includes(b.dataset.part),
        ),
      );
    if (!pinned && lastPhase !== s.index) inspect(s.phase.ids[0]);
    lastPhase = s.index;
  }
  function select(id) {
    if (!ready) return;
    const m = MODELS.find((m) => m.id === id);
    if (!m) return;
    current = m;
    pinned = false;
    lastPhase = -1;
    document.documentElement.style.setProperty("--accent", m.color);
    $("empty").hidden = true;
    $("detail").hidden = false;
    $("intro").hidden = true;
    $("p-tag").textContent = m.tag;
    $("p-title").textContent = m.name;
    $("p-idea").textContent = m.idea;
    $("p-formula").textContent = m.formula;
    $("p-insight").textContent = m.insight;
    $("view-mode").textContent = `结构细节 / ${m.name}`;
    $("hint").textContent =
      "点击 3D 部件或编号查看详情 · 拖动旋转 / 滚轮或双指缩放";
    $("parts").replaceChildren();
    STRUCTURES[id].parts.forEach((p, i) => {
      const b = document.createElement("button");
      b.textContent = `${i + 1} ${p.label}`;
      b.dataset.part = p.id;
      b.setAttribute("aria-pressed", "false");
      b.onclick = () => inspect(p.id, true);
      $("parts").append(b);
    });
    $("timeline").replaceChildren();
    STRUCTURES[id].phases.forEach((p, i) => {
      const b = document.createElement("button");
      b.title = `${i + 1}. ${p.title}`;
      b.setAttribute("aria-label", `跳到阶段 ${i + 1}：${p.title}`);
      b.onclick = () => {
        pinned = false;
        lastPhase = -1;
        PipelineAnim.seek(i);
      };
      $("timeline").append(b);
    });
    document.querySelectorAll(".chip").forEach((b) => {
      b.classList.toggle("active", b.dataset.id === id);
      b.setAttribute("aria-pressed", String(b.dataset.id === id));
    });
    ["run", "step", "reset", "speed"].forEach((id) => ($(id).disabled = false));
    $("panel").scrollTop = 0;
    Museum.setSelected(id);
    PipelineAnim.start(m);
  }
  function overview() {
    if (!ready) return;
    current = null;
    pinned = false;
    PipelineAnim.stop();
    Museum.setSelected(null);
    $("detail").hidden = true;
    $("empty").hidden = false;
    $("intro").hidden = false;
    $("timeline").replaceChildren();
    document.documentElement.style.setProperty("--accent", "#46e0ff");
    document.querySelectorAll(".chip").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-pressed", "false");
    });
    ["run", "step", "reset", "speed"].forEach((id) => ($(id).disabled = true));
    $("view-mode").textContent = `展厅概览 / ${MODELS.length} 件展品`;
    $("phase-count").textContent = "等待选择模型";
    $("phase-title").textContent = "先选一件展品，再逐步拆解";
    $("phase-text").textContent =
      "所有动画均为教学示意，无需下载模型或连接网络。";
    $("run-state").textContent = "未开始 · 可使用键盘操作部件列表";
    $("hint").textContent = "点击晶体或上方模型按钮进入 · 拖动旋转 / 滚轮或双指缩放";
    $("run").textContent = "▶ 继续";
    $("panel").scrollTop = 0;
  }
  MODELS.forEach((m, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.dataset.id = m.id;
    b.style.setProperty("--c", m.color);
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<b>${String(i + 1).padStart(2, "0")}</b><span class="model-name">${m.name}</span><span class="model-short">${m.short}</span>`;
    b.setAttribute("aria-label", m.name);
    b.title = m.name;
    b.onclick = () => select(m.id);
    $("chips").append(b);
  });
  $("run").onclick = () => PipelineAnim.toggle();
  $("step").onclick = () => {
    pinned = false;
    lastPhase = -1;
    PipelineAnim.seek(PipelineAnim.getState().index + 1);
  };
  $("reset").onclick = () => {
    pinned = false;
    lastPhase = -1;
    PipelineAnim.seek(0);
  };
  $("speed").onchange = (e) => PipelineAnim.setSpeed(Number(e.target.value));
  $("close").onclick = () => {
    overview();
    document.querySelector(".chip").focus();
  };
  $("home").onclick = (e) => {
    e.preventDefault();
    overview();
  };
  $("camera").onclick = () => Museum.resetCamera();
  $("zoom-in").onclick = () => Museum.zoom(0.8);
  $("zoom-out").onclick = () => Museum.zoom(1.25);
  $("explore").onclick = () => select("symbolic");
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      overview();
      document.querySelector(".chip").focus();
      return;
    }
    if (e.target.closest("button,select,summary,a,input")) return;
    if (current && e.code === "Space") {
      e.preventDefault();
      PipelineAnim.toggle();
    }
    if (current && e.key === "ArrowRight") {
      e.preventDefault();
      pinned = false;
      lastPhase = -1;
      PipelineAnim.seek(PipelineAnim.getState().index + 1);
    }
    if (current && e.key === "ArrowLeft") {
      e.preventDefault();
      pinned = false;
      lastPhase = -1;
      PipelineAnim.seek(PipelineAnim.getState().index - 1);
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && PipelineAnim.getState().playing)
      PipelineAnim.toggle();
  });
  matchMedia("(prefers-reduced-motion: reduce)").addEventListener(
    "change",
    (e) => {
      if (e.matches && PipelineAnim.getState().playing) PipelineAnim.toggle();
    },
  );
  PipelineAnim.onChange(renderState);
  try {
    Museum.init({
      canvas: $("stage"),
      onSelect: select,
      onPart: (id) => inspect(id, true),
    });
    ready = true;
    $("loading").classList.add("gone");
  } catch (err) {
    $("loading").textContent =
      "无法创建 WebGL 场景。请使用支持 WebGL 的浏览器并刷新；本教具无需网络或模型下载。";
    console.error(err);
  }
})();
