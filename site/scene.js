/* 展厅与独立结构共用渲染器；按实际画布尺寸取景，手机不被说明抽屉遮挡。 */
const Museum = (() => {
  let renderer,
    scene,
    camera,
    canvas,
    clock,
    selectedId = null,
    onSelect,
    onPart,
    drag = false,
    moved = 0,
    last = { x: 0, y: 0 },
    selectedPart = null;
  const ring = new THREE.Group(),
    crystals = [],
    ray = new THREE.Raycaster(),
    pointer = new THREE.Vector2(),
    labels = document.getElementById("labels"),
    touches = new Map();
  let pinchDistance = 0;
  const cam = { theta: 0.12, phi: 1.38, radius: 20 },
    goal = { ...cam };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  function geometry(kind) {
    switch (kind) {
      case "octa":
        return new THREE.OctahedronGeometry(0.8);
      case "dodeca":
        return new THREE.DodecahedronGeometry(0.7);
      case "torus":
        return new THREE.TorusGeometry(0.6, 0.2, 16, 32);
      case "sphere":
        return new THREE.SphereGeometry(0.75, 20, 16);
      case "box":
        return new THREE.BoxGeometry(1.1, 1.1, 1.1);
      default:
        return new THREE.IcosahedronGeometry(0.8, 1);
    }
  }
  function buildOverview() {
    MODELS.forEach((m, i) => {
      const g = new THREE.Group();
      g.position.set(((i % 4) - 1.5) * 3.5, 3.5 - Math.floor(i / 4) * 3.5, 0);
      const mesh = new THREE.Mesh(
        geometry(m.glyph),
        new THREE.MeshStandardMaterial({
          color: m.color,
          emissive: m.color,
          emissiveIntensity: 0.25,
          metalness: 0.2,
          roughness: 0.4,
        }),
      );
      mesh.rotation.set(0.2, 0.3, 0.1);
      g.add(mesh);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color: m.color,
          transparent: true,
          opacity: 0.4,
        }),
      );
      edges.scale.setScalar(1.2);
      g.add(edges);
      const t = makeTextSprite(m.name, "#e8eefc", 36, 600, 0.009);
      t.position.y = -1.25;
      g.add(t);
      const hit = new THREE.Mesh(
        new THREE.SphereGeometry(1.2, 12, 10),
        new THREE.MeshBasicMaterial({ visible: false }),
      );
      hit.userData.modelId = m.id;
      g.add(hit);
      ring.add(g);
      crystals.push(hit);
    });
    scene.add(ring);
  }
  function fit() {
    if (!renderer) return;
    const w = canvas.clientWidth,
      h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (!camera.userData.fitted) {
      resetCamera();
      camera.userData.fitted = true;
    }
  }
  function resetCamera() {
    const aspect = camera.aspect;
    goal.radius = Math.max(
      selectedId ? 13 : 11,
      (selectedId ? 8.6 : 9) / (Math.tan((48 * Math.PI) / 360) * aspect),
      (selectedId ? 5.2 : 6.5) / Math.tan((48 * Math.PI) / 360),
    );
    goal.theta = selectedId ? 0.12 : 0;
    goal.phi = selectedId ? 1.38 : 1.57;
  }
  function init(opts) {
    canvas = opts.canvas;
    onSelect = opts.onSelect;
    onPart = opts.onPart;
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setClearColor(0x05070f, 0);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(48, 1, 0.1, 300);
    scene.add(new THREE.AmbientLight(0xc3ddff, 0.8));
    const key = new THREE.DirectionalLight(0xe4efff, 1.2);
    key.position.set(3, 5, 10);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x46e0ff, 0.5);
    fill.position.set(-5, -2, 4);
    scene.add(fill);
    buildOverview();
    PipelineAnim.attach(scene);
    fit();
    Object.assign(cam, goal);
    clock = new THREE.Clock();
    new ResizeObserver(fit).observe(canvas);
    window.addEventListener("resize", fit);
    const distance = () => {
      const [a, b] = [...touches.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    canvas.addEventListener("pointerdown", (e) => {
      if (e.pointerType !== "touch" && e.button !== 0) return;
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      canvas.setPointerCapture(e.pointerId);
      if (touches.size === 1) {
        drag = true;
        moved = 0;
        last = { x: e.clientX, y: e.clientY };
      } else if (touches.size === 2) {
        pinchDistance = distance();
        moved = 7;
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      if (touches.has(e.pointerId)) {
        touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (touches.size === 2) {
          const next = distance();
          if (next && pinchDistance) zoom(pinchDistance / next);
          pinchDistance = next;
        } else if (drag) {
          const dx = e.clientX - last.x, dy = e.clientY - last.y;
          moved += Math.abs(dx) + Math.abs(dy);
          goal.theta -= dx * 0.004;
          goal.phi = Math.max(0.65, Math.min(2.3, goal.phi - dy * 0.004));
          last = { x: e.clientX, y: e.clientY };
        }
      } else if (e.pointerType !== "touch") canvas.style.cursor = pick(e) ? "pointer" : "grab";
    });
    const endPointer = (e, cancelled = false) => {
      if (!cancelled && drag && touches.size === 1 && moved < 7) {
        const hit = pick(e);
        if (hit?.userData.partId) onPart(hit.userData.partId);
        else if (hit?.userData.modelId) onSelect(hit.userData.modelId);
      }
      touches.delete(e.pointerId);
      pinchDistance = 0;
      if (touches.size === 1) {
        last = [...touches.values()][0];
        moved = 7;
      } else drag = false;
    };
    canvas.addEventListener("pointerup", (e) => endPointer(e));
    canvas.addEventListener("pointercancel", (e) => endPointer(e, true));
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        zoom(Math.exp(e.deltaY * 0.001));
      },
      { passive: false },
    );
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      document.getElementById("loading").classList.remove("gone");
      document.getElementById("loading").textContent =
        "3D 上下文已中断，请刷新页面重新载入。本页无需联网。";
    });
    animate();
  }
  function zoom(factor) {
    goal.radius = Math.min(80, Math.max(6, goal.radius * factor));
  }
  function pick(e) {
    const r = canvas.getBoundingClientRect();
    pointer.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      (-(e.clientY - r.top) / r.height) * 2 + 1,
    );
    ray.setFromCamera(pointer, camera);
    return ray.intersectObjects(
      selectedId ? PipelineAnim.getPickables() : crystals,
      false,
    )[0]?.object;
  }
  function setSelected(id) {
    selectedId = id;
    selectedPart = null;
    ring.visible = !id;
    labels.replaceChildren();
    if (id)
      STRUCTURES[id].parts.forEach((p, i) => {
        const el = document.createElement("span");
        el.className = "node-label";
        el.dataset.part = p.id;
        el.dataset.number = i + 1;
        labels.append(el);
        el.addEventListener("click", () => onPart(p.id));
      });
    resetCamera();
  }
  function project(p) {
    const v = p.clone().project(camera);
    return {
      x: ((v.x + 1) * canvas.clientWidth) / 2,
      y: ((1 - v.y) * canvas.clientHeight) / 2,
      z: v.z,
    };
  }
  function updateLabels() {
    if (!selectedId) return;
    const state = PipelineAnim.getState(),
      mobile = canvas.clientWidth < 600,
      occupied = [];
    PipelineAnim.getEntries().forEach(({ p, g, i }) => {
      const el = labels.children[i];
      if (!el) return;
      const active = state.phase.ids.includes(p.id),
        selected = selectedPart === p.id;
      const v = g.getWorldPosition(new THREE.Vector3());
      v.y += p.size[1] / 2 + 0.35;
      const xy = project(v);
      el.classList.toggle("current", active);
      el.classList.toggle("selected", selected);
      let gradient = "";
      if (selectedId === "autograd") {
        const values =
          state.index < 4
            ? { x: 0, w: 0, a: 0, b: 0, loss: 0 }
            : state.index === 4
              ? { x: 0, w: 0, a: 0, b: 0, loss: 1 }
              : state.index === 5
                ? { x: 16, w: 0, a: 16, b: 16, loss: 1 }
                : { x: 64, w: 32, a: 16, b: 16, loss: 1 };
        gradient = `∂L/∂${p.id === "loss" ? "L" : p.id} = ${values[p.id]}`;
      }
      const text =
        mobile && !active && !selected ? `${i + 1}` : `${i + 1} · ${p.label}`;
      const content = text + (gradient ? `<small>${gradient}</small>` : "");
      if (el.innerHTML !== content) el.innerHTML = content;
      const w = el.offsetWidth,
        h = el.offsetHeight;
      let x = Math.max(
          w / 2 + 6,
          Math.min(canvas.clientWidth - w / 2 - 6, xy.x),
        ),
        y = Math.max(
          50 + h / 2,
          Math.min(canvas.clientHeight - 28 - h / 2, xy.y),
        );
      // 标签避让仅改变标注位置，不裁切画布或教学结构。
      for (let pass = 0; pass < 8; pass++) {
        const collision = occupied.find(
          (r) =>
            Math.abs(r.x - x) < (r.w + w) / 2 + 3 &&
            Math.abs(r.y - y) < (r.h + h) / 2 + 3,
        );
        if (!collision) break;
        y = collision.y + (collision.h + h) / 2 + 4;
        if (y > canvas.clientHeight - 25 - h / 2) {
          y = collision.y - (collision.h + h) / 2 - 4;
          x = Math.min(canvas.clientWidth - w / 2 - 6, x + 12);
        }
      }
      occupied.push({ x, y, w, h });
      el.style.left = x + "px";
      el.style.top = y + "px";
      el.style.display = xy.z < 1 ? "" : "none";
    });
  }
  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.1),
      k = reduced.matches ? 1 : 1 - Math.exp(-dt * 12);
    for (const key of ["theta", "phi", "radius"])
      cam[key] += (goal[key] - cam[key]) * k;
    camera.position.set(
      cam.radius * Math.sin(cam.phi) * Math.sin(cam.theta),
      cam.radius * Math.cos(cam.phi),
      cam.radius * Math.sin(cam.phi) * Math.cos(cam.theta),
    );
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    PipelineAnim.update(dt);
    renderer.render(scene, camera);
    updateLabels();
  }
  return {
    init,
    setSelected,
    resetCamera,
    zoom,
    selectPart: (id) => {
      selectedPart = id;
      PipelineAnim.select(id);
    },
    projectPart: (id) => {
      const e = PipelineAnim.getEntries().find((e) => e.p.id === id);
      return e ? project(e.g.getWorldPosition(new THREE.Vector3())) : null;
    },
  };
})();
