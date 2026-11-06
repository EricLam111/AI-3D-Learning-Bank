/* 专属计算结构 + 可暂停的离散阶段时钟。无模型推理、无外部资源。 */
function makeTextSprite(
  text,
  color = "#e8eefc",
  fontSize = 40,
  weight = 600,
  scale = 0.005,
) {
  const c = document.createElement("canvas"),
    ctx = c.getContext("2d");
  ctx.font = `${weight} ${fontSize}px system-ui`;
  c.width = Math.ceil(ctx.measureText(text).width) + 24;
  c.height = fontSize * 1.7;
  ctx.font = `${weight} ${fontSize}px system-ui`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, c.width / 2, c.height / 2);
  const s = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(c),
      depthTest: false,
    }),
  );
  s.scale.set(c.width * scale, c.height * scale, 1);
  return s;
}
const PipelineAnim = (() => {
  let group,
    model,
    spec,
    index = 0,
    elapsed = 0,
    playing = false,
    speed = 1,
    selected = null,
    callback = () => {};
  let entries = [],
    links = [],
    pickables = [],
    age = 0;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const mat = (color) =>
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.45,
      metalness: 0.15,
      emissive: color,
      emissiveIntensity: 0.12,
    });
  function box(parent, w, h, d, x = 0, y = 0, z = 0, color = model.color) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(color));
    mesh.position.set(x, y, z);
    parent.add(mesh);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }),
    );
    mesh.add(edges);
    return mesh;
  }
  function gridTexture(kind, progress = 0) {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const ctx = c.getContext("2d");
    const n = kind === "attention" ? 4 : 16,
      cell = 256 / n;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const noise = (Math.sin(x * 127.1 + y * 311.7) * 43758.5453) % 1;
        let value = Math.abs(noise);
        if (kind === "noise") {
          const r = Math.hypot(x - 7.5, y - 7.5);
          const target = r > 3 && r < 6 ? 0.88 : 0.08;
          value = value * (1 - progress) + target * progress;
          ctx.fillStyle = `rgb(${Math.round(35 + value * 135)},${Math.round(35 + value * 140)},${Math.round(60 + value * 185)})`;
        } else if (kind === "attention") {
          ctx.fillStyle =
            x > y ? "#263244" : `hsl(190 75% ${22 + (1 / (y + 1)) * 40}%)`;
        } else if (kind === "image") {
          ctx.fillStyle =
            x > 3 && x < 9 && y > 4 && y < 12
              ? "#f6c976"
              : `rgb(${25 + x * 3},${65 + y * 4},${95 + x * 4})`;
        } else {
          ctx.fillStyle = `hsl(200 45% ${18 + value * 40}%)`;
        }
        ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
        if (kind === "attention") {
          ctx.fillStyle = x > y ? "#8493aa" : "#e8ffff";
          ctx.font = "bold 23px system-ui";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(
            x > y ? "×" : (1 / (y + 1)).toFixed(2),
            x * cell + cell / 2,
            y * cell + cell / 2,
          );
        }
      }
    const texture = new THREE.CanvasTexture(c);
    texture.minFilter = THREE.LinearFilter;
    return texture;
  }
  function grid(parent, w, h, kind, z = 0) {
    const mesh = box(parent, w, h, 0.12, 0, 0, z);
    mesh.material.dispose();
    mesh.material = new THREE.MeshBasicMaterial({
      map: gridTexture(kind),
      color: "#ffffff",
    });
    mesh.userData.gridKind = kind;
    return mesh;
  }
  function buildPart(p, i) {
    const g = new THREE.Group();
    g.position.fromArray(p.pos);
    group.add(g);
    const [w, h] = p.size;
    let dynamic = null,
      agent = null;
    switch (p.kind) {
      case "attention":
        dynamic = grid(g, w, h, "attention");
        break;
      case "image":
        grid(g, w, h, "image");
        break;
      case "noise":
        dynamic = grid(g, w, h, "noise");
        break;
      case "feature":
        for (let k = 0; k < 3; k++) {
          const plane = grid(g, w, h, "feature", -k * 0.23);
          plane.position.x = k * 0.09;
        }
        break;
      case "heads":
        for (let k = 0; k < 2; k++) {
          grid(g, w, h * 0.65, "feature", k * 0.5).position.set(
            k * 0.25,
            k * 0.35,
            k * 0.5,
          );
        }
        break;
      case "frozen":
        grid(g, w, h, "feature");
        {
          const t = makeTextSprite("冻结", "#d0d9e8", 36, 700, 0.009);
          t.position.z = 0.2;
          g.add(t);
        }
        break;
      case "rankA":
      case "rankB":
        for (let j = 0; j < 2; j++)
          for (let k = 0; k < 8; k++)
            box(
              g,
              p.kind === "rankA" ? (w / 8) * 0.85 : (w / 2) * 0.85,
              p.kind === "rankA" ? (h / 2) * 0.85 : (h / 8) * 0.85,
              0.25,
              p.kind === "rankA" ? ((k - 3.5) * w) / 8 : ((j - 0.5) * w) / 2,
              p.kind === "rankA" ? ((j - 0.5) * h) / 2 : ((k - 3.5) * h) / 8,
            );
        break;
      case "tokens":
        for (let j = 0; j < 4; j++) {
          box(g, w, (h / 4) * 0.8, 0.35, 0, ((1.5 - j) * h) / 4);
          if (model.id === "transformer") {
            const t = makeTextSprite(
              ["我", "喜欢", "学", "AI"][j],
              "#ffffff",
              30,
              600,
              0.007,
            );
            t.position.set(0, ((1.5 - j) * h) / 4, 0.25);
            g.add(t);
          }
        }
        break;
      case "scalar":
      case "op":
      case "loss":
      case "sum": {
        const geometry =
          p.kind === "op"
            ? new THREE.OctahedronGeometry(w * 0.55)
            : p.kind === "loss"
              ? new THREE.IcosahedronGeometry(w * 0.6)
              : new THREE.SphereGeometry(w * 0.5, 24, 16);
        const mesh = new THREE.Mesh(geometry, mat(model.color));
        g.add(mesh);
        const t = makeTextSprite(
          p.kind === "sum"
            ? "+"
            : p.kind === "op"
              ? p.id === "a"
                ? "×"
                : p.id === "b"
                  ? "+"
                  : "Â"
              : p.id === "x"
                ? "2"
                : p.id === "w"
                  ? "3"
                  : p.id === "loss"
                    ? "64"
                    : p.id === "error"
                      ? "L"
                      : model.id === "perceptron" && p.id === "output"
                        ? "1"
                    : p.id === "value"
                      ? ".4"
                      : "t",
          "#ffffff",
          40,
          700,
          0.01,
        );
        t.position.z = w * 0.6;
        g.add(t);
        break;
      }
      case "ffn":
        for (let l = 0; l < 3; l++)
          for (let j = 0; j < (l === 1 ? 8 : 4); j++)
            box(
              g,
              0.2,
              0.15,
              0.4,
              (l - 1) * 0.45,
              (j - ((l === 1 ? 8 : 4) - 1) / 2) * 0.25,
            );
        break;
      case "detHeads":
        for (let j = 0; j < 3; j++) {
          box(g, 1, 0.6, 0.5, 0, (1 - j) * 1.6);
          const t = makeTextSprite(
            ["80²", "40²", "20²"][j],
            "#ffffff",
            30,
            600,
            0.007,
          );
          t.position.set(0, (1 - j) * 1.6, 0.35);
          g.add(t);
        }
        break;
      case "boxes": {
        grid(g, w, h, "image");
        dynamic = [];
        [
          [-0.4, 0.1, 1.1, 1.3, "#9dffb0", "A .92"],
          [-0.3, 0.2, 1.1, 1.3, "#ff5b7a", "B .81"],
          [0.65, -0.4, 0.6, 0.65, "#9dffb0", "C .88"],
        ].forEach(([x, y, bw, bh, c, text]) => {
          const b = new THREE.LineSegments(
            new THREE.EdgesGeometry(new THREE.BoxGeometry(bw, bh, 0.08)),
            new THREE.LineBasicMaterial({ color: c }),
          );
          b.position.set(x, y, 0.2);
          g.add(b);
          const t = makeTextSprite(text, c, 26, 700, 0.006);
          t.position.set(x, y + bh / 2 + 0.12, 0.3);
          g.add(t);
          dynamic.push([b, t]);
        });
        break;
      }
      case "environment":
        grid(g, w, h, "feature");
        box(g, 0.4, 0.4, 0.12, 0.65, -0.15, 0.15, "#ffd479");
        agent = box(g, 0.35, 0.35, 0.4, -0.55, -0.15, 0.25, "#46e0ff");
        break;
      case "policy":
        [0.1, 0.7, 0.1, 0.1].forEach((v, j) => {
          box(g, 0.25, v * 1.8, 0.35, (j - 1.5) * 0.4, v * 0.9 - 0.6);
          const t = makeTextSprite(String(v), "#e8eefc", 24, 600, 0.006);
          t.position.set((j - 1.5) * 0.4, -0.85, 0.25);
          g.add(t);
        });
        break;
      case "reward": {
        const m = new THREE.Mesh(
          new THREE.TorusGeometry(0.4, 0.13, 12, 32),
          mat("#ffd479"),
        );
        g.add(m);
        const t = makeTextSprite("+1", "#ffd479", 32, 700, 0.009);
        t.position.z = 0.2;
        g.add(t);
        break;
      }
      case "clip": {
        box(g, w, h, 0.12, 0, 0, 0, "#243149");
        for (let j = 0; j < 9; j++) {
          const x = (j - 4) * 0.2,
            y = Math.min(x, 0.25);
          box(g, 0.15, 0.06, 0.12, x, y, 0.15, "#ff8a5c");
        }
        box(g, 0.025, h, 0.15, 0.25, 0, 0.15, "#ffd479");
        const t = makeTextSprite("0.8   1.0   1.2", "#e8eefc", 24, 600, 0.006);
        t.position.set(0, -0.6, 0.2);
        g.add(t);
        break;
      }
      default:
        for (let y = 0; y < 4; y++)
          for (let x = 0; x < 4; x++)
            box(
              g,
              (w / 4) * 0.83,
              (h / 4) * 0.83,
              0.3,
              ((x - 1.5) * w) / 4,
              ((y - 1.5) * h) / 4,
              0,
            );
    }
    const hit = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(w, 0.9), Math.max(h, 0.9), 1.1),
      new THREE.MeshBasicMaterial({ visible: false }),
    );
    hit.userData.partId = p.id;
    g.add(hit);
    pickables.push(hit);
    entries.push({ p, g, dynamic, agent, i });
  }
  function buildEdge([from, to, kind]) {
    const a = new THREE.Vector3(...spec.parts.find((p) => p.id === from).pos),
      b = new THREE.Vector3(...spec.parts.find((p) => p.id === to).pos);
    a.z = b.z = -0.25;
    let points = [a, b];
    if (kind === "skip" || kind === "loop") {
      const y = kind === "skip" ? 3.9 : -3.6;
      points = [
        a,
        new THREE.Vector3(a.x, y, -0.6),
        new THREE.Vector3(b.x, y, -0.6),
        b,
      ];
    }
    const curve = new THREE.CurvePath();
    for (let i = 1; i < points.length; i++)
      curve.add(new THREE.LineCurve3(points[i - 1], points[i]));
    const color = kind ? "#46e0ff" : model.color;
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.38 }),
    );
    group.add(line);
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 10, 8),
      new THREE.MeshBasicMaterial({ color: "#ffffff" }),
    );
    group.add(dot);
    const arrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.1, 0.24, 8),
      new THREE.MeshBasicMaterial({ color }),
    );
    group.add(arrow);
    links.push({ from, to, kind, curve, line, dot, arrow });
  }
  function dispose(obj) {
    obj.traverse((c) => {
      c.geometry?.dispose();
      if (c.material) {
        for (const m of Array.isArray(c.material) ? c.material : [c.material]) {
          m.map?.dispose();
          m.dispose();
        }
      }
    });
  }
  function clear() {
    if (!group) return;
    group.children.slice().forEach((c) => {
      group.remove(c);
      dispose(c);
    });
    entries = [];
    links = [];
    pickables = [];
  }
  function emit() {
    callback(getState());
  }
  function apply() {
    if (!spec) return;
    const ph = spec.phases[index];
    entries.forEach(({ p, g, dynamic, agent }) => {
      const active = ph.ids.includes(p.id),
        sel = selected === p.id;
      g.traverse((c) => {
        if (c.isMesh && c.material.emissive)
          c.material.emissiveIntensity = sel ? 0.65 : active ? 0.4 : 0.08;
      });
      if (p.kind === "boxes")
        dynamic.forEach(([b, t], i) => {
          b.visible = t.visible = index >= 4 && (index < 5 || i !== 1);
        });
      if (agent) agent.position.x = index >= 2 ? 0.1 : -0.55;
      if (model.id === "diffusion" && p.kind === "noise") {
        dynamic.material.map.dispose();
        dynamic.material.map = gridTexture(
          "noise",
          Math.min(
            1,
            index / (spec.phases.length - 1) + (p.id === "sample" ? 0.08 : 0),
          ),
        );
      }
    });
  }
  function attach(scene) {
    group = new THREE.Group();
    scene.add(group);
    group.visible = false;
  }
  function start(m) {
    clear();
    model = m;
    spec = STRUCTURES[m.id];
    index = 0;
    elapsed = 0;
    age = 0;
    selected = null;
    spec.parts.forEach(buildPart);
    spec.edges.forEach(buildEdge);
    group.visible = true;
    playing = !reduced.matches;
    apply();
    emit();
  }
  function stop() {
    playing = false;
    clear();
    spec = null;
    model = null;
    group.visible = false;
  }
  function seek(i) {
    if (!spec) return;
    index = Math.max(0, Math.min(spec.phases.length - 1, i));
    elapsed = 0;
    playing = false;
    apply();
    emit();
  }
  function toggle() {
    if (!spec) return;
    if (index === spec.phases.length - 1 && !playing) {
      index = 0;
      elapsed = 0;
      apply();
    }
    playing = !playing;
    emit();
  }
  function update(dt) {
    if (!spec) return;
    age += dt;
    if (playing) {
      elapsed += dt * speed;
      if (elapsed >= 3) {
        elapsed = 0;
        if (index < spec.phases.length - 1) {
          index++;
          apply();
        } else playing = false;
        emit();
      }
    }
    const ph = spec.phases[index];
    const dagEdges = [
      [],
      ["x:a", "w:a"],
      ["a:b", "x:b"],
      ["b:loss"],
      [],
      ["b:loss", "a:b", "x:b"],
      ["x:a", "w:a"],
    ];
    links.forEach((l) => {
      const active =
        model.id === "autograd"
          ? dagEdges[index].includes(`${l.from}:${l.to}`)
          : ph.ids.includes(l.to) || ph.ids.includes(l.from);
      l.line.material.opacity = active ? 0.85 : 0.25;
      const reverse = !!ph.reverse;
      const u = elapsed / 3;
      const eased = u * u * (3 - 2 * u);
      const t = reverse ? 1 - eased : eased;
      l.dot.visible = active;
      l.dot.position.copy(l.curve.getPoint(t));
      const end = reverse ? 0.18 : 0.82,
        dir = l.curve
          .getTangent(end)
          .multiplyScalar(reverse ? -1 : 1)
          .normalize();
      l.arrow.position.copy(l.curve.getPoint(end));
      l.arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      l.arrow.material.color.set(
        reverse ? "#ffd479" : l.kind ? "#46e0ff" : model.color,
      );
    });
    const entrance = reduced.matches
      ? 1
      : 1 - Math.pow(1 - Math.min(1, age / 0.5), 3);
    group.scale.setScalar(0.94 + 0.06 * entrance);
    entries.forEach(({ g, i }) => {
      const u = reduced.matches
        ? 1
        : Math.max(0, Math.min(1, (age - i * 0.025) / 0.36));
      g.scale.setScalar(0.88 + 0.12 * (1 - Math.pow(1 - u, 3)));
    });
  }
  function getState() {
    return {
      model: model?.id,
      index,
      playing,
      speed,
      phase: spec?.phases[index],
      total: spec?.phases.length || 0,
      elapsed,
    };
  }
  function select(id) {
    selected = id;
    apply();
  }
  return {
    attach,
    start,
    stop,
    update,
    seek,
    toggle,
    select,
    setSpeed: (v) => {
      speed = v;
      emit();
    },
    onChange: (fn) => (callback = fn),
    getState,
    getEntries: () => entries,
    getPickables: () => pickables,
    getGroup: () => group,
  };
})();
