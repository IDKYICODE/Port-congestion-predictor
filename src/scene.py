"""3D port diorama (Three.js) for one prediction, rendered inside Streamlit via components.html."""
import json

MAX_QUEUE_DRAWN = 30


def scene_data(row, result, total_berths=None, closed=0, thr=0.25, whatif=None):
    r = row.iloc[0]

    def num(col, default=0.0):
        v = r.get(col, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if v != v else v  # NaN -> default

    berths = int(max(1, min(12, round(total_berths or num("term_berths", 1)))))
    closed = int(max(0, min(berths - 1, closed)))
    occupied = int(max(0, min(berths - closed, round(num("term_occupancy")))))
    queue = int(max(0, round(num("term_queue"))))
    lo, mid, hi = result["range_h"]
    return {
        "berths": berths, "closed": closed, "occupied": occupied, "queue": queue, "queueDrawn": min(queue, MAX_QUEUE_DRAWN),
        "rain": num("precip_3d"), "wind": num("wind_max_3d", 10), "hour": int(num("hour", 12)),
        "prob": result["prob"], "thr": thr, "mid": mid, "lo": lo, "hi": hi, "cargo": str(r.get("cargo_nature", "")),
        "whatif": whatif,
    }


def scene_html(data, height=460):
    return TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__H__", str(height))


TEMPLATE = r"""
<style>html, body {margin: 0; overflow: hidden; background: transparent;}</style>
<div id="wrap" style="position:relative;width:100%;height:__H__px;border-radius:14px;overflow:hidden;
     background:#0d2340;font-family:'Source Sans Pro',Segoe UI,Arial,sans-serif">
  <div id="legend" style="position:absolute;left:14px;top:12px;z-index:2;color:#dbe6f6;font-size:12.5px;
       background:rgba(8,20,36,.8);padding:9px 12px;border-radius:10px;line-height:1.75"></div>
  <div id="whatif" style="position:absolute;right:14px;top:12px;z-index:2;display:none;color:#1b1206;font-size:12.5px;
       font-weight:600;background:#f2b04a;padding:8px 12px;border-radius:10px;line-height:1.5;max-width:46%"></div>
  <div style="position:absolute;right:14px;bottom:10px;z-index:2;color:#8ea6c6;font-size:11.5px">drag to rotate</div>
</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const D = __DATA__;
const wrap = document.getElementById("wrap");
const W = () => wrap.clientWidth, H = () => wrap.clientHeight;
const RISK = D.prob >= 0.7 ? 0xf0716b : D.prob >= D.thr ? 0xf2b04a : 0x4cc38a;
const RISK_CSS = "#" + RISK.toString(16).padStart(6, "0");
const fmtH = h => h >= 48 ? (h / 24).toFixed(1) + " days" : Math.round(h) + "h";
const open = D.berths - D.closed, free = Math.max(0, open - D.occupied);
const straightIn = D.queue === 0 && free > 0;

const sw = c => `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${c};margin-right:6px;vertical-align:-1px"></span>`;
document.getElementById("legend").innerHTML =
  `${sw("#e5534b")}Unloading at berth <b>${D.occupied}</b>&nbsp;&nbsp;${sw("#3fb27f")}Free berths <b>${free}</b>` +
  (D.closed ? `&nbsp;&nbsp;${sw("#6b7480")}Closed <b>${D.closed}</b>` : "") + "<br>" +
  `${sw("#7f93aa")}<b>Waiting at anchor ${D.queue}</b>&nbsp;&nbsp;${sw(RISK_CSS)}Your ship` +
  (D.rain > 5 ? `<br><span style="color:#8ea6c6">Rain ${Math.round(D.rain)} mm, wind ${Math.round(D.wind)} km/h (last 3 days)</span>` : "");

if (D.whatif) {
  const w = D.whatif, el = document.getElementById("whatif");
  el.style.display = "block";
  el.innerHTML = `WHAT-IF SCENARIO<br>${w.queue} ship${w.queue === 1 ? "" : "s"} waiting (real: ${w.real_queue})` +
    (w.closed ? ` · ${w.closed} berth${w.closed === 1 ? "" : "s"} closed` : "") +
    (w.moved ? `<br>${w.moved} ship${w.moved === 1 ? "" : "s"} from closed berths joined the line` : "");
}

// ---------- renderer ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(W(), H());
renderer.shadowMap.enabled = true;
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const night = D.hour < 6 || D.hour >= 19;
const SKY = night ? 0x2b4a70 : 0x86afd4;   // dusk rather than black: legibility first
scene.background = new THREE.Color(SKY);
scene.add(new THREE.HemisphereLight(night ? 0x9fb6d8 : 0xeef5ff, 0x1b2a3a, night ? 1.0 : 1.05));
const sun = new THREE.DirectionalLight(night ? 0xc9d6ff : 0xfff2dc, night ? 1.1 : 1.5);
sun.position.set(-60, 90, 50); sun.castShadow = true;
Object.assign(sun.shadow.camera, { left: -140, right: 140, top: 140, bottom: -140 });
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun);

// ---------- layout ----------
const SP = 15;                                                   // berth spacing
const quayLen = Math.max(D.berths, 3) * SP + 16;
const berthX = b => -((D.berths - 1) * SP) / 2 + b * SP;
const LINE_Z = 34, GAP = 17;
// waiting ships at the anchorage, in rows parallel to the quay: #1 at the front right, later ships further left / out
const PER_ROW = Math.max(5, Math.floor((quayLen + 10) / GAP));
const ROW_GAP = 13;
const qPos = k => {
  const row = Math.floor(k / PER_ROW), col = k % PER_ROW;
  return new THREE.Vector3(quayLen / 2 - 4 - col * GAP, 0, LINE_Z + row * ROW_GAP + (col % 2) * 3);
};

// ---------- water ----------
const waterGeo = new THREE.PlaneGeometry(560, 560, 110, 110);
waterGeo.rotateX(-Math.PI / 2);
const water = new THREE.Mesh(waterGeo, new THREE.MeshStandardMaterial({
  color: night ? 0x1f5580 : 0x2b6f9d, roughness: 0.4, metalness: 0.15, flatShading: true }));
water.receiveShadow = true; water.position.z = 140;
scene.add(water);
const wpos = waterGeo.attributes.position, wbase = Float32Array.from(wpos.array);
const waveAmp = 0.12 + Math.min(D.wind, 60) / 60 * 0.75;

// ---------- land, container yard, warehouses ----------
const land = new THREE.Mesh(new THREE.BoxGeometry(quayLen + 140, 4, 100), new THREE.MeshStandardMaterial({ color: night ? 0x4d5661 : 0x66707a }));
land.position.set(0, 1.4, -62); land.receiveShadow = true; scene.add(land);
const quay = new THREE.Mesh(new THREE.BoxGeometry(quayLen, 3.2, 14), new THREE.MeshStandardMaterial({ color: 0x9ea5ae }));
quay.position.set(0, 1.6, -17); quay.receiveShadow = true; scene.add(quay);
const edge = new THREE.Mesh(new THREE.BoxGeometry(quayLen, 0.4, 0.6), new THREE.MeshStandardMaterial({ color: 0xe8c547 }));
edge.position.set(0, 3.3, -10.3); scene.add(edge);
const BOX = [0xc0504d, 0x3b6ea5, 0xd9a441, 0x4f9a6a, 0x8064a2, 0xd07a3a, 0x5aa0b4];
let seed = 7; const rnd = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
const boxGeo = new THREE.BoxGeometry(5.6, 2.4, 2.4);
const boxMats = BOX.map(c => new THREE.MeshStandardMaterial({ color: c }));
for (let bx = -quayLen / 2 - 30; bx < quayLen / 2 + 30; bx += 6.3)
  for (let row = 0; row < 7; row++) {
    if (rnd() < 0.12) continue;
    const stack = 1 + Math.floor(rnd() * 3);
    for (let k = 0; k < stack; k++) {
      const c = new THREE.Mesh(boxGeo, boxMats[Math.floor(rnd() * BOX.length)]);
      c.position.set(bx, 4.6 + k * 2.45, -30 - row * 3.3); c.castShadow = true; scene.add(c);
    }
  }
for (const wx of [-quayLen / 2 - 52, quayLen / 2 + 52]) {
  const wh = new THREE.Mesh(new THREE.BoxGeometry(30, 10, 34), new THREE.MeshStandardMaterial({ color: night ? 0x3a4350 : 0xc3c8ce }));
  wh.position.set(wx, 8.4, -45); wh.castShadow = true; scene.add(wh);
}

// ---------- labels ----------
function label(text, color = "#ffffff", scale = 1) {
  const cv = document.createElement("canvas"), ctx = cv.getContext("2d"), fs = 46;
  ctx.font = `600 ${fs}px Segoe UI, Arial`;
  const lines = text.split("\n"), w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 44;
  cv.width = w; cv.height = lines.length * (fs + 12) + 26;
  ctx.font = `600 ${fs}px Segoe UI, Arial`;
  ctx.fillStyle = "rgba(8,20,36,0.86)"; ctx.beginPath(); ctx.roundRect(0, 0, cv.width, cv.height, 24); ctx.fill();
  ctx.fillStyle = color; ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, 22, 15 + i * (fs + 12)));
  // fixed on-screen size (sizeAttenuation off): readable however far the camera is framed
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false, sizeAttenuation: false }));
  const hh = 0.056 * scale * lines.length;
  sp.scale.set(hh * cv.width / cv.height, hh, 1);
  sp.renderOrder = 10;
  return sp;
}

// ---------- ships ----------
// Hull = top-view outline (square stern, tapered bow) extruded upwards, in two tones:
// red antifouling below the waterline band, the ship's colour above.
function hullGeo(len, beam, h) {
  const s = new THREE.Shape(), b = beam / 2, bow = len * 0.22;
  s.moveTo(-len / 2, -b * 0.92);
  s.quadraticCurveTo(-len / 2 - 0.4, 0, -len / 2, b * 0.92);           // rounded stern
  s.lineTo(len / 2 - bow, b);
  s.quadraticCurveTo(len / 2 - bow * 0.15, b * 0.8, len / 2, 0);         // bow curves to a point
  s.quadraticCurveTo(len / 2 - bow * 0.15, -b * 0.8, len / 2 - bow, -b);
  s.lineTo(-len / 2, -b * 0.92);
  const g = new THREE.ExtrudeGeometry(s, { depth: h, bevelEnabled: true, bevelSize: 0.12, bevelThickness: 0.12, bevelSegments: 2 });
  g.rotateX(-Math.PI / 2);                                              // extrude direction -> up
  return g;
}
const WHITE = new THREE.MeshStandardMaterial({ color: 0xf2f4f6, roughness: 0.6 });
const GLASS = new THREE.MeshStandardMaterial({ color: 0x1b2633, roughness: 0.2, metalness: 0.4 });
const RED_BOTTOM = new THREE.MeshStandardMaterial({ color: 0xa1342d, roughness: 0.8 });
const DECK = new THREE.MeshStandardMaterial({ color: 0x6f7a6a, roughness: 0.9 });
const HULLS = [0x2a3f57, 0x24493a, 0x3d4148, 0x5b2c2c, 0x34506e];
const kindOf = c => /Conteiner/i.test(c) ? "container" : /Gasoso|Líquido|Liquido/i.test(c) ? "tanker" : /Sólido|Solido/i.test(c) ? "bulk" : "container";

function makeShip(hullColor, len = 13, kind = "container") {
  const g = new THREE.Group(), beam = len * 0.17, hm = new THREE.MeshStandardMaterial({ color: hullColor, roughness: 0.55 });
  const low = new THREE.Mesh(hullGeo(len, beam, 0.7), RED_BOTTOM); low.position.y = -0.2; g.add(low);
  const up = new THREE.Mesh(hullGeo(len, beam, 1.5), hm); up.position.y = 0.5; up.castShadow = true; g.add(up);
  const deck = new THREE.Mesh(hullGeo(len * 0.97, beam * 0.9, 0.08), DECK); deck.position.y = 2.05; g.add(deck);

  // bridge superstructure at the stern: stacked white decks, a dark window band, a funnel
  const sx = -len / 2 + len * 0.12;
  const block = new THREE.Mesh(new THREE.BoxGeometry(len * 0.11, 2.6, beam * 0.86), WHITE);
  block.position.set(sx, 3.4, 0); block.castShadow = true; g.add(block);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(len * 0.09, 0.5, beam * 1.08), WHITE);
  wing.position.set(sx + 0.2, 4.85, 0); g.add(wing);
  const win = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.38, beam * 0.8), GLASS);
  win.position.set(sx + len * 0.055 + 0.02, 4.3, 0); g.add(win);
  const funnel = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.5, 1.6, 12), new THREE.MeshStandardMaterial({ color: 0x23303d }));
  funnel.position.set(sx - len * 0.05, 5.1, 0); funnel.castShadow = true; g.add(funnel);

  const x0 = sx + len * 0.09, x1 = len / 2 - len * 0.2;                   // cargo area between bridge and bow
  if (kind === "container") {
    const bays = Math.max(3, Math.floor((x1 - x0) / 1.55));
    for (let i = 0; i < bays; i++) for (let r = -1; r <= 1; r++) {
      const tiers = 1 + ((i * 7 + r * 3 + Math.round(len)) % 3);
      for (let t = 0; t < tiers; t++) {
        const c = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.62, beam * 0.27), boxMats[(i * 5 + r * 3 + t * 2 + Math.round(len)) % BOX.length]);
        c.position.set(x0 + 0.8 + i * 1.55, 2.45 + t * 0.64, r * beam * 0.29); c.castShadow = true; g.add(c);
      }
    }
  } else if (kind === "bulk") {                                          // hatch covers + deck cranes
    const hatches = Math.max(3, Math.floor((x1 - x0) / 2.1));
    for (let i = 0; i < hatches; i++) {
      const hc = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.45, beam * 0.62), new THREE.MeshStandardMaterial({ color: 0x8c4a2f }));
      hc.position.set(x0 + 1 + i * 2.1, 2.3, 0); hc.castShadow = true; g.add(hc);
      if (i % 2 === 0) {
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 2, 6), new THREE.MeshStandardMaterial({ color: 0xe0b232 }));
        post.position.set(x0 + 2.05 + i * 2.1, 3.1, beam * 0.3); g.add(post);
      }
    }
  } else {                                                               // tanker: pipe rack + domes
    const rack = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, 0.25, 0.5), new THREE.MeshStandardMaterial({ color: 0x9aa4ae }));
    rack.position.set((x0 + x1) / 2 + 0.5, 2.45, 0); g.add(rack);
    for (let i = 0; i < 4; i++) {
      const dome = new THREE.Mesh(new THREE.SphereGeometry(beam * 0.18, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), WHITE);
      dome.position.set(x0 + 1.2 + i * (x1 - x0) / 4, 2.1, beam * 0.22 * (i % 2 ? 1 : -1)); g.add(dome);
    }
  }
  return g;
}

const bobbers = [], cranes = [];
for (let b = 0; b < D.berths; b++) {
  const x = berthX(b);
  const isClosed = b >= open, busy = !isClosed && b < D.occupied;
  const col = isClosed ? 0x6b7480 : busy ? 0xe5534b : 0x3fb27f;
  const pad = new THREE.Mesh(new THREE.PlaneGeometry(13.5, 6), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: isClosed ? 0.85 : 0.55 }));
  pad.rotation.x = -Math.PI / 2; pad.position.set(x, 0.2, -6.5); scene.add(pad);
  const tag = label(`Berth ${b + 1}: ${isClosed ? "closed" : busy ? "unloading" : "free"}`, isClosed ? "#c4cad2" : busy ? "#ffb4ae" : "#a8ecc8", 0.65);
  tag.position.set(x, 21, -11); scene.add(tag);
  const crane = new THREE.Group(), cm = new THREE.MeshStandardMaterial({ color: isClosed ? 0x7d858f : 0xe7a93a });
  for (const dx of [-2.6, 2.6]) for (const dz of [-14, -10.5]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.55, 13, 0.55), cm); leg.position.set(dx, 9.5, dz); leg.castShadow = true; crane.add(leg);
  }
  const boom = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.9, 22), cm); boom.position.set(0, 16, -7); boom.castShadow = true; crane.add(boom);
  const trolley = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.8, 1.5), new THREE.MeshStandardMaterial({ color: 0x30363d }));
  trolley.position.set(0, 15.3, -6.5); crane.add(trolley);
  const box = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1.2, 1.3), boxMats[b % BOX.length]);
  box.position.set(0, 10, -6.5); box.visible = busy; crane.add(box);
  crane.position.x = x; scene.add(crane);
  cranes.push({ trolley, box, busy, ph: b * 1.7 });
  if (isClosed) for (const r of [Math.PI / 5, -Math.PI / 5]) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(12, 0.6, 0.7), new THREE.MeshStandardMaterial({ color: 0xd64541 }));
    bar.position.set(x, 0.7, -6.5); bar.rotation.y = r; scene.add(bar);
  }
  if (busy) { const s = makeShip(HULLS[b % HULLS.length], 13, ["container", "bulk", "tanker"][b % 3]); s.position.set(x, 0, -6.5); scene.add(s); bobbers.push({ s, ph: b }); }
}

// ---------- the line of waiting ships, single file towards the quay ----------
for (let k = 0; k < D.queueDrawn; k++) {
  const s = makeShip(HULLS[(k + 2) % HULLS.length], 11.5, ["bulk", "container", "tanker", "container"][k % 4]);
  s.position.copy(qPos(k));
  scene.add(s); bobbers.push({ s, ph: k * 0.8 });
  if (k === 0 || D.queueDrawn <= 6 || k === D.queueDrawn - 1) {
    const t = label(k === 0 ? "#1 next in" : `#${k + 1}`, "#dbe6f6", 0.6);
    t.position.copy(qPos(k)).add(new THREE.Vector3(0, 7.5, 0)); scene.add(t);
  }
}
if (D.queue > D.queueDrawn) {
  const t = label(`+${D.queue - D.queueDrawn} more waiting further out`, "#dbe6f6", 0.45);
  t.position.copy(qPos(D.queueDrawn - 1)).add(new THREE.Vector3(-4, 4, 14)); scene.add(t);
}

// ---------- anchorage zone: where waiting ships sit (clearly apart from ships unloading at the quay) ----------
{
  const n = Math.max(D.queueDrawn + (straightIn ? 0 : 1), 1);
  const ps = [...Array(n).keys()].map(k => qPos(k));
  const x0 = Math.min(...ps.map(p => p.x)) - 12, x1 = Math.max(...ps.map(p => p.x)) + 12;
  const z0 = LINE_Z - 9, z1 = Math.max(...ps.map(p => p.z)) + 9;
  const zone = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, z1 - z0),
    new THREE.MeshBasicMaterial({ color: 0x9fc3ea, transparent: true, opacity: 0.10, depthWrite: false }));
  zone.rotation.x = -Math.PI / 2; zone.position.set((x0 + x1) / 2, 0.35, (z0 + z1) / 2); scene.add(zone);
  const pts = [[x0, z0], [x1, z0], [x1, z1], [x0, z1], [x0, z0]].map(([x, z]) => new THREE.Vector3(x, 0.45, z));
  const border = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineDashedMaterial({ color: 0xcfe3ff, dashSize: 3, gapSize: 2 }));
  border.computeLineDistances(); scene.add(border);
  const zl = label(D.queue ? `ANCHORAGE: ${D.queue} ship${D.queue === 1 ? "" : "s"} waiting for a berth` : "ANCHORAGE: nobody waiting",
                   "#dbe6f6", 0.6);
  zl.position.set(x0 + 6, 7, z1 + 2); scene.add(zl);
}

// ---------- your ship ----------
const you = makeShip(0x14518a, 14, kindOf(D.cargo));
let youTarget;
if (straightIn) {
  let fb = 0;
  for (let b = 0; b < open; b++) if (b >= D.occupied) { fb = b; break; }
  youTarget = new THREE.Vector3(berthX(fb), 0, 8);
} else {
  youTarget = qPos(D.queueDrawn).clone();
}
scene.add(you);
const ring = new THREE.Mesh(new THREE.RingGeometry(8.2, 9.4, 56), new THREE.MeshBasicMaterial({ color: RISK, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
ring.rotation.x = -Math.PI / 2; scene.add(ring);
const where = straightIn ? "a berth is free, no line to wait in" : `joins the line as #${D.queue + 1}`;
const yTag = label(`YOUR SHIP: ${where}\n${Math.round(D.prob * 100)}% delay risk, expect ${fmtH(D.mid)}`, RISK_CSS, 0.62);
scene.add(yTag);

// ---------- rain ----------
let rain = null;
if (D.rain > 5) {
  const n = Math.min(5000, 500 + D.rain * 50), g = new THREE.BufferGeometry(), p = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { p[i * 3] = (Math.random() - .5) * 260; p[i * 3 + 1] = Math.random() * 80; p[i * 3 + 2] = (Math.random() - .5) * 260 + 40; }
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  rain = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xbcd9ff, size: 0.35, transparent: true, opacity: 0.5 }));
  scene.add(rain);
}

// ---------- camera framed to the content: quay at the back, your ship in front ----------
const front = Math.max(youTarget.z, qPos(Math.max(D.queueDrawn - 1, 0)).z, LINE_Z + 6) + 12, back = -32;
const left = Math.min(-quayLen / 2, youTarget.x - 10), right = quayLen / 2 + 6;
const target = new THREE.Vector3((left + right) / 2, 0, (front + back) / 2);
const depth = front - back, width = right - left;
const dist = Math.max(depth * 1.2, width * 0.78) + 18;
const camera = new THREE.PerspectiveCamera(40, W() / H(), 0.5, 1500);
camera.position.set(target.x - dist * 0.38, dist * 0.62, target.z + dist * 0.7);
scene.fog = new THREE.Fog(SKY, dist * 1.3, dist * 3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(target);
controls.enableDamping = true; controls.enableZoom = false; controls.enablePan = false;
controls.maxPolarAngle = Math.PI * 0.46;

// ---------- animate ----------
const clock = new THREE.Clock();
function tick() {
  const t = clock.getElapsedTime();
  for (let i = 0; i < wpos.count; i++) {
    const x = wbase[i * 3], z = wbase[i * 3 + 2];
    wpos.array[i * 3 + 1] = Math.sin(x * 0.07 + t * 1.1) * waveAmp + Math.cos(z * 0.09 + t * 0.8) * waveAmp * 0.6;
  }
  wpos.needsUpdate = true;
  for (const b of bobbers) b.s.position.y = Math.sin(t * 1.2 + b.ph) * waveAmp * 0.45;
  for (const c of cranes) {
    if (!c.busy) continue;
    const k = (Math.sin(t * 0.6 + c.ph) + 1) / 2;
    c.trolley.position.z = -6.5 - k * 7.5; c.box.position.z = c.trolley.position.z;
    c.box.position.y = 10 + Math.abs(Math.sin(t * 1.2 + c.ph)) * 4.5;
  }
  // your ship glides in from open sea once, then holds its place
  const e = 1 - Math.pow(1 - Math.min(1, t / 5), 3);
  you.position.set(youTarget.x, Math.sin(t * 1.2) * waveAmp * 0.45, youTarget.z + 30 * (1 - e));
  ring.position.set(you.position.x, 0.3, you.position.z);
  yTag.position.set(you.position.x, 11, you.position.z);
  if (rain) { const p = rain.geometry.attributes.position; for (let i = 0; i < p.count; i++) { p.array[i * 3 + 1] -= 1.1; if (p.array[i * 3 + 1] < 0) p.array[i * 3 + 1] = 80; } p.needsUpdate = true; }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
new ResizeObserver(() => { camera.aspect = W() / H(); camera.updateProjectionMatrix(); renderer.setSize(W(), H()); }).observe(wrap);
</script>
"""
