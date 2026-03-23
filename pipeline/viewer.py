"""PLY viewer — serves a Spark.js-based 3D Gaussian Splat viewer."""
import http.server
import functools
import os
import shutil
import subprocess
import threading
import webbrowser
from typing import Optional, Callable

VIEWER_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OttoSplatto Viewer</title>
<style>
  body { margin:0; background:#0d1117; overflow:hidden; font-family:monospace; }
  canvas { display:block; }
  #hud {
    position:absolute; top:12px; left:12px; color:#58a6ff; z-index:10;
    background:rgba(13,17,23,0.85); padding:12px 16px; border-radius:8px;
    border:1px solid #30363d; font-size:13px; line-height:1.6;
    pointer-events:none;
  }
  #hud h1 { margin:0 0 4px; font-size:15px; color:#f0f6fc; }
  #hud .dim { color:#8b949e; }
</style>
</head>
<body>
<div id="hud">
  <h1>OttoSplatto Viewer</h1>
  <span id="status">Loading splat…</span><br>
  <span class="dim">Drag=rotate · Scroll=zoom · Right-click=pan</span>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/",
    "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SplatMesh } from '@sparkjsdev/spark';

const statusEl = document.getElementById('status');
const plyFile = new URLSearchParams(location.search).get('ply') || 'point_cloud.ply';
const plyUrl = new URL(plyFile, location.href).href;

// Scene
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.05, 500);
camera.position.set(0, 2, 6);

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

// Expose for debugging
window.__scene = scene;
window.__camera = camera;
window.__controls = controls;

function frameCamera(box) {
  if (!box || box.isEmpty()) {
    camera.position.set(0, 1, 5);
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = size.length() / 2;
  // Position camera to see the whole scene: use FOV to compute distance
  const fov = camera.fov * (Math.PI / 180);
  const dist = radius / Math.sin(fov / 2);
  controls.target.copy(center);
  camera.position.set(center.x, center.y + radius * 0.3, center.z + dist * 0.8);
  camera.near = dist * 0.01;
  camera.far = dist * 5;
  camera.updateProjectionMatrix();
  controls.update();
}

// SH DC coefficient to linear RGB
function shToColor(f0, f1, f2) {
  const C0 = 0.2820947917738781;
  return [
    Math.max(0, Math.min(1, 0.5 + C0 * f0)),
    Math.max(0, Math.min(1, 0.5 + C0 * f1)),
    Math.max(0, Math.min(1, 0.5 + C0 * f2)),
  ];
}

// Point cloud fallback — parse 3DGS PLY into Three.js Points
async function loadPointCloud(url) {
  const resp = await fetch(url);
  const buf = await resp.arrayBuffer();
  const bytes = new Uint8Array(buf);

  // Parse header
  let headerEnd = 0;
  const decoder = new TextDecoder();
  for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
    if (bytes[i] === 0x0A) { // newline
      const line = decoder.decode(bytes.slice(headerEnd, i)).trim();
      if (line === 'end_header') { headerEnd = i + 1; break; }
      headerEnd = i + 1;
    }
  }

  // Re-parse header for property layout
  const headerStr = decoder.decode(bytes.slice(0, headerEnd));
  const lines = headerStr.split('\n').map(l => l.trim());
  let vertexCount = 0;
  const props = [];
  for (const line of lines) {
    if (line.startsWith('element vertex')) vertexCount = parseInt(line.split(' ')[2]);
    if (line.startsWith('property float')) props.push(line.split(' ')[2]);
  }

  const stride = props.length * 4; // all floats
  const propIndex = (name) => props.indexOf(name);

  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const data = new DataView(buf, headerEnd);

  const ix = propIndex('x'), iy = propIndex('y'), iz = propIndex('z');
  const idc0 = propIndex('f_dc_0'), idc1 = propIndex('f_dc_1'), idc2 = propIndex('f_dc_2');
  for (let i = 0; i < vertexCount; i++) {
    const off = i * stride;
    positions[i*3]   = data.getFloat32(off + ix*4, true);
    positions[i*3+1] = data.getFloat32(off + iy*4, true);
    positions[i*3+2] = data.getFloat32(off + iz*4, true);

    if (idc0 >= 0) {
      const [r, g, b] = shToColor(
        data.getFloat32(off + idc0*4, true),
        data.getFloat32(off + idc1*4, true),
        data.getFloat32(off + idc2*4, true),
      );
      colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
    } else {
      colors[i*3] = 1; colors[i*3+1] = 1; colors[i*3+2] = 1;
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.computeBoundingBox();

  // Scale point size relative to scene extent
  const bbox = geom.boundingBox;
  const extent = new THREE.Vector3();
  bbox.getSize(extent);
  const maxDim = Math.max(extent.x, extent.y, extent.z);
  const pointSize = Math.max(0.01, maxDim / 500);

  const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true });
  return new THREE.Points(geom, mat);
}

// Start render loop immediately — Spark.js needs active rendering during init
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Load splat with Spark.js
statusEl.textContent = 'Loading splat…';

let splatLoaded = false;
try {
  const splat = new SplatMesh({ url: plyUrl });
  window.__splat = splat;
  scene.add(splat);

  await splat.initialized;
  splatLoaded = true;
  statusEl.textContent = `Splat loaded — ${plyFile}`;
  frameCamera(splat.getBoundingBox());

  // Verify Spark.js actually rendered something by reading pixels
  // immediately after a render call (before the buffer is cleared)
  let hasPixels = false;
  const gl = renderer.getContext();
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;

  for (let attempt = 0; attempt < 10 && !hasPixels; attempt++) {
    await new Promise(r => setTimeout(r, 500));
    renderer.render(scene, camera);
    // Read a row of pixels across the middle of the screen right after render
    const row = new Uint8Array(w * 4);
    gl.readPixels(0, Math.floor(h / 2), w, 1, gl.RGBA, gl.UNSIGNED_BYTE, row);
    for (let i = 0; i < row.length; i += 4) {
      if (row[i] > 2 || row[i+1] > 2 || row[i+2] > 2) { hasPixels = true; break; }
    }
  }

  if (!hasPixels) {
    console.warn('Spark.js rendered black after 5s — falling back to point cloud');
    scene.remove(splat);
    splat.dispose();
    splatLoaded = false;
  }
} catch(e) {
  console.warn('Spark.js failed:', e.message, '— using point cloud fallback');
  splatLoaded = false;
}

if (!splatLoaded) {
  statusEl.textContent = 'Loading point cloud…';
  try {
    const points = await loadPointCloud(plyUrl);
    scene.add(points);
    frameCamera(points.geometry.boundingBox);
    statusEl.textContent = `Point cloud loaded — ${plyFile} (${(points.geometry.attributes.position.count/1000).toFixed(0)}K points)`;
  } catch(e) {
    console.error('Point cloud load error:', e);
    statusEl.textContent = 'Error: ' + e.message;
  }
}

</script>
</body>
</html>"""


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves files from project directory + injects viewer HTML."""

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(VIEWER_HTML.encode())
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        pass  # silence default logging


def _is_wayland() -> bool:
    """Check if the current session is running on Wayland."""
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def _open_browser(url: str, on_output: Optional[Callable[[str], None]] = None) -> None:
    """Open URL in a browser, working around Wayland+NVIDIA WebGL issues."""
    if _is_wayland():
        # Wayland + NVIDIA breaks Chrome's GPU process — WebGL is unavailable.
        # Launch Chromium/Chrome with --ozone-platform=x11 to force X11 backend.
        for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
            path = shutil.which(name)
            if path:
                if on_output:
                    on_output(f"Wayland detected — launching {name} with X11 backend for WebGL")
                subprocess.Popen(
                    [path, "--ozone-platform=x11", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
    webbrowser.open(url)


def launch_viewer(
    ply_path: str,
    port: int = 8765,
    open_browser: bool = True,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Start a local HTTP server and open the PLY viewer in a browser."""
    if not os.path.isfile(ply_path):
        return {"status": "error", "message": f"PLY not found: {ply_path}"}

    serve_dir = os.path.dirname(ply_path)
    ply_name = os.path.basename(ply_path)

    handler = functools.partial(_Handler, directory=serve_dir)
    server = http.server.HTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}/?ply={ply_name}"
    if on_output:
        on_output(f"Viewer running at {url}")
        on_output(f"Serving {ply_path}")
        on_output("Press Ctrl+C to stop")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        try:
            _open_browser(url, on_output)
        except Exception:
            pass  # non-fatal — user can open URL manually

    return {"status": "success", "url": url, "server": server}
