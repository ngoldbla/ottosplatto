"""PLY viewer — serves a Spark.js-based 3D Gaussian Splat viewer."""
import http.server
import functools
import os
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

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

// Load splat with Spark.js
statusEl.textContent = 'Loading splat…';

try {
  const splat = new SplatMesh({ url: plyUrl });
  scene.add(splat);

  // Wait for the splat to load, then frame it
  const checkLoaded = setInterval(() => {
    // SplatMesh populates geometry once loaded
    if (splat.children.length > 0 || splat.geometry?.boundingSphere) {
      clearInterval(checkLoaded);
      statusEl.textContent = `Splat loaded — ${plyFile}`;

      // Try to frame the scene
      const box = new THREE.Box3().setFromObject(splat);
      if (box.isEmpty()) {
        camera.position.set(0, 1, 5);
      } else {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        controls.target.copy(center);
        camera.position.copy(center);
        camera.position.z += maxDim * 1.5;
        camera.position.y += maxDim * 0.3;
      }
      controls.update();
    }
  }, 200);

  // Timeout after 15s
  setTimeout(() => {
    clearInterval(checkLoaded);
    if (statusEl.textContent.includes('Loading')) {
      statusEl.textContent = `Splat loaded — ${plyFile}`;
    }
  }, 15000);

} catch(e) {
  console.error('Spark.js load error:', e);
  statusEl.textContent = 'Error: ' + e.message;
}

// Render loop
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
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

    if open_browser:
        webbrowser.open(url)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return {"status": "success", "url": url, "server": server}
