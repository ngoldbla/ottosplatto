"""PLY viewer — serves a lightweight web-based 3D point cloud viewer."""
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
  #hud {
    position:absolute; top:12px; left:12px; color:#58a6ff; z-index:10;
    background:rgba(13,17,23,0.85); padding:12px 16px; border-radius:8px;
    border:1px solid #30363d; font-size:13px; line-height:1.6;
  }
  #hud h1 { margin:0 0 4px; font-size:15px; color:#f0f6fc; }
  #hud .dim { color:#8b949e; }
</style>
</head>
<body>
<div id="hud">
  <h1>OttoSplatto Viewer</h1>
  <span id="status">Loading PLY…</span><br>
  <span class="dim">Drag=rotate · Scroll=zoom · Right-click=pan</span>
</div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.01,500);
camera.position.set(0,2,5);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const ctrl = new OrbitControls(camera,renderer.domElement);

const plyPath = new URLSearchParams(location.search).get('ply') || 'point_cloud.ply';
new PLYLoader().load(plyPath, geo => {
  geo.computeBoundingBox();
  const c = new THREE.Vector3();
  geo.boundingBox.getCenter(c);
  geo.translate(-c.x,-c.y,-c.z);
  const mat = geo.hasAttribute('color')
    ? new THREE.PointsMaterial({size:0.008,vertexColors:true,sizeAttenuation:true})
    : new THREE.PointsMaterial({size:0.008,color:0x58a6ff,sizeAttenuation:true});
  scene.add(new THREE.Points(geo,mat));
  const s = new THREE.Vector3(); geo.boundingBox.getSize(s);
  camera.position.set(0,s.y*0.5,s.z*1.5); ctrl.update();
  document.getElementById('status').textContent =
    `${geo.attributes.position.count.toLocaleString()} points loaded`;
}, xhr => {
  if(xhr.total) document.getElementById('status').textContent =
    `Loading: ${(xhr.loaded/xhr.total*100).toFixed(0)}%`;
}, () => {
  document.getElementById('status').textContent = 'Error loading PLY';
});

addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
});
(function loop(){requestAnimationFrame(loop);ctrl.update();renderer.render(scene,camera)})();
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
