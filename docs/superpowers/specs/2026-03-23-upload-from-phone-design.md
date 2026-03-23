# Upload from Phone — Design Spec

## Overview

Add a "Upload from Phone" button to the TUI Project tab that spins up a temporary copyparty file upload server, displays a QR code + URL in the terminal, and lets the user upload photos from their phone's browser. When done, the server shuts down and the pipeline advances to reconstruction.

## User Flow

1. User fills in **Project Name** and **Output Directory** on the Project tab (Input path left blank)
2. Clicks **"Upload from Phone"**
3. TUI validates name + output dir are filled in
4. TUI creates `<output>/<name>/` and `<output>/<name>/images/`
5. TUI starts copyparty on port 3210, serving `<project>/images/` with read-write access
6. TUI detects the machine's LAN IP (first non-loopback IPv4 address)
7. TUI displays in the log panel:
   - The upload URL: `http://<lan-ip>:3210/`
   - An ASCII QR code encoding that URL
   - Instruction text: "Open this URL on your phone to upload photos"
8. The "Upload from Phone" button is replaced with a **"Done Uploading"** button
9. A background watcher polls `<project>/images/` every 2 seconds, logging: "N images received…" (only when count changes)
10. User clicks **"Done Uploading"**
11. Copyparty process is terminated
12. "Done Uploading" button reverts to "Upload from Phone"
13. Final image count is logged
14. Project config (`project.json`) is saved with `input_type: "images"` and `steps_completed: ["extract"]`
15. TUI auto-advances to the Reconstruct tab

## Implementation

### Components

**`_do_upload_from_phone` worker** (in `tui/app.py`):
- Validates project name + output dir
- Creates project directory structure
- Saves initial project config
- Starts copyparty as a subprocess: `copyparty -v <images_dir>::rw --http-only -p 3210 -q`
- Determines LAN IP via UDP socket probe with fallback to 127.0.0.1
- Generates QR code string via `qrcode` library (`qr.print_ascii()` captured to string)
- Logs URL + QR + instructions
- Swaps button label to "Done Uploading" via `call_from_thread`
- Enters poll loop: every 2 seconds, counts images in `<project>/images/`, logs when count changes
- Poll loop exits when `self._upload_done` flag is set

**`_do_finish_upload` handler** (runs on button press, NOT a worker):
- Sets `self._upload_done = True`
- That's it — the upload worker detects the flag and handles all cleanup

**Upload worker cleanup** (after poll loop exits):
- Terminates copyparty subprocess via `.terminate()` + `.wait(timeout=5)`
- Counts final images (filtering by extension: jpg, jpeg, png, heic)
- Logs summary
- Saves project config with `input_type: "images"`, `steps_completed: ["extract"]`
- Swaps button back to "Upload from Phone" via `call_from_thread`
- Auto-advances to Reconstruct tab

**App exit cleanup**: Store `_copyparty_proc` on self and terminate in `on_unmount()` to prevent orphaned processes.

### Button State Management

The "Upload from Phone" button (`id="btn-upload"`) and "Done Uploading" button (`id="btn-upload-done"`) are both placed in the Project tab's button row. One is hidden at a time using Textual's `display` CSS property:
- Initial state: btn-upload visible, btn-upload-done hidden
- During upload: btn-upload hidden, btn-upload-done visible
- After done: reverts

### Copyparty Subprocess

Started via `subprocess.Popen`:
```python
self._copyparty_proc = subprocess.Popen(
    ["copyparty", "-v", f"{images_dir}::rw", "--http-only", "-p", "3210", "-q"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```

Terminated via `self._copyparty_proc.terminate()` then `.wait(timeout=5)`.

The `-q` flag suppresses copyparty's own logging. `--http-only` avoids HTTPS cert issues on phones.

### QR Code Generation

```python
import io, qrcode
qr = qrcode.QRCode(box_size=1, border=1)
qr.add_data(url)
qr.make()
buf = io.StringIO()
qr.print_ascii(out=buf, invert=True)
qr_text = buf.getvalue()
```

Each line of `qr_text` is logged to the RichLog panel.

### LAN IP Detection

```python
import socket
def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
```

### Dependencies

Add to `requirements.txt`:
```
copyparty>=1.20.0
qrcode>=7.0
```

## Files Changed

| File | Change |
|------|--------|
| `tui/app.py` | New button, upload worker, done handler, QR display, button state management |
| `requirements.txt` | Add copyparty, qrcode |

### QR Code Display

Log QR lines using `Text` objects to avoid Rich markup parsing issues with Unicode block characters:
```python
from rich.text import Text
for line in qr_text.splitlines():
    self.query_one("#log", RichLog).write(Text(line))
```

### HEIC/HEIF Support

Modern iPhones shoot HEIC by default. The image counter during upload should count `*.heic` and `*.HEIC` files. However, COLMAP does not support HEIC natively. If HEIC files are detected after upload, log a warning: "HEIC images detected — set your iPhone to 'Most Compatible' (JPEG) in Settings → Camera → Formats, or convert with: `mogrify -format jpg *.heic`"

## Not Included (YAGNI)

- No authentication on copyparty (local network, temporary server)
- No direct USB/ifuse integration (future enhancement)
- No per-file progress tracking (copyparty's web UI handles this)
- No custom port selection (hardcoded 3210, can be made configurable later)
- No multi-device support (single upload session at a time)

## Error Handling

- If port 3210 is already in use: log error, suggest killing the process
- If no LAN IP found: fall back to 127.0.0.1 with warning
- If copyparty is not installed: log error with install instructions
- If user clicks "Done" with 0 images: warn but allow proceeding
