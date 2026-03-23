# Capture Guide Analysis — Actionable Items for OttoSplatto

Source: "iPhone capture for 3D Gaussian splatting: the complete guide"

## Critical Findings That Should Change Our Software

### 1. COLMAP camera model should be OPENCV, not SIMPLE_RADIAL
The guide explicitly recommends `--ImageReader.camera_model OPENCV` with `--ImageReader.single_camera 1` for iPhone images. Our EXIF auto-detection currently recommends SIMPLE_RADIAL for iPhones. The guide's reasoning: OPENCV handles the full distortion model better.

**Action:** Update exif_detect.py — iPhone → OPENCV with single_camera flag.

### 2. single_camera flag is critical for phone captures
When all images come from the same device, `--ImageReader.single_camera 1` tells COLMAP to share intrinsics across all frames. This dramatically improves reconstruction speed and accuracy.

**Action:** Add single_camera auto-detection to reconstruct.py when EXIF shows same camera.

### 3. COLMAP feature extraction should use higher limits
Guide recommends `--SiftExtraction.max_image_size 3200` and `--SiftExtraction.max_num_features 8192` instead of defaults. More features = better matching.

**Action:** Add these as optimized defaults in reconstruct.py.

### 4. Sequential matcher for video frames, exhaustive for photos
We already do this, but the guide confirms it's important. Sequential is faster and sufficient for ordered frames.

**Action:** Already implemented. Validate our auto-detection logic.

### 5. Frame extraction should cull blurry frames BEFORE COLMAP
The guide recommends automatic blur detection and culling after FFmpeg extraction and before COLMAP. Our precheck.py warns but doesn't cull.

**Action:** Add auto-cull option to precheck.py — move blurry frames to a quarantine folder.

### 6. Exposure compensation during training
The original 3DGS repo has exposure compensation flags for auto-exposed smartphone footage. This dramatically improves results from casually captured footage.

**Action:** Detect auto-exposure variation in precheck, enable compensation flag if detected.

### 7. LiDAR bypass of COLMAP (Record3D integration)
For iPhone Pro users, Record3D exports RGBD + camera poses, completely bypassing COLMAP. This is faster and works on textureless scenes where COLMAP fails.

**Action:** Add Record3D import as an alternative to COLMAP reconstruction. Major feature.

### 8. ProRes codec recommendation
ProRes (intra-frame only, 10-bit) produces much cleaner individual frames than HEVC. If storage permits, this is the best video codec.

**Action:** Add ProRes recommendation to capture guidance in the TUI.

## Capture Guidance That Should Be In The TUI

### Pre-capture checklist (show to user before/during upload)
- Lock exposure (AE/AF Lock or manual app)
- Use main 1× lens (not ultra-wide or telephoto)
- Disable HDR, Live Photos, Portrait mode
- Shoot at 4K 30fps for video, max resolution 4:3 for photos
- Move slowly, maintain 70-80% overlap
- Multi-height spiral: knee, eye, overhead
- Diffuse/overcast lighting preferred
- Clear scene of moving objects
- Minimum 50 photos (small object), 100-200 (room), 200-500 (large scene)

### Image count recommendations
| Subject | Minimum | Recommended | Maximum useful |
|---------|---------|-------------|---------------|
| Small object (<1 ft) | 50 | 80-100 | 150 |
| Medium object (1-10 ft) | 100 | 150-200 | 300 |
| Room / building facade | 200 | 300-500 | 1000 |

### Six mistakes to warn about
1. Motion blur (moving too fast)
2. Unlocked auto-exposure
3. Insufficient angular coverage (single height)
4. Using ultra-wide lens
5. Portrait mode / shallow DOF
6. Scene changes during capture

## Technical Improvements to Implement

### Priority A — Change existing behavior
1. **COLMAP camera model**: iPhone → OPENCV (not SIMPLE_RADIAL)
2. **single_camera flag**: Auto-detect same-camera captures, enable shared intrinsics
3. **SIFT features**: Increase max_image_size to 3200, max_num_features to 8192
4. **Blur culling**: Auto-remove blurry frames before COLMAP (not just warn)

### Priority B — New features
5. **Capture guidance panel** in TUI: show checklist + recommendations before upload
6. **Exposure variation detector**: analyze histogram consistency across images
7. **FFmpeg extraction improvements**: quality-preserving settings (-qscale:v 1 -qmin 1)

### Priority C — Future integrations
8. **Record3D import**: bypass COLMAP entirely for LiDAR iPhones
9. **Exposure compensation**: enable 3DGS training flag when auto-exposure detected
10. **SPZ export**: Scaniverse-compatible compressed format (90% smaller)
