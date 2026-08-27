#!/usr/bin/env python
# MRIcroGL: uRPE vs surprise direct contrast (model7 con13), 3D cut-away render.
# Style matches notes/mricron_settings.png (OverlaySurface + sagittal clip plane,
# smooth overlays). Manuscript colors: green = uRPE>surprise, blue = surprise>uRPE.
# Run:  /Applications/MRIcroGL.app/Contents/MacOS/MRIcroGL notes/figures/mricrogl_con13.py
import gl
D = '/Users/gdehol/git/multlearn-sns/notes/data/surprise_vs_rpe'

gl.resetdefaults()
gl.loadimage('mni152')              # built-in rendered template (same underlay as the screenshot)
gl.overlayloadsmooth(1)            # smooth overlays (menu: Load Smooth Overlays)

# Overlay 1: uRPE > surprise -> GREEN
gl.overlayload(D + '/con13_uRPE_gt_surprise.nii.gz')
gl.minmax(1, 3.1, 7.0)
gl.colorname(1, '2green')
gl.colorfromzero(1, 1)

# Overlay 2: surprise > uRPE -> BLUE
gl.overlayload(D + '/con13_surprise_gt_uRPE.nii.gz')
gl.minmax(2, 3.1, 7.0)
gl.colorname(2, '3blue')
gl.colorfromzero(2, 1)

# --- 3D cut-away render ---
gl.shadername('OverlaySurface')
gl.backcolor(255, 255, 255)        # white background, as in the screenshot
gl.azimuthelevation(90, 0)         # view from the side (sagittal)
gl.clipazimuthelevation(0.5, 0, 180)   # midline clip -> exposes the medial wall
gl.shaderadjust('overlayClip', 1)  # let the clip plane cut through overlays (show interior blobs)
gl.shaderadjust('overlayFuzzy', 0.1)
gl.shaderadjust('overlayDepth', 0.3)
gl.colorbarposition(1)
# --- tweak interactively with the Depth / Azimuth / Elevation sliders (Clipping panel) ---
# gl.savebmp(D + '/../../figures/mricrogl_con13_render.png')
