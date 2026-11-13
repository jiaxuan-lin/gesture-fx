# Gesture FX

Webcam effects you control with your hands and face, running entirely in the browser.

**Try it:** open the GitHub Pages link for this repo in Chrome (desktop works best), click **Start camera**, and allow camera and microphone access.

- Frame a box with both hands (thumbs and index fingers) to put a filter inside it; pinch both hands to switch filters
- Snap your fingers to change the whole-camera filter (right hand = next, left hand = previous)
- Open your mouth and blow to fog the screen, point a finger to write in the fog, and wipe it away with an open palm
- Swipe an open hand left to right to reset everything; click or press Space to pause

Privacy: hand and face tracking (MediaPipe) and the blow/snap sound detection run on your device. No video or audio is uploaded anywhere.

To run it locally, serve this folder over http (for example `python3 -m http.server 8765`) and open http://localhost:8765.
