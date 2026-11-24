# Leaf Area Camera

A small, mobile-first web app for estimating the area of a single leaf from a photograph.

## Open the app

The public PWA is deployed at:

<https://navjot-nangia.github.io/leaf_area_camera_fov/>

On Android/Chrome, use **Install app** when prompted. On iPhone/iPad, open the link in Safari, tap **Share**, then select **Add to Home Screen**.

## How it works

1. Place one leaf on a clean, unmarked white sheet.
2. Select the sheet size (A4, US Letter, or custom dimensions).
3. Photograph the entire sheet from above.
4. The app detects the sheet, corrects perspective, separates the leaf from the white background, and reports leaf area in cm².
5. If automatic paper detection is inaccurate, tap the four paper corners manually.

All image processing happens in the browser. Photos are not uploaded. The installed app shell also works offline.

## Run locally

Serve the repository with any static file server, then open `dist/index.html`. For example:

```bash
python3 -m http.server 8000 --directory dist
```

Then visit <http://localhost:8000>.

The camera picker and service worker require HTTPS in production; `localhost` is permitted for development.

## Test

```bash
npm test
```

## Measurement assumptions

- The sheet is flat and its full outline is visible.
- The leaf is flat, fully inside the sheet, and does not overlap the paper edge.
- The paper is plain and contrasts with the leaf.
- Strong shadows, glare, printed marks, and similarly colored surfaces can reduce accuracy.

This is an estimation tool. Validate it against a laboratory leaf-area meter before using measurements in research.

## Deployment

GitHub Actions tests the app and deploys the contents of `dist/` to GitHub Pages after every push to `main`.
