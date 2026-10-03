# Pretrained CLIP playground

[Open the playground](https://nipunbatra.github.io/clip-lab/playground/).

Students choose an image and descriptions, search the 22-image gallery with words, or compare a before/after pair. Eleven questions cover specificity, wording, negation, retrieval, edits, style, temperature and an incomplete answer menu. Inputs stay editable, and students can upload up to six local images per visit.

Every result is computed on the student's device. There are no recorded results or invented vectors in this app. It is a separate companion to [How CLIP learns](../loss-tutorial/), which uses chosen vectors and shows the training loss.

## Run locally

From the repository root:

```sh
python3 -m http.server 5190 --bind 127.0.0.1
```

Open `http://127.0.0.1:5190/playground/`. No build, API key or extra dependency installation is needed. HTTPS or localhost is required for WebGPU. First use downloads the pinned Transformers.js runtime and model weights; the browser caches the weights for later visits.

## Model and calculation

- Original model: OpenAI CLIP ViT-B/32, using Xenova's ONNX conversion, revision `d15189d7028b43f1d3e65039190477f6af591c2a`.
- Runtime: Transformers.js 3.8.1 in a module worker. WebGPU/fp32 is preferred; WebAssembly/q8 is available explicitly and is the fallback if WebGPU cannot initialize. The displayed runtime always identifies the backend actually used. About 600 MB of model data is needed for fp32, or 170 MB for q8.
- Each encoder includes its learned projection. Its raw output z has 512 coordinates. The app explicitly computes `u = z / ||z||` and the analogous text vector v.
- Matching and retrieval use the full-vector dot product. The inspector shows an eight-coordinate window and the sum of the other 504 products. Its strips show all 512 coordinates, without a dimensionality-reduction plot.
- Change experiments compute `d = normalize(u_after - u_before)` and compare d with each normalized text vector. An identical pair is rejected. The difference is not a generated image or caption.
- Optional temperature shares use `softmax(cosines / tau)` over the displayed candidate list. This is a teaching control, not the checkpoint's learned scale or a calibrated confidence estimate.
- Downloaded JSON includes raw and unit vectors, exact scores, model revision, actual backend/precision, inputs and the student's prediction. Uploaded image bytes are excluded.

The text encoder accepts at most 77 tokens including SOT/EOT. The app rejects longer inputs rather than silently truncating them. Input edits mark previous results as stale. Gallery image credits and generated-image labels are retained; see [sources](../sources.html). Different precisions and runtimes can give slightly different results, so compare runs with their metadata.

## Files

- `index.html`, `style.css`: accessible controls and responsive layout.
- `app.js`: questions, input state, the three inspection stages and export.
- `questions.js`: the eleven editable teaching prompts.
- `model-worker.js`: pinned model loading, live inference and embedding caches.
- `math.js`: normalization, differences, dot products, coordinate contributions and softmax.

The existing Pages workflow copies this directory alongside the shared gallery and engine constants.

## Verification

`node scripts/playground-check.cjs` checks real model inference, all 512-vector norms, the three inspection stages, custom descriptions, stale inputs, local uploads, gallery retrieval, temperature controls, sign reversal after swapping an edit, and 390px mobile layouts. It uses the existing Puppeteer installation in the teaching workspace; set `CLIP_PUPPETEER_PATH` to use another installation. Set `CLIP_TEST_BACKEND=wasm` to test the CPU path.

By default, the test warms the browser cache from this repo's exact pinned `.cache/` model files, then executes actual browser inference. `CLIP_COLD_DOWNLOAD=1` uses normal Hugging Face delivery. `CLIP_TEST_URL` can point at the published page's origin and base path, and `CLIP_BROWSER_PROFILE` chooses a browser profile. Reports distinguish model delivery from execution. Reports and screenshots live in `output/verification/playground/`.

Verified on 3 October 2026 with actual browser WebGPU/fp32 and WebAssembly/q8 inference. Both backends passed the end-to-end checks with no page errors. Desktop and phone screenshots were inspected. The math check also verifies all 64 windows of a 512-coordinate dot product.

The public GitHub Pages URL also passed the complete CPU test in a fresh browser profile, using a normal Hugging Face download. `scripts/playground-export-check.cjs` verifies the actual JSON download, the 77-token limit and rejection of identical images.
