# CLIP Lab

A teaching lab that starts with applications, then makes the image–text comparison inspectable. Public photographs and labelled generated pairs cover open-vocabulary classification, search, similarity, hats, colour, chimneys, medical imagery and style changes.

- [Live app](https://nipunbatra.github.io/clip-lab/)
- [Before / after training](https://nipunbatra.github.io/clip-lab/training.html)
- [Executed notebook](notebooks/clip-lab.ipynb)
- [Sources and credits](https://nipunbatra.github.io/clip-lab/sources.html)

## Run the app

```sh
python3 -m http.server 8770
```

Open http://localhost:8770. No build or API key is required. **Run experiment** loads about 170 MB of pinned model files on first use; inference executes in a Web Worker using WebAssembly. Uploaded images remain in the browser. **View recorded run** displays explicitly labelled measured CPU outputs for unchanged presets.

The app uses OpenAI CLIP ViT-B/32, converted to ONNX by Xenova, model revision `d15189d7028b43f1d3e65039190477f6af591c2a`, q8, with Transformers.js 3.8.1. It loads the image and text encoders once and caches their embeddings. It is a static GitHub Pages site. WebGPU is not required or claimed as tested.

## Execute the notebook and training

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python scripts/execute_notebook.py
python scripts/train_tiny.py
```

The notebook also includes a Colab setup cell. Python runs the original OpenAI CLIP weights in float32 on CPU. The first checkpoint download is approximately 338 MB. Browser/Node q8 scores and original float32 scores can differ, especially for subtle differences; image resizing also differs between runtimes. Do not mix these as one result table.

The separate tiny training model starts from **random initialization**, with no CLIP weights. It learns from 270 coloured-shape image–caption pairs and is evaluated on 90 held-out images. Both encoders and the log scale train. With fixed seed 7 and 500 steps, the recorded accuracy is **10/90 before, 89/90 after**. It tests new renders within nine known concepts, not unseen-concept generalization. Checkpoints and complete run data are saved under `output/` and `data/training.json`.

## Reproduce browser-model measurements

```sh
npm ci
npm run evidence
```

This uses ONNX Runtime on Node CPU and writes `data/evidence.json`. Browser measurements are separately saved in `data/browser-evidence.json`. The original Python notebook is a third, explicitly identified runtime.

## Teaching route

1. Recall the previous ViT’s fixed classifier head.
2. Write new labels; search a gallery with words; choose a sentence; find similar images.
3. Predict and run the hat, colour, chimney, device and style differences.
4. Keep failures: the synthetic device pair does not reliably identify the intended concept.
5. Introduce two encoders, normalization and a score matrix; calculate the symmetric contrastive loss.
6. Show the tiny model before and after actual training on a held-out split.

See `SOURCES.md`, `sources.html`, `images/provenance.json` and `generated-prompts.md`. No CodeEmporium photographs are included. The hat idea is credited to CodeEmporium; search and embedding exploration credit Supabase and Roboflow. Synthetic medical images and aerial scenes are teaching illustrations, not clinical or geospatial evidence.

## Structure

- `app.js`: classroom controls and result display.
- `worker.js`, `engine.js`: browser model inference, embedding caches and scoring.
- `data/experiments.json`, `data/gallery.json`: editable lesson presets and image metadata.
- `scripts/train_tiny.py`: full small-model training experiment.
- `scripts/build_notebook.py`: reproducible notebook source.
- `output/verification/`: browser and execution checks.

Original code is MIT licensed. Public photos retain their own licenses; consult their individual credits. Generated images are clearly labelled and their prompts recorded.

## Verification

All 15 presets have completed real browser WebAssembly inference. The executed notebook contains 11 successful code cells and actual original-model inference plus fresh tiny-model training. Lecture checks separately cover all 87 slide layouts and exact worksheet arithmetic.

`output/verification/ui-check.json` records empty-input, identical-image, stale-recording, phone-overflow and training-toggle checks. To rerun browser inference checks with an optional local Puppeteer installation, use `node scripts/browser-suite.cjs`. If Puppeteer is installed outside this repository, set `CLIP_PUPPETEER_PATH` to that module directory. UI/layout checks use the Playwright CLI with `scripts/ui-check.js`.
