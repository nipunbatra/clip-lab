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

The app uses OpenAI CLIP ViT-B/32, converted to ONNX by Xenova, model revision `d15189d7028b43f1d3e65039190477f6af591c2a`, q8, with Transformers.js 3.8.1. It loads the image and text encoders once and caches their embeddings. It is a static GitHub Pages site. The original gallery uses WebAssembly. The new calculation worksheet below also supports and has been tested with WebGPU.

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

## Using an example

Open **Usage tips & a guided first visit** for three specific steps and eight hand-picked examples. Predict before running. Every change example spells out the intended edit, what should remain fixed, and confounding differences in the generated pair. **Swap before / after** reverses the image order; a new live run negates each score when the embeddings and candidate words stay fixed. Pair descriptions explicitly refer to the original preset when you edit the inputs.

Under **What happens under the hood?**, each task explains its actual encoder inputs, comparisons and outputs. Difference examples give the full normalization/subtraction formula and explain signed scores. CLIP does not make an image edit or train during a change experiment. **What should we learn?** retains the observed successes and failures.

Usage copy and the curated route live in `data/guidance.json`. The optional lecture companion preserves all 15 examples as exact-input/recorded-output pairs. The rebuilt main lecture starts with three demonstrations, then explains the mechanism.

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

All 15 presets have completed real browser WebAssembly inference. The executed notebook contains 11 successful code cells and actual original-model inference plus fresh tiny-model training. The October rebuild separately checks all 144 main lecture layouts and the exact worksheet arithmetic.

`output/verification/ui-check.json` records empty-input, identical-image, stale-recording, phone-overflow and training-toggle checks. To rerun browser inference checks with an optional local Puppeteer installation, use `node scripts/browser-suite.cjs`. If Puppeteer is installed outside this repository, set `CLIP_PUPPETEER_PATH` to that module directory. UI/layout checks use the Playwright CLI with `scripts/ui-check.js`.

`output/verification/guidance-check.json` covers all 15 guides, all five pair descriptions and swap controls, recorded-run guards and phone layouts. `output/verification/swap-live-check.json` records an actual live forward/reverse color pair and exact sign reversal.

## One calculation at a time (October main-lecture rebuild)

[Open the loss worksheet](https://nipunbatra.github.io/clip-lab/learn.html). It starts with the same chosen 2D vectors as the rebuilt lecture. Select a matrix cell, switch the matching direction, and advance from dot product through the symmetric loss. The score-scale slider recomputes the shares. WebGPU computes the toy dot products when available; JavaScript CPU is the fallback. Softmax and loss arithmetic run on the CPU and the runtime label says so.

**Try pretrained CLIP** loads the pinned original-model conversion through Transformers.js 3.8.1: WebGPU/fp32 where supported, or WebAssembly/q8. It encodes the actual three Oxford-IIIT Pet photos and three supplied breed captions. It also accepts a new description for the first photo. Backend and precision are displayed; these live scores can differ from the lecture's saved CPU q8 scores. Downloads are cached and may be slow on first use.

The [simple notebook](notebooks/clip-simple.ipynb) has seven core PyTorch cells and one optional cross-check of the other tutorial's 3D example. Run `pip install torch jupyterlab`, then `jupyter lab notebooks/clip-simple.ipynb` and Run All. The core uses chosen features and one-hot pair IDs, without any model/data download. It reproduces loss 0.5200275889 and trains two tiny branches from loss 2.328686 to 0.003873. Its optional eighth cell independently reproduces the exact 3D tutorial calculation and updates in float64.

### Companion interactive training tutorial

[How CLIP Learns](https://nipunbatra.github.io/clip-lab/loss-tutorial/) is a deployment snapshot of the user's **Build interactive CLIP loss tutorial** project at `~/git/interactives/clip-loss`. The four source files are preserved unchanged; see [snapshot provenance](loss-tutorial/SNAPSHOT.md). It uses chosen cat/dog/car 3D outputs, persistent matrix cells, source-number tracing, and real gradient updates of the six raw vectors. Initial loss 0.5605264020 becomes 0.1249371300 after 100 updates at τ=0.5 and learning rate 0.25. This direct vector optimization is distinct from full CLIP encoder training.

The rebuilt lecture retains its requested 2D hand calculation, then uses these exact 3D vectors and results as a linked follow-up. The tutorial calls cosine `S` and logits `L`; the lecture calls them `C` and `S`. [Exact cross-reference data](data/tutorial-example.json) records the notation, original inputs, every derived value, and source-file hashes.

### Verification for the new worksheet

`node scripts/learn-browser-check.cjs` with a server on port 5190 checks the 2D loss, controls, phone width, actual pretrained inference and a new prompt. Set `CLIP_PUPPETEER_PATH` if Puppeteer is outside this repository. For a repeatable warm-cache run, set `CLIP_LOCAL_MODEL_CACHE=1` with the original pinned weights under `.cache/`. `CLIP_TEST_BACKEND=wasm` explicitly checks the smaller model. Cache warming changes delivery only; actual inference still runs in the browser. Reports distinguish cold network download from pre-cached tests.
