# How CLIP Learns: The Contrastive Loss, Step by Step

An offline-ready, client-side teaching interactive. Follow a toy batch from paired examples through normalization, cosine similarities, temperature, both softmax directions, symmetric cross-entropy and actual gradient updates.

## Run

```sh
cd ~/git/interactives/clip-loss
python3 -m http.server 8000
```

Open **http://localhost:8000**. No install, build, backend, external fonts or CDN is needed. Opening `index.html` directly also works. To deploy to GitHub Pages, publish these four files together: `index.html`, `styles.css`, `app.js`, `README.md`. No deployment has been performed.

## Teach with it

- Use the 11-stage navigation or ← / →. Links such as `#image-text`, `#loss` and `#align` open a specific stage.
- Hover or select a matrix cell to follow its image and caption, dot product, logit, exponential and probability in the provenance strip. Expand calculations for both denominators and the raw vectors.
- The row/column switch uses the same matrix positions. Select individual examples or loss terms to inspect every classification problem.
- Focus mode dims competing rows or columns while retaining the candidates in the selected classification problem.
- Drag the SVG sphere to rotate its 3D projection. A focused sphere also supports arrow keys. Images are solid lines/circles; texts are dashed lines/squares.
- Temperature defaults to 0.5. The 0.07 preset illustrates CLIP's initial temperature. Training holds the selected temperature fixed.
- “One gradient step,” “Train 10 steps,” and Train/Pause update the raw vectors using real analytical gradients. R resets training; Space starts/stops training on the last two stages, or auto-plays the story elsewhere.
- The Controls drawer provides temperature, learning rate, precision, equation visibility, expanded calculations, auto-play and a full reset.
- “Exact*” means eight displayed decimal places, not symbolic arithmetic. Every computation uses JavaScript double precision regardless of display settings.
- The guided batch supports two or three pairs from cat/dog/car. Shuffling reorders pairs together and resets the training experiment. The other nine dataset examples illustrate a larger dataset; they are not assigned invented encoder outputs.
- Before/after losses and the history plot are recomputed at the current temperature, so comparisons always use the same objective.

## Numerical model

The six specified raw vectors live centrally in `DATASET` in `app.js`. Every downstream quantity comes from the pure `clipLoss()` forward pass.

1. Normalize each raw vector to unit L2 norm.
2. Compute `S = E_I E_Tᵀ` and logits `L = S / tau`.
3. Apply stable softmax across rows and down columns.
4. Compute cross-entropy using log-sum-exp; average per-example losses, then average directions.
5. Compute `G = (P + Q − 2 Identity) / (2N)`.
6. Propagate through the matrix product, scale and normalization. For `u = x / ||x||`, use `(g − u (u·g)) / ||x||`.
7. Update both sets of raw vectors by gradient descent.

The display shows unshifted exponentials because they are readable at the bounded temperature range. The probability implementation subtracts the maximum logit; the common factor cancels and yields the same values. These are toy encoder outputs, not outputs measured from a pretrained CLIP. Real CLIP optimizes both neural-network encoders and a learned logit scale, not independent vectors for every training example.

Default initial loss: **0.5605264020425855**. At learning rate 0.25 and temperature 0.5, 100 updates give **0.12493712997191139**. No loss or similarity trajectory is scripted.

## Verify

The browser runs `runMathSanityChecks()` at startup and prints the result in its console. The same dependency-free checks run with Node:

```sh
node -e "require('./app.js').runMathSanityChecks()"
```

225 assertions cover both batch sizes and temperatures 0.05, 0.07, 0.5 and 1: unit norms, matrix shape, row/column sums, symmetric loss, a decreasing small gradient step, stable extreme-logit softmax, and finite-difference agreement for every coordinate in both modalities.

Browser verification passed 48 checks covering all 11 stages at 1440 × 810 and 390 × 844, matrix cell continuity, hover tracing, both softmax directions, focus mode, precision, drawers, training, batch changes and reset. No browser runtime errors occurred. Representative screens were visually inspected. The report and screenshots are in `output/playwright/`. A separate numerical sweep confirmed 1,200 finite, non-increasing updates across 12 temperature/learning-rate settings.

For browser diagnostics, `window.clipLab` exposes `getState()`, `getMetrics()`, `go(index)`, `step()`, `pause()`, `reset()`, `setTemperature(value)` and the pure numerical functions under `.math`.

## Implementation

- `index.html`: persistent lesson shell, navigation, controls and reference drawer.
- `styles.css`: responsive editorial layout, semantic pair colors, reduced-motion support.
- `app.js`: dataset, math, state, reusable SVG matrix, original local SVG illustrations, rotatable 3D projection, provenance, training and navigation.
- Matrix cell DOM nodes persist through similarity → logits → probabilities, and numerical values interpolate between states. Selection identifies the same image/text combination everywhere.
- System fonts and native equation notation keep the full application usable offline. A 1440 × 900 or larger desktop display is recommended; smaller screens scroll vertically.

## Sources and context

- [Radford et al. (2021), Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), especially Figure 3 and the symmetric cross-entropy objective.
- [OpenAI CLIP implementation](https://github.com/openai/CLIP/blob/main/clip/model.py): normalization, cosine logits, learned logit scale initialized to `log(1 / 0.07)`.
- This standalone loss lab complements the existing CLIP lecture in `~/git/dl-teaching/attention-followups/clip/`. It does not modify the lecture or its measured demonstrations.

Illustrations are original SVGs included in the application. No assets are downloaded.
