# Sources and attribution

- [Radford et al., ICML 2021](https://proceedings.mlr.press/v139/radford21a.html): Primary source for the encoders, bidirectional contrastive loss, prompt-based transfer and reported benchmarks. Original training data: 400 million pairs. The lecture’s small experiments are not replications of its benchmarks.

- [OpenAI CLIP implementation](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6): Original weights and Python implementation used in the notebook; MIT. The browser uses the Xenova ONNX conversion of ViT-B/32, revision d15189d7028b43f1d3e65039190477f6af591c2a, with q8 weights.

- [CodeEmporium — CLIP Explained!](https://www.youtube.com/watch?v=FCDKn-vpn_o&t=445s): At 7:25: compare a portrait with and without a hat by subtracting normalized image embeddings and comparing the difference with word embeddings. This is the source of our hat experiment idea. No creator photographs or slide artwork are used. Our generated pairs and chimney, device, colour and style extensions are separately labelled. We show raw cosine; the reference notebook’s optional scale of 10 is a display choice, not proof of a probability.

- [Computerphile — How AI Understands Images](https://www.youtube.com/watch?v=KcSXcpluDe4): Around 1:28–4:00: fixed categories versus descriptions and a shared representation. Around 13:15–15:50: downstream zero-shot matching. We adopt the motivation-first teaching pattern, not the video’s artwork.

- [Yannic Kilcher — OpenAI CLIP paper explanation](https://www.youtube.com/watch?v=T9XSU0pKX2E): Around 4:40–8:00 and 14:40–19:20: image/text pairs, the representation-learning goal and matching within a batch. This informs our gradual move from a retrieval task to a score matrix.

- [Data Science Gems — OpenAI CLIP](https://www.youtube.com/watch?v=rdMnjvjSAkQ): Around 8:38–12:00: zero-shot classification and wording ambiguity. Around 27:38: examples across visual domains. We use this to motivate testing domains and prompts, while checking technical claims against the paper.

- [Supabase — Image Search in Python with OpenAI CLIP](https://www.youtube.com/watch?v=S7VZErcTN5Y&t=414s): Around 6:54–11:10: query a gallery using text; a nearest result can still be irrelevant. Our small browser gallery uses direct dot products instead of a database. Search workflow also documented in the linked Supabase tutorial.

- [Roboflow — CLIP, T-SNE, and UMAP](https://www.youtube.com/watch?v=YxJkE6FvGF4&t=640s): From 10:40: image embeddings for dataset exploration; from 17:22: similarity and possible duplicates. Our app implements nearest-image inspection. It does not claim that similar embeddings prove duplication, or that a 2D projection preserves all distances.

Image licenses, creators and checksums: [provenance.json](images/provenance.json). Generation record: [generated-prompts.md](generated-prompts.md).

## Calculation worksheet and linked tutorial, October 2026

`learn.html` uses the same symmetric contrastive objective from Radford et al., 2021, Figure 3. Its 2D vectors are chosen teaching inputs. Its live mode uses the separately credited OpenAI → Xenova ONNX → Transformers.js pipeline. [WebGPU runtime guide](https://huggingface.co/docs/transformers.js/guides/webgpu).

`loss-tutorial/` is a snapshot of the original course interactive created in **Build interactive CLIP loss tutorial**, source project `~/git/interactives/clip-loss`, 3 October 2026. Its original SVG illustrations and four app files are unchanged. Source SHA-256 values and all derived numerical examples are in `data/tutorial-example.json`. It optimizes raw chosen 3D vectors at fixed temperature, rather than claiming to train a full image/text encoder. Its README includes the original paper and implementation references.

`data/lecture-prompts.json` records the seven exact prompt embeddings for the rebuilt lecture's menu and wording experiments. Runtime, precision and pinned revision are recorded in that file.
