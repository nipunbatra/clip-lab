# Generated teaching images

Created 29 September 2026 using the built-in OpenAI image-generation tool. These are synthetic illustrations. They do not depict actual patients, surveyed sites or the video creators. We preserve the original outputs and crop their panels only during display and model preprocessing. Small unintended changes remain; the pairs do not establish causal interventions.

The prompts below record the supplied scene and control requirements. The image files and SHA-256 checksums are listed in [image provenance](images/provenance.json).

## hat-pair.png

Use case: scientific-educational, photorealistic-natural. Create ONE landscape image consisting of two equal square photographic panels side by side, no border/gutter/text/labels/watermark. It is a controlled before/after image pair for a CLIP embedding subtraction classroom experiment. Left panel: chest-up portrait of a fictional adult with short dark hair, plain charcoal crewneck shirt, relaxed neutral expression, arms down outside frame, centered against plain warm gray wall. Right panel: EXACT SAME person, face, pose, shirt, framing, lighting and background, with only one addition: a clearly visible natural straw sun hat with a dark band on their head. Both panels must have identical camera distance and generous headroom so hat fits entirely without cropping. Natural skin texture, soft daylight, realistic ordinary photograph. No celebrities. No visible writing anywhere. Images aligned exactly. The split must be at the exact horizontal midpoint. Purpose is isolate presence of the hat as much as possible. Output 1536x768 if possible.

## mug-pair.png

Use case: scientific-educational, photorealistic-natural. Create ONE landscape image of two equal square photographic panels side by side, with no border/gutter/text/labels/watermark. Controlled before/after pair for machine learning classroom. In each panel there is EXACTLY one plain ceramic mug with a C-shaped handle facing right, on a pale oak tabletop in front of a soft neutral kitchen wall. Identical camera, perspective, framing, shadows, object shape, position, and lighting in both panels. The ONLY difference: left mug saturated red, right mug saturated blue. Mug fills central 55% of panel with ample surrounding table. Eye-level close-up product photography, photorealistic subtle ceramic texture, soft daylight from left. Absolutely no letters or logos anywhere. Split exactly at horizontal midpoint. Output 1536x768 if possible.

## remote-sensing-scenes.png

Use case: scientific-educational. Make ONE square contact sheet with exactly FOUR equal square panels in a 2 by 2 grid, no gutters, no borders, no text or labels or watermarks. These will be four separate synthetic overhead remote-sensing teaching examples: upper left a plausible South Asian fixed-chimney brick kiln with an oval earth-colored firing trench, central tall chimney casting a shadow, many neat rows of drying red bricks, and dirt access paths; upper right a large solar farm with regular rows of dark blue photovoltaic panels on arid terrain and service lanes; lower left green and brown rectangular agricultural fields with narrow irrigation channels and a dirt road; lower right an industrial warehouse complex with large pale gray rectangular roofs, access roads and parking spaces. Photorealistic true-color, near-nadir overhead satellite/aerial style, detailed ground texture and natural muted colors. Scale and lighting consistent across the four panels, no tilted cinematic aerial perspective. They must be plausible but FICTIONAL scenes, not copies of a specific location or branded map. Exact panel boundaries at 50% width and height.

## photo-sketch-pair.png

Use case: scientific-educational, photorealistic-natural. ONE landscape two-panel image with exactly two equal square panels side by side, no margins/gutters/borders/text/labels. Left: a realistic ordinary photograph of a seated golden retriever dog in profile facing right on short green grass with softly blurred trees behind, full body and tail visible, daylight, natural fur detail. Right: a black graphite pencil sketch on plain off-white paper of exactly the same dog pose, size and framing, recognizable retriever silhouette, expressive drawn fur, no background except a few grounding pencil strokes. Purpose is test whether CLIP recognizes the same concept in a natural photo and a sketch. No people, no hats, no writing, no watermark. Panel split at exact midpoint.

## chimney-pair.png

Use case: scientific-educational. ONE wide photorealistic image with exactly TWO equal square panels side by side, no gutters or labels or text. Controlled synthetic aerial before/after pair. Both panels show the EXACT SAME fictional South Asian brick kiln from near-nadir aerial viewpoint, an oval earth-colored firing trench, rows of drying red bricks, dirt road and sparse green fields around it. Same camera, same sun angle, same geometry, same framing. Left panel has NO chimney: the center of the oval kiln is a low brick base only. Right panel adds exactly one tall narrow dark brick chimney rising from that base, with a realistic cast shadow, no smoke. Everything else must be unchanged. High-resolution satellite/aerial style with plausible ground texture. Do not show workers or markings or logos. Exact split at midpoint. This is a labelled-as-synthetic classroom experiment about an added chimney, not a real site survey.

## device-pair.png

Use case: scientific-educational. ONE wide image with exactly TWO equal square panels, side by side without borders/gutters/text/letters/labels. A clearly synthetic teaching illustration in realistic grayscale chest-radiograph style. Both panels show the same generic adult frontal chest X-ray at same scale and pose: lungs, ribs, heart silhouette, spine. Right panel differs ONLY by adding a clearly visible implanted cardiac pacemaker pulse generator in upper left chest as viewed in the image, with two thin radiopaque leads routed toward the heart; left panel has no device or leads. Keep rib positions and lung appearance identical. No disease cues, no patient identifiers or diagnostic annotations. Exact split at midpoint. This is for a concept-matching classroom demonstration, not clinical interpretation; prioritize making the added device visible while preserving the paired image anatomy.

The generator did not fully preserve the requested anatomy, placement or pixel-level invariants. We retain the result as an explicitly synthetic concept-matching test and report its failure to rank the intended word first. It must not be used as an anatomical reference.

## Programmatic small dataset

`images/tiny-test/` is not generated by an image model. It is rendered deterministically from circles, squares and triangles by `scripts/train_tiny.py`, with published random seeds. The training split is recreated from seed 101; the held-out split from seed 202.
