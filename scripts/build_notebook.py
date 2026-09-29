from pathlib import Path
import nbformat as n
ROOT=Path(__file__).resolve().parents[1]
c=[]
def md(s):c.append(n.v4.new_markdown_cell(s))
def code(s):c.append(n.v4.new_code_cell(s))
md('''# CLIP Lab: experience first, explain second

Run real OpenAI CLIP on public photographs and labelled generated pairs, inspect its vectors, and train a tiny two-encoder model from scratch.

**Predict → run → change one input → explain.** [Browser lab](https://nipunbatra.github.io/clip-lab/) · [Full source credits](https://nipunbatra.github.io/clip-lab/sources.html).

The hat subtraction idea is adapted from [CodeEmporium, 7:25](https://www.youtube.com/watch?v=FCDKn-vpn_o&t=445s). We use our own generated images, not the creator’s portraits. Search follows [Supabase](https://www.youtube.com/watch?v=S7VZErcTN5Y); similarity analysis is inspired by [Roboflow, 10:40](https://www.youtube.com/watch?v=YxJkE6FvGF4&t=640s). The mathematical method is [Radford et al. (2021)](https://proceedings.mlr.press/v139/radford21a.html).

The Python notebook uses original OpenAI weights in float32. The browser uses the pinned Xenova ONNX conversion in q8. Small score differences are expected from quantization and image resizing. All numbers below are execution outputs, not handwritten predictions.''')
md('''## 0. Set up
Clone the repository and run from its root. In Colab, run the following setup cell once. Model download is approximately 338 MB for the original checkpoint. No API key or paid inference endpoint is used.''')
code('''import os, sys, subprocess
from pathlib import Path
if 'google.colab' in sys.modules:
    if not Path('/content/clip-lab').exists():
        subprocess.run(['git','clone','https://github.com/nipunbatra/clip-lab.git','/content/clip-lab'],check=True)
    os.chdir('/content/clip-lab')
    subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements.txt'],check=True)
ROOT = Path.cwd()
if not (ROOT/'data/gallery.json').exists(): ROOT = ROOT.parent
assert (ROOT/'data/gallery.json').exists(), 'Run inside the cloned clip-lab repository.'
print('Project:', ROOT.name)''')
code('''import json, numpy as np, torch, clip
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
torch.set_num_threads(2)
DEVICE = 'cpu'  # reproducible reference; choose cuda if available
model, preprocess = clip.load('ViT-B/32', device=DEVICE, jit=False)
model.eval()
gallery = json.loads((ROOT/'data/gallery.json').read_text())
experiments = json.loads((ROOT/'data/experiments.json').read_text())

def read_image(item):
    im = Image.open(ROOT/'images'/item['file']).convert('RGB')
    w,h = im.size
    if 'panel' in item:
        width=w//2; x=item['panel']*width; im=im.crop((x,0,x+width,h))
    elif 'quad' in item:
        width,height=w//2,h//2; x=item['quad']%2*width; y=item['quad']//2*height
        im=im.crop((x,y,x+width,y+height))
    return im

@torch.inference_mode()
def encode_images(items):
    z=model.encode_image(torch.stack([preprocess(read_image(g)) for g in items]).to(DEVICE)).float()
    return z/z.norm(dim=-1,keepdim=True)

@torch.inference_mode()
def encode_text(texts):
    z=model.encode_text(clip.tokenize(texts,truncate=False).to(DEVICE)).float()
    return z/z.norm(dim=-1,keepdim=True)

U=encode_images(gallery)
by_id={g['id']:i for i,g in enumerate(gallery)}
print('Image embeddings:',tuple(U.shape),'learned logit scale:',model.logit_scale.exp().item())
assert U.shape==(len(gallery),512)
assert torch.allclose(U.norm(dim=-1),torch.ones(len(gallery)),atol=1e-5)''')
md('''## 1. Same photo, new task
Before running: would the Vision I classifier head accept a new class just because we typed its name? What has to replace that fixed head here?''')
code('''def compare(image_id, descriptions):
    v=encode_text(descriptions); cosine=(U[by_id[image_id]]@v.T).cpu().numpy()
    share=torch.tensor(cosine*model.logit_scale.exp().item()).softmax(-1).numpy()
    for j in np.argsort(-cosine):print(f'{descriptions[j]:50} cosine={cosine[j]:+.4f}  candidate share={share[j]:.3f}')
    return cosine

display(read_image(gallery[by_id['newfoundland']]).resize((350,234)))
compare('newfoundland',['a photo of a dog','a photo of a cat','a photo of a bird'])
print('Now change the task:')
compare('newfoundland',['a photo of a Newfoundland','a photo of a pug','a photo of a Persian cat']);''')
md('''## 2. Search, without filenames
The gallery is encoded once. A query changes only the text vector. [Application inspiration: Supabase](https://www.youtube.com/watch?v=S7VZErcTN5Y&t=414s). Try a query that has no good answer too.''')
code('''query='something to drink'  # Try: a pet / a spacecraft taking off / a laptop
scores=(U@encode_text([query]).T).squeeze().cpu().numpy()
fig,axes=plt.subplots(1,4,figsize=(12,3))
for ax,j in zip(axes,np.argsort(-scores)[:4]):
    ax.imshow(read_image(gallery[j]));ax.set_title(f"{gallery[j]['id']}\\ncosine {scores[j]:.3f}");ax.axis('off')
plt.show()''')
md('''## 3. What changed? Test a direction
[CodeEmporium’s hat experiment](https://www.youtube.com/watch?v=FCDKn-vpn_o&t=445s) motivates this. These paired pictures are generated specifically for this lesson. The chimney, device, colour and style pairs are our extensions. No CodeEmporium photos are redistributed.

Normalize each image vector, subtract **after − before**, normalize the difference, then compare to text. There is no trained change-detection head and no guarantee of clean semantic subtraction. The synthetic medical pair is not anatomically validated or a clinical experiment.''')
code('''def difference(before,after,words):
    delta=U[by_id[after]]-U[by_id[before]]
    assert delta.norm()>1e-8
    delta=delta/delta.norm()
    sims=(delta@encode_text(words).T).cpu().numpy()
    for j in np.argsort(-sims):print(f'{words[j]:20} {sims[j]:+.4f}')
    return sims

for ex in [e for e in experiments if e['mode']=='difference']:
    print('\\n'+ex['title'])
    difference(ex['image'],ex['second'],ex['candidates'])''')
md('''**Work it out:** reverse the pair. Every cosine must change sign if the text vectors stay fixed. Does the top label remain top? Which extra scene details changed besides the intended object?''')
code('''a=difference('portrait','portrait-hat',['hat','cup','cat','boat'])
b=difference('portrait-hat','portrait',['hat','cup','cat','boat'])
np.testing.assert_allclose(a,-b,atol=1e-6)
print('Reversal check passed.')''')
md('''## 4. Domain transfer is a question, not a promise
Use descriptions for public medical imagery, synthetic overhead landscapes and generated sketches. The public chest radiograph is credited to Stillwaterising (CC0). The satellite-style panels are fictional scenes, not observations of actual sites. Matching an imaging modality does not demonstrate disease diagnosis.''')
code('''for id in ['sustainability','healthcare','sketch','negation','competition']:
    ex=next(x for x in experiments if x['id']==id)
    print('\\n'+ex['title']);compare(ex['image'],ex['candidates'])''')
md('''## 5. Two directions, one score matrix
[Radford et al., Figure 1 and §2.2](https://proceedings.mlr.press/v139/radford21a.html). Rows are images; columns are descriptions. For training, paired image i and text i supply diagonal targets. Row softmax answers “which caption?”; column softmax answers “which image?”.''')
code('''ids=['newfoundland','Persian','pug']
texts=['a photo of a Newfoundland','a photo of a Persian cat','a photo of a pug']
A=U[[by_id[i] for i in ids]]; B=encode_text(texts)
cosine=A@B.T
logits=model.logit_scale.exp()*cosine
target=torch.arange(3)
loss_i=torch.nn.functional.cross_entropy(logits,target)
loss_t=torch.nn.functional.cross_entropy(logits.T,target)
print('Cosine matrix:\\n',cosine)
print('Row probabilities:\\n',logits.softmax(1))
print('Column probabilities:\\n',logits.softmax(0))
print('Symmetric loss:',((loss_i+loss_t)/2).item())
# This diagnoses these three selected pairs; it is not a training or benchmark claim.
assert torch.allclose(logits.softmax(1).sum(1),torch.ones(3),atol=1e-5)''')
md('''## 6. Watch alignment being learned, from scratch
We now switch models and data deliberately. This small experiment trains a CNN image encoder and a two-token learned text encoder from random initialization on **270 coloured-shape images**. It uses no CLIP weights or pretrained features. There are nine familiar colour–shape descriptions. Ninety held-out images vary position, size, shade and background; the nine concepts themselves are not unseen.

We keep one item per concept in each batch so identical descriptions are not treated as negatives. Both encoders and a log scale learn with the same bidirectional contrastive loss. The fixed 500-step budget is specified before evaluation. This is a demonstration of learning alignment, not a small reproduction of web-scale CLIP. [Objective source: Radford et al.](https://proceedings.mlr.press/v139/radford21a.html).''')
code('''sys.path.insert(0,str(ROOT/'scripts'))
from train_tiny import train
training=train(steps=500,seed=7,write=False)
print(f"Held-out accuracy: {training['before']['accuracy']:.1%} → {training['after']['accuracy']:.1%}")
assert training['split_overlap'] is False
assert training['after']['accuracy']>training['before']['accuracy']''')
code('''fig,axes=plt.subplots(1,3,figsize=(13,3.6))
axes[0].plot([h['step'] for h in training['history']],[h['train_loss'] for h in training['history']]);axes[0].set(xlabel='Training step',ylabel='Training pair loss')
for ax,key in zip(axes[1:],['before','after']):
    # One held-out image per concept; same examples before and after.
    indices=np.arange(9)*10
    im=ax.imshow(np.asarray(training[key]['cosine'])[indices],vmin=-1,vmax=1,cmap='coolwarm')
    ax.set(title=key+' training',xlabel='Text index',ylabel='Held-out image index')
plt.colorbar(im,ax=axes[1:],label='Cosine');plt.show()''')
md('''## 7. Inspect the code; rerun a meaningful ablation
The source below is the exact implementation just executed. Try training with shuffled pair targets or freezing one encoder, keeping the same test split and fixed step budget. Record all runs; do not select a checkpoint using test accuracy.

The toy model’s text path concatenates two learned token embeddings and projects them. It is intentionally smaller than CLIP’s causal Transformer. Its CNN image path replaces the ViT. The shared-space projections, normalization, temperature, score matrix and loss are the mechanisms under study.''')
code("print((ROOT/'scripts/train_tiny.py').read_text())")
md('''## References and ownership
- [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html) and [OpenAI implementation](https://github.com/openai/CLIP).
- [CodeEmporium](https://www.youtube.com/watch?v=FCDKn-vpn_o): mechanism walkthrough and hat-vector experiment.
- [Computerphile](https://www.youtube.com/watch?v=KcSXcpluDe4): why fixed labels are limiting; representation before downstream use.
- [Yannic Kilcher](https://www.youtube.com/watch?v=T9XSU0pKX2E): paired matching objective and zero-shot classifier construction.
- [Data Science Gems](https://www.youtube.com/watch?v=rdMnjvjSAkQ): task diversity, prompt context and evaluation.
- [Supabase](https://www.youtube.com/watch?v=S7VZErcTN5Y): gallery search.
- [Roboflow](https://www.youtube.com/watch?v=YxJkE6FvGF4&t=640s): image embeddings, similarity and dataset exploration.
- [Image provenance and generated-image prompts](https://nipunbatra.github.io/clip-lab/sources.html).

The classroom exercises, tiny shape dataset, tiny model implementation and experimental extensions are newly authored for this lab. Source ideas remain credited. Generated pictures are synthetic teaching material; measured model outputs remain measured results, including failures.''')
nb=n.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
n.write(nb,ROOT/'notebooks/clip-lab.ipynb');print(len(c),'cells')
