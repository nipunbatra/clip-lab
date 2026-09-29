"""Train both encoders from random initialization on a tiny paired synthetic dataset.
No OpenAI CLIP weights, no pretrained features, no fitted evaluation images.
"""
from pathlib import Path
import hashlib,json,random,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
COLORS={'red':(220,45,45),'green':(30,155,75),'blue':(45,90,220)}
SHAPES=['circle','square','triangle']
LABELS=[f'{c} {s}' for c in COLORS for s in SHAPES]

def dataset(seed,n):
 rng=np.random.default_rng(seed);images=[];labels=[];raw=[]
 for label in range(9):
  color=list(COLORS.values())[label//3];shape=SHAPES[label%3]
  for _ in range(n):
   bg=int(rng.integers(230,256));im=Image.new('RGB',(32,32),(bg,bg,bg));d=ImageDraw.Draw(im)
   size=int(rng.integers(12,23));x=int(rng.integers(2,31-size));y=int(rng.integers(2,31-size))
   rgb=tuple(int(np.clip(v+rng.integers(-16,17),0,255)) for v in color)
   if shape=='circle':d.ellipse((x,y,x+size,y+size),fill=rgb)
   elif shape=='square':d.rectangle((x,y,x+size,y+size),fill=rgb)
   else:d.polygon([(x+size//2,y),(x,y+size),(x+size,y+size)],fill=rgb)
   a=np.asarray(im).copy();raw.append(im);images.append(a);labels.append(label)
 return torch.tensor(np.stack(images).transpose(0,3,1,2)/255,dtype=torch.float32),torch.tensor(labels),raw

class TinyCLIP(nn.Module):
 def __init__(self):
  super().__init__()
  self.image=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(24*8*8,64),nn.ReLU(),nn.Linear(64,24))
  self.words=nn.Embedding(6,24)
  self.text_projection=nn.Linear(48,24)
  self.log_scale=nn.Parameter(torch.tensor(np.log(1/.07),dtype=torch.float32))
 def text(self,tokens):return F.normalize(self.text_projection(self.words(tokens).flatten(1)),dim=-1)
 def encode_image(self,x):return F.normalize(self.image(x),dim=-1)
 def forward(self,x,tokens):return self.log_scale.exp()*self.encode_image(x)@self.text(tokens).T

def train(steps=500,seed=7,write=True):
 torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 torch.use_deterministic_algorithms(True)
 x,y,raw=dataset(101,30);xt,yt,test_raw=dataset(202,10)
 train_hash={hashlib.sha256(np.asarray(i).tobytes()).hexdigest() for i in raw}
 test_hash={hashlib.sha256(np.asarray(i).tobytes()).hexdigest() for i in test_raw}
 assert train_hash.isdisjoint(test_hash)
 tokens=torch.tensor([[c,s+3] for c in range(3) for s in range(3)])
 model=TinyCLIP();opt=torch.optim.Adam(model.parameters(),lr=.001)
 @torch.no_grad()
 def evaluate():
  model.eval();scores=model(xt,tokens);p=scores.argmax(1);return {'accuracy':float((p==yt).float().mean()),'cosine':(model.encode_image(xt)@model.text(tokens).T).tolist(),'predictions':p.tolist(),'image_embeddings':model.encode_image(xt).tolist(),'text_embeddings':model.text(tokens).tolist(),'logit_scale':model.log_scale.exp().item()}
 before=evaluate();history=[];start=time.time()
 if write:torch.save(model.state_dict(),ROOT/'output/tiny-before.pt')
 for step in range(steps):
  model.train()
  # One item from each semantic class: duplicate captions are never false negatives.
  ids=torch.arange(9)*30+torch.randint(30,(9,))
  logits=model(x[ids],tokens);target=torch.arange(9)
  loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
  opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():model.log_scale.clamp_(0,np.log(100))
  if step%25==0 or step==steps-1:history.append({'step':step+1,'train_loss':loss.item()})
 after=evaluate()
 result={'kind':'actual training from random initialization','seed':seed,'steps':steps,'train_images':len(x),'test_images':len(xt),'train_seed':101,'test_seed':202,'split_overlap':False,'classes':LABELS,'test_labels':yt.tolist(),'before':before,'after':after,'history':history,'seconds':time.time()-start,'torch_version':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),'scope':'Tiny CNN image encoder + two-token learned text encoder, trained jointly. Not pretrained CLIP, not zero-shot to unseen concepts. Test uses new positions, sizes and colors within the same nine concepts. Test results are evaluated only before and after a fixed 500 steps; no test-driven early stopping.'}
 if write:
  (ROOT/'images/tiny-test').mkdir(exist_ok=True)
  for i,im in enumerate(test_raw):im.save(ROOT/f'images/tiny-test/{i:03}.png')
  (ROOT/'data/training.json').write_text(json.dumps(result,indent=2));torch.save(model.state_dict(),ROOT/'output/tiny-after.pt')
 print(json.dumps({k:result[k] for k in ['train_images','test_images','parameters','seconds']},indent=2));print('Accuracy before',before['accuracy'],'after',after['accuracy'])
 return result
if __name__=='__main__':train()
