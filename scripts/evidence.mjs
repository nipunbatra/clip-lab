import * as lib from '@huggingface/transformers';
import fs from 'node:fs/promises';
import {createEngine,MODEL,REVISION,dot,normalize,softmax} from '../engine.js';
lib.env.cacheDir='./.cache';
const gallery=JSON.parse(await fs.readFile('data/gallery.json'));
const experiments=JSON.parse(await fs.readFile('data/experiments.json'));
const e=await createEngine(lib,{device:'cpu',progress:p=>{if(p.status==='done')console.log('Loaded',p.file);}});
const images={};for(const im of gallery){images[im.id]=await e.image({...im,url:'images/'+im.file});console.log('Encoded',im.id);}
const texts=[...new Set(experiments.flatMap(x=>x.candidates||[x.query]).filter(Boolean))];
const vectors=await e.text(texts);const text=Object.fromEntries(texts.map((s,i)=>[s,vectors[i]]));
const runs={};
for(const ex of experiments){
 let q,labels,vs;
 if(ex.mode==='search'){q=text[ex.query];labels=gallery.map(x=>x.id);vs=labels.map(x=>images[x]);}
 else if(ex.mode==='neighbors'){q=images[ex.image];labels=gallery.map(x=>x.id).filter(x=>x!==ex.image);vs=labels.map(x=>images[x]);}
 else {q=ex.mode==='difference'?normalize(images[ex.second].map((v,i)=>v-images[ex.image][i])):images[ex.image];labels=ex.candidates;vs=labels.map(x=>text[x]);}
 const scores=vs.map(v=>dot(q,v));runs[ex.id]=labels.map((label,i)=>({label,cosine:scores[i]})).sort((a,b)=>b.cosine-a.cosine);console.log(ex.id,runs[ex.id]);
}
await fs.writeFile('data/evidence.json',JSON.stringify({model:MODEL,revision:REVISION,runtime:'Transformers.js 3.8.1 / ONNX Runtime Node CPU',dtype:'q8',created:new Date().toISOString(),note:'Measured embeddings, not synthetic examples. Browser live inference recomputes these; raw cosine is the comparison score.',images,text,runs},null,2));
await e.dispose();
