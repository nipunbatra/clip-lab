import {createEngine,dot,normalize,softmax} from './engine.js';
let engine;
self.onmessage=async({data:m})=>{
 try {
  if(m.type==='load'){
   const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
   engine=await createEngine(lib,{progress:p=>self.postMessage({type:'progress',progress:p})});
   self.postMessage({type:'ready'});return;
  }
  if(!engine)throw Error('Load CLIP first.');
  const {experiment:ex,gallery}=m;
  let labels,vs,q;
  if(ex.mode==='search'||ex.mode==='neighbors'){
   q=ex.mode==='search'?(await engine.text([ex.query]))[0]:await engine.image(gallery.find(g=>g.id===ex.image));
   labels=[];vs=[];
   for(const item of gallery){
    if(ex.mode==='neighbors'&&item.id===ex.image)continue;
    self.postMessage({type:'status',text:'Encoding gallery: '+item.title});
    labels.push(item.id);vs.push(await engine.image(item));
   }
  }else{
   q=await engine.image(gallery.find(g=>g.id===ex.image));
   if(ex.mode==='difference'){
    const after=await engine.image(gallery.find(g=>g.id===ex.second));
    q=normalize(after.map((v,i)=>v-q[i]));
   }
   labels=ex.candidates;vs=await engine.text(labels);
  }
  const scores=vs.map(v=>dot(q,v));
  const probs=softmax(scores.map(s=>s/0.01));
  const results=labels.map((label,i)=>({label,cosine:scores[i],...(ex.mode==='difference'?{}:{share:probs[i]})})).sort((a,b)=>b.cosine-a.cosine);
  self.postMessage({type:'result',results,experiment:ex,runtime:'Live browser inference · WebAssembly · q8',elapsed:performance.now()-m.started});
 }catch(error){self.postMessage({type:'error',text:error.message||String(error)});}
};
