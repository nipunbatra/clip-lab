import {createEngine,dot} from './engine.js';
let engine;
self.onmessage=async({data:m})=>{try{
 if(m.type==='load'){
  if(engine)await engine.dispose();engine=null;
  const lib=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
  let device='wasm',dtype='q8';
  if(m.backend==='auto'&&self.navigator.gpu){const adapter=await self.navigator.gpu.requestAdapter();if(adapter){device='webgpu';dtype='fp32';}}
  const progress=p=>self.postMessage({type:'status',text:`Loading ${p.file||'CLIP'}${p.progress?' · '+Math.round(p.progress)+'%':''} · ${device}/${dtype}`});
  const encode=async()=>{
   engine=await createEngine(lib,{device,dtype,progress});
   const files=['newfoundland_31.jpg','Persian_98.jpg','pug_57.jpg'],texts=['a photo of a Newfoundland','a photo of a Persian cat','a photo of a pug'];
   const U=[];for(const file of files){self.postMessage({type:'status',text:'Encoding '+file+' · '+device});U.push(await engine.image({url:new URL('./images/'+file,self.location.href).href}));}
   const V=await engine.text(texts);return{U,V};
  };
  let output;
  try{output=await encode();}
  catch(err){
   if(device!=='webgpu')throw err;
   try{await engine?.dispose();}catch{}engine=null;
   self.postMessage({type:'status',text:'WebGPU could not run this model. Loading the WebAssembly q8 fallback.'});
   device='wasm';dtype='q8';output=await encode();
  }
  self.postMessage({type:'ready',...output,device,dtype});
 }else if(m.type==='match'){
  if(!engine)throw Error('Load CLIP first.');const [v]=await engine.text([m.text]);self.postMessage({type:'match',text:m.text,cosine:dot(m.image,v)});
 }
}catch(e){self.postMessage({type:'error',text:e.message||String(e)});}};
