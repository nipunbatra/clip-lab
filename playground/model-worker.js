import {MODEL,REVISION} from '../engine.js';
import {norm,unit,dot,difference} from './math.js';
let model,loadPromise;
const post=(type,data={})=>self.postMessage({type,...data});
async function load(backend) {
 if(model){await model.textModel.dispose();await model.visionModel.dispose();model=null;}
 const {AutoTokenizer,AutoProcessor,CLIPTextModelWithProjection,CLIPVisionModelWithProjection,RawImage,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
 env.allowLocalModels=false;env.backends.onnx.wasm.numThreads=1;
 let device='wasm',dtype='q8';
 if(backend!=='wasm'&&self.navigator.gpu){const adapter=await self.navigator.gpu.requestAdapter();if(adapter){device='webgpu';dtype='fp32';}}
 let fallback=backend!=='wasm'&&device==='wasm'?'WebGPU is unavailable; using the CPU fallback.':'';
 async function build(){
  const opts={revision:REVISION,device,dtype,progress_callback:p=>post('progress',{file:p.file,status:p.status,loaded:p.loaded,total:p.total,progress:p.progress,device,dtype})};
  const tokenizer=await AutoTokenizer.from_pretrained(MODEL,opts),processor=await AutoProcessor.from_pretrained(MODEL,opts);
  let textModel,visionModel;
  try{textModel=await CLIPTextModelWithProjection.from_pretrained(MODEL,opts);visionModel=await CLIPVisionModelWithProjection.from_pretrained(MODEL,opts);}
  catch(e){await textModel?.dispose();await visionModel?.dispose();throw e;}
  return {tokenizer,processor,textModel,visionModel,RawImage,device,dtype,imageCache:new Map(),textCache:new Map()};
 }
 try{model=await build();}catch(e){if(device!=='webgpu')throw e;fallback='WebGPU could not initialize this model; using the CPU fallback.';post('status',{text:fallback});device='wasm';dtype='q8';model=await build();}
 post('ready',{device,dtype,model:MODEL,revision:REVISION,fallback});
}
function vector(raw,extra){const v=Array.from(raw);return {...extra,raw:v,norm:norm(v),unit:unit(v)};}
async function image(item){
 const key=item.url+JSON.stringify([item.panel,item.quad]);if(model.imageCache.has(key))return {...model.imageCache.get(key),id:'image:'+item.id,label:item.title};
 post('status',{text:'Encoding '+item.title+'…'});
 let image=await model.RawImage.read(item.url);
 if(item.quad!==undefined){const w=Math.floor(image.width/2),h=Math.floor(image.height/2),x=item.quad%2,y=Math.floor(item.quad/2);image=await image.crop([x*w,y*h,(x+1)*w-1,(y+1)*h-1]);}
 if(item.panel!==undefined){const w=Math.floor(image.width/2);image=await image.crop([item.panel*w,0,(item.panel+1)*w-1,image.height-1]);}
 const inputs=await model.processor(image);let outputs;
 try{outputs=await model.visionModel(inputs);const result=vector(outputs.image_embeds.data,{id:'image:'+item.id,label:item.title,kind:'image',imageId:item.id});model.imageCache.set(key,result);return result;}
 finally{Object.values(outputs||{}).forEach(t=>t.dispose?.());Object.values(inputs).forEach(t=>t.dispose?.());}
}
async function text(value,index){
 const count=model.tokenizer.encode(value).length;if(count>77)throw Error('A description has '+count+' tokens. CLIP accepts at most 77 including its start/end tokens; shorten that description.');
 if(!model.textCache.has(value)){
  post('status',{text:'Encoding “'+value.slice(0,55)+'”…'});
  const inputs=model.tokenizer([value],{padding:true,truncation:false});let outputs;
  try{outputs=await model.textModel(inputs);model.textCache.set(value,vector(outputs.text_embeds.data,{kind:'text',label:value,tokenCount:count}));}
  finally{Object.values(outputs||{}).forEach(t=>t.dispose?.());Object.values(inputs).forEach(t=>t.dispose?.());}
 }
 return {...model.textCache.get(value),id:'text:'+index};
}
async function run(m){
 if(!model)throw Error('Load CLIP first.');const started=performance.now(),vectors=[];let query,candidates;
 if(m.mode==='search'){
  query=await text(m.query,0);vectors.push(query);candidates=[];for(const item of m.gallery)candidates.push(await image(item));vectors.push(...candidates);
 }else{
  query=await image(m.images[0]);vectors.push(query);
  if(m.mode==='difference'){
   const after=await image(m.images[1]);vectors.push(after);
   query=vector(difference(query.unit,after.unit),{id:'change',kind:'change',label:'After − before'});vectors.push(query);
  }
  candidates=[];for(let i=0;i<m.texts.length;i++)candidates.push(await text(m.texts[i],i));vectors.push(...candidates);
 }
 const scores=candidates.map(v=>({id:v.id,label:v.label,imageId:v.imageId,cosine:dot(query.unit,v.unit)})).sort((a,b)=>b.cosine-a.cosine);
 post('result',{requestId:m.requestId,result:{mode:m.mode,queryId:query.id,vectors,scores,model:MODEL,revision:REVISION,library:'Transformers.js 3.8.1',device:model.device,dtype:model.dtype,elapsedMs:performance.now()-started,computedAt:new Date().toISOString()}});
}
self.onmessage=async({data:m})=>{try{if(m.type==='load'){if(loadPromise)return;loadPromise=load(m.backend);try{await loadPromise;}finally{loadPromise=null;}}else if(m.type==='run'){if(loadPromise)await loadPromise;await run(m);}}catch(e){post('error',{text:e.message||String(e),requestId:m.requestId});}};
