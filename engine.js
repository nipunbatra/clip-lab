export const MODEL = 'Xenova/clip-vit-base-patch32';
export const REVISION = 'd15189d7028b43f1d3e65039190477f6af591c2a';
export const normalize = v => {const n=Math.hypot(...v); if(n<1e-9) throw Error('The difference is too small to normalize. Choose two different images.'); return v.map(x=>x/n);};
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export const softmax = v => {const m=Math.max(...v),e=v.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z);};
export async function createEngine(lib, {progress=()=>{},device='wasm',dtype='q8'}={}) {
  const {AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage, env}=lib;
  env.allowLocalModels=false;
  if(env.backends.onnx.wasm) env.backends.onnx.wasm.numThreads=1;
  const opts={revision:REVISION,device,dtype,progress_callback:progress};
  const tokenizer=await AutoTokenizer.from_pretrained(MODEL,opts);
  const processor=await AutoProcessor.from_pretrained(MODEL,opts);
  const textModel=await CLIPTextModelWithProjection.from_pretrained(MODEL,opts);
  const visionModel=await CLIPVisionModelWithProjection.from_pretrained(MODEL,opts);
  const imageCache=new Map(),textCache=new Map();
  const rows=t=>t.tolist().map(normalize);
  async function text(texts) {
    const missing=[...new Set(texts.filter(t=>!textCache.has(t)))];
    for(let i=0;i<missing.length;i+=8){
      const batch=missing.slice(i,i+8);
      const tokens=tokenizer(batch,{padding:true,truncation:true,max_length:77});
      const out=await textModel(tokens);
      rows(out.text_embeds).forEach((v,j)=>textCache.set(batch[j],v));
      for(const t of Object.values(out)) t.dispose?.();
    }
    return texts.map(t=>textCache.get(t));
  }
  async function image(item) {
    const key=item.url+JSON.stringify(item.crop)+JSON.stringify(item.panel)+JSON.stringify(item.quad);
    if(!imageCache.has(key)){
      let img=await RawImage.read(item.url);
      if(item.quad!==undefined){const w=Math.floor(img.width/2),h=Math.floor(img.height/2),x=item.quad%2,y=Math.floor(item.quad/2);img=await img.crop([x*w,y*h,(x+1)*w-1,(y+1)*h-1]);}
      if(item.panel!==undefined){const w=Math.floor(img.width/2);img=await img.crop([item.panel*w,0,(item.panel+1)*w-1,img.height-1]);}
      if(item.crop){const [l,t,r,b]=item.crop;img=await img.crop([l,t,img.width+r,img.height+b]);}
      const inputs=await processor(img);
      const out=await visionModel(inputs);
      imageCache.set(key,rows(out.image_embeds)[0]);
      for(const t of Object.values(out)) t.dispose?.();
    }
    return imageCache.get(key);
  }
  return {image,text,device,dtype,async dispose(){await textModel.dispose();await visionModel.dispose();}};
}
