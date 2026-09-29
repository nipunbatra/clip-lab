import {softmax,MODEL,REVISION} from './engine.js';
const $=s=>document.querySelector(s);
const [experiments,gallery]=await Promise.all(['experiments','gallery'].map(n=>fetch(`data/${n}.json`).then(r=>{if(!r.ok)throw Error('Could not load '+n);return r.json();})));
gallery.forEach(g=>g.url=new URL('images/'+g.file,location.href).href);
let current,worker,ready=false,busy=false,loading=false,uploadCount=0,lastResult;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function status(s,error=false){$('#status').textContent=s;$('#status').classList.toggle('error',error);}
function lock(v){busy=v;$('#run').disabled=v||loading;$('#recorded').disabled=v;$('#reset').disabled=v;document.querySelectorAll('#workspace input,#workspace select,#workspace textarea').forEach(x=>x.disabled=v);}
function clearResults(){lastResult=null;$('#results').replaceChildren();status('Inputs changed. Predict the outcome, then run.');}
function options(id){return gallery.map(g=>`<option value="${esc(g.id)}" ${g.id===id?'selected':''}>${esc(g.title)}</option>`).join('');}
function figure(g,caption=true){return `<figure><canvas class="picture" data-image="${esc(g.id)}" role="img" aria-label="${esc(g.title)}"></canvas>${caption?`<figcaption>${esc(g.title)}<br>${esc(g.credit)}</figcaption>`:''}</figure>`;}
async function paint(){await Promise.all([...document.querySelectorAll('canvas[data-image]')].map(async canvas=>{
 const g=gallery.find(g=>g.id===canvas.dataset.image);if(!g)return;
 try{const image=new Image();image.src=g.url;await image.decode();let x=0,y=0,w=image.width,h=image.height;
 if(g.panel!==undefined){w=Math.floor(w/2);x=g.panel*w;}if(g.quad!==undefined){w=Math.floor(w/2);h=Math.floor(h/2);x=(g.quad%2)*w;y=Math.floor(g.quad/2)*h;}
 canvas.width=w;canvas.height=h;canvas.getContext('2d').drawImage(image,x,y,w,h,0,0,w,h);
 }catch{canvas.replaceWith(Object.assign(document.createElement('p'),{textContent:'Image could not load: '+g.title}));}
}));}
function render(){
 $('#number').textContent=`Experiment ${String(experiments.findIndex(x=>x.id===current.id)+1).padStart(2,'0')} / ${experiments.length}`;
 $('#title').textContent=current.title;$('#question').textContent=current.question;$('#try').textContent=current.try;$('#lesson').textContent=current.lesson;
 $('#source').innerHTML=`Idea / method: <a href="${esc(current.sourceURL)}" target="_blank" rel="noopener">${esc(current.sourceLabel)}</a>. Images: <a href="sources.html">provenance & generated-image prompts</a>.`;
 document.querySelectorAll('#experiments a').forEach(a=>a.setAttribute('aria-current',String(a.hash==='#'+current.id)));
 const im=gallery.find(g=>g.id===current.image), second=gallery.find(g=>g.id===current.second);
 let left='',right='';
 if(current.mode==='search'){
  left=`<div class="controls"><label for="query">Search this gallery</label><input id="query" type="text" maxlength="240" value="${esc(current.query)}"><p class="fine">The model sees image pixels, not filenames or these gallery labels.</p></div>`;
  right=`<div class="gallery-preview">${gallery.map(g=>figure(g,false)).join('')}</div>`;
  $('#workspace').innerHTML=left+right;
 }else{
  left= current.mode==='difference'?`<div class="pair">${figure(im)}${figure(second)}</div><div class="pair"><label>Before<select id="image">${options(im.id)}</select></label><label>After<select id="second">${options(second.id)}</select></label></div><p class="fine">Direction = normalize(after − before), after each image embedding is normalized.</p>`:figure(im)+`<label class="fine">Choose image<select id="image">${options(im.id)}</select></label>`;
  left+=`<label class="fine">Use your own image <input id="upload" type="file" accept="image/png,image/jpeg,image/webp"></label>`;
  right=current.mode==='neighbors'?'<div class="controls"><p>Use this image as the query. Rank every other gallery image by cosine similarity.</p><p class="fine">The query itself is excluded. No text encoder is needed for this experiment.</p></div>':`<div class="controls"><label for="candidates">Candidate descriptions · one per line</label><textarea id="candidates" rows="6" maxlength="2000">${esc(current.candidates.join('\n'))}</textarea><p class="fine">Up to 12 candidates, 160 characters each. Long text is truncated to CLIP’s 77-token context.</p><p class="fine">${current.mode==='difference'?'Difference experiments show raw cosine only.':'Results show cosine and a relative softmax share at τ = 0.01. This fixed teaching temperature is not a calibrated confidence.'}</p></div>`;
  $('#workspace').innerHTML=`<div class="bench"><div>${left}</div>${right}</div>`;
 }
 $('#results').replaceChildren();status(ready?'Model ready. Predict first, then run.':'Run downloads CLIP once; recorded runs need no model download.');paint();
 $('#candidates')?.addEventListener('input',()=>{current.candidates=$('#candidates').value.split('\n').map(s=>s.trim()).filter(Boolean);clearResults();});
 $('#query')?.addEventListener('input',()=>{current.query=$('#query').value;clearResults();});
 for(const id of ['image','second'])$('#'+id)?.addEventListener('change',()=>{current[id]=$('#'+id).value;render();});
 $('#upload')?.addEventListener('change',async ev=>{
  const file=ev.target.files[0];if(!file)return;
  if(file.size>20e6){status('Please choose an image smaller than 20 MB.',true);return;}
  const url=URL.createObjectURL(file);try{const img=new Image();img.src=url;await img.decode();if(img.width*img.height>40e6)throw Error('Image is too large. Resize below 40 megapixels.');
   const item={id:'upload-'+(++uploadCount),title:file.name,credit:'Your local image · not uploaded',url};gallery.push(item);current.image=item.id;render();
  }catch(e){URL.revokeObjectURL(url);status(e.message||'Could not decode image.',true);}
 });
}
function select(){if(busy||loading){history.replaceState(null,'','#'+current.id);return;}current=structuredClone(experiments.find(e=>e.id===location.hash.slice(1))||experiments[0]);render();}
$('#experiments').innerHTML=experiments.map((e,i)=>`<a href="#${e.id}"><span>${String(i+1).padStart(2,'0')}</span>${esc(e.short)}</a>`).join('');
window.addEventListener('hashchange',select);select();
function validate(){
 if(current.candidates){if(current.candidates.length<2||current.candidates.length>12)throw Error('Enter 2–12 non-empty descriptions.');if(current.candidates.some(s=>s.length>160))throw Error('Keep each description under 160 characters.');if(new Set(current.candidates).size!==current.candidates.length)throw Error('Remove duplicate descriptions.');}
 if(current.mode==='search'&&!current.query.trim())throw Error('Enter a search query.');
 if(current.mode==='difference'&&current.image===current.second)throw Error('Choose two different images for the subtraction.');
}
let loadPromise;
function load(){
 if(ready)return Promise.resolve();if(loadPromise)return loadPromise;
 loading=true;$('#load').disabled=true;$('#run').disabled=true;$('#model-state').textContent='Downloading / preparing model…';$('#download').hidden=false;
 loadPromise=new Promise((resolve,reject)=>{
  worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data:m})=>{
   if(m.type==='progress'){const p=m.progress;$('#model-state').textContent=p.file?`${p.file.split('/').pop()}${p.progress?': '+p.progress.toFixed(0)+'%':''}`:'Preparing model…';if(p.progress){$('#download').max=100;$('#download').value=p.progress;}}
   if(m.type==='ready'){ready=true;loading=false;$('#model-state').textContent='Ready · WebAssembly · q8';$('#load').textContent='CLIP is loaded';$('#download').hidden=true;$('#run').disabled=false;status('Ready. All inference runs in this browser.');resolve();}
   if(m.type==='status')status(m.text);
   if(m.type==='result'){lock(false);renderResults(m.results,m.runtime,m.experiment);status('Live inference complete. Change one input and compare.');}
   if(m.type==='error'){const wasLoading=loading;loading=false;lock(false);status(m.text,true);if(wasLoading){worker.terminate();loadPromise=null;$('#load').disabled=false;$('#load').textContent='Retry loading CLIP';$('#model-state').textContent='Load failed. Check connection and retry.';$('#download').hidden=true;reject(Error(m.text));}}
  };
  worker.onerror=ev=>{loading=false;lock(false);ready=false;loadPromise=null;$('#load').disabled=false;$('#load').textContent='Retry loading CLIP';$('#download').hidden=true;status('Browser runtime failed. Reload or retry. '+ev.message,true);reject(Error(ev.message));};
  worker.postMessage({type:'load'});
 });return loadPromise;
}
$('#load').onclick=()=>load().catch(()=>{});
$('#run').onclick=async()=>{try{validate();await load();lock(true);status('Encoding inputs…');$('#results').replaceChildren();worker.postMessage({type:'run',experiment:structuredClone(current),gallery,started:performance.now()});}catch(e){status(e.message,true);}};
function renderResults(results,runtime,ex){
 lastResult={model:MODEL,revision:REVISION,created:new Date().toISOString(),temperature:ex.mode==='classify'?0.01:null,results,runtime,experiment:structuredClone(ex)};window.lastClipResult=lastResult;
 const images=ex.mode==='search'||ex.mode==='neighbors';
 const probs=softmax(results.map(r=>r.cosine/0.01));
 let html=`<p class="result-meta">${esc(runtime)} · ${images?'cosine similarity':'cosine'+(ex.mode==='difference'?' · signed direction':' · softmax share at τ = 0.01')}</p>`;
 if(images)html+=`<div class="result-gallery">${results.map((r,i)=>{const g=gallery.find(g=>g.id===r.label);return `<div>${figure(g)}<p><b>${i+1}. cosine ${r.cosine.toFixed(4)}</b></p></div>`;}).join('')}</div>`;
 else html+=results.map((r,i)=>`<div class="score-row"><div class="score-label">${esc(r.label)}<i style="width:${Math.max(0,r.cosine)*100}%"></i></div><b>${r.cosine.toFixed(4)}</b><small>${ex.mode==='difference'?'':(probs[i]*100).toFixed(1)+'%'}</small></div>`).join('');
 html+='<p class="fine">Save this run to compare exact inputs and scores.</p><button id="export">Download result JSON</button>';
 $('#results').innerHTML=html;paint();$('#export').onclick=()=>{const blob=new Blob([JSON.stringify(lastResult,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='clip-'+ex.id+'-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
}
$('#recorded').onclick=async()=>{try{const e=await fetch('data/evidence.json').then(r=>r.json());const original=experiments.find(x=>x.id===current.id);if(JSON.stringify(current)!==JSON.stringify(original))throw Error('Recorded results use the original example. Reset it first, or run live with your changes.');renderResults(e.runs[current.id],'Recorded inference · '+e.runtime+' · '+e.dtype,current);status('Recorded measurement, not a live run. Run experiment to recompute locally.');}catch(e){status(e.message,true);}};
$('#reset').onclick=()=>{current=structuredClone(experiments.find(e=>e.id===current.id));render();};
window.addEventListener('pagehide',()=>{worker?.terminate();gallery.filter(g=>g.url.startsWith('blob:')).forEach(g=>URL.revokeObjectURL(g.url));});
window.clipLab={experiments,gallery,get current(){return current;},get ready(){return ready;}};
