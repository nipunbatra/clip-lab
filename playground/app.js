import {questions} from './questions.js';
import {norm,shares,contributions} from './math.js';
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=5)=>(n>=0?'+':'−')+Math.abs(n).toFixed(d),short=v=>'['+v.slice(0,3).map(x=>fmt(x,4)).join(', ')+', …]';
const state={mode:'match',image:'chelsea',second:'portrait-hat',slot:'image',gallery:[],ready:false,loading:false,busy:false,stage:0,result:null,candidate:null,start:0,tau:.07,requestId:0,stale:false,pending:false};
let worker;const uploads=[];
function picture(item,label=''){
 let style='',crop=false;if(item.panel!==undefined){style=`width:200%;height:100%;left:${-100*item.panel}%;top:0`;crop=true;}
 if(item.quad!==undefined){style=`width:200%;height:200%;left:${-100*(item.quad%2)}%;top:${-100*Math.floor(item.quad/2)}%`;crop=true;}
 return `<div class="picture${crop?' crop':''}"><img src="${esc(item.url)}" alt="${esc(item.title)}" ${style?`style="${style}"`:''}>${label?`<span class="slot-label">${esc(label)}</span>`:''}</div>`;
}
function item(id){return state.gallery.find(x=>x.id===id);}
function err(message=''){ $('error').hidden=!message;$('error').textContent=message;}
function locks(){
 $('load').disabled=state.loading||state.busy;$('backend').disabled=state.loading||state.busy;
 $('run').disabled=state.loading||state.busy;$('run').textContent=state.loading?'Loading CLIP…':state.busy?'Computing…':state.ready?'Run comparison':'Load CLIP & compare';
 document.querySelectorAll('#question-select,.modes button,#descriptions,#upload,#swap,#gallery button,.slot-choice').forEach(n=>n.disabled=state.busy);
}
function dirty(){if(state.result){state.stale=true;$('stale').hidden=false;}err();}
function renderInputs(){
 document.querySelectorAll('[data-mode]').forEach(n=>n.setAttribute('aria-pressed',n.dataset.mode===state.mode));
 const search=state.mode==='search',pair=state.mode==='difference';
 $('image-heading').textContent=search?'1. Browse the gallery':pair?'1. Choose before and after':'1. Choose an image';
 $('text-heading').textContent=search?'2. Describe what to find':pair?'2. Name possible changes':'2. Write the descriptions';
 $('text-label').textContent=search?'Your search description':'One description per line';
 $('text-help').textContent=search?'One description is compared with every gallery image.':'Use 1–12 descriptions. CLIP scores the words you supply; it does not write a caption.';
 $('swap').hidden=!pair;$('gallery-note').hidden=!search;$('gallery-count').textContent=state.gallery.length;$('sample-count').textContent=`· ${state.gallery.length} images`;
 $('selected-images').classList.toggle('pair',pair);
 $('selected-images').innerHTML=search?'<div class="search-preview">'+[...state.gallery.slice(0,3)].map(x=>picture(x)).join('')+'</div>':(pair?['image','second']:['image']).map(slot=>{const im=item(state[slot]);return `<figure class="selected-figure">${picture(im,pair?(slot==='image'?'Before':'After'):'')}<figcaption><b>${esc(im.title)}</b><br>${esc(im.credit)}</figcaption>${pair?`<button class="slot-choice" data-slot="${slot}" aria-pressed="${state.slot===slot}">Choose ${slot==='image'?'before':'after'} image</button>`:''}</figure>`;}).join('');
 $('gallery-help').textContent=search?'Every sample is a search candidate. Generated teaching images are labelled.':`Select an image${pair?' for '+(state.slot==='image'?'before':'after'):''}.`;
 $('gallery').innerHTML=state.gallery.map(x=>`<button data-image="${esc(x.id)}" title="${esc(x.credit)}" aria-label="Select ${esc(x.title)}" aria-pressed="${!search&&state[state.slot]===x.id}">${picture(x)}<span class="label">${esc(x.title)}</span></button>`).join('');
 locks();
}
function applyQuestion(id){const q=questions.find(x=>x.id===id)||questions[0];state.mode=q.mode;state.image=q.image||'chelsea';state.second=q.second||'portrait-hat';state.slot='image';$('question-select').value=q.id;$('question-title').textContent=q.question;$('question-try').textContent=q.try;$('question-notice').textContent=q.notice;$('descriptions').value=q.query||q.texts.join('\n');$('prediction').value='';state.result=null;$('results').hidden=true;err();renderInputs();}
function inputSnapshot(){
 const content=$('descriptions').value.trim();if(!content)throw Error(state.mode==='search'?'Write something to search for.':'Write at least one description.');
 const texts=content.split('\n').map(s=>s.trim()).filter(Boolean);if(state.mode!=='search'&&(texts.length>12||new Set(texts).size!==texts.length))throw Error('Use up to 12 different descriptions, one per line.');
 if(state.mode==='search'&&texts.length!==1)throw Error('Use one line for a gallery search.');if(content.length>6000)throw Error('Please shorten the descriptions.');
 if(state.mode==='difference'&&state.image===state.second)throw Error('Choose different before and after images. An identical pair has a zero difference.');
 return {mode:state.mode,texts,query:content,images:[item(state.image),...(state.mode==='difference'?[item(state.second)]:[])],gallery:state.gallery.map(x=>({...x})),prediction:$('prediction').value.trim()};
}
function initWorker(){
 worker=new Worker(new URL('./model-worker.js',import.meta.url),{type:'module'});
 worker.onerror=e=>{state.loading=false;state.busy=false;state.pending=false;err('The model worker stopped. Try loading CLIP again. '+e.message);worker.terminate();worker=null;state.ready=false;locks();};
 worker.onmessage=({data:m})=>{
  if(m.type==='progress'){
   $('runtime-status').textContent=`Loading ${m.file||'CLIP'} · ${m.device} / ${m.dtype}${m.total?' · '+(m.loaded/1e6).toFixed(1)+' / '+(m.total/1e6).toFixed(1)+' MB':''}`;
   $('download').hidden=false;if(m.progress!==undefined){$('download').max=100;$('download').value=m.progress;}else $('download').removeAttribute('value');
  }else if(m.type==='status'){$(state.loading?'runtime-status':'run-status').textContent=m.text;
  }else if(m.type==='ready'){
   state.loading=false;state.ready=true;$('download').hidden=true;$('runtime-badge').textContent=`Ready · ${m.device==='webgpu'?'WebGPU':'CPU / WASM'} · ${m.dtype}`;$('runtime-status').textContent=m.fallback||'Both encoders are loaded. New inputs will be encoded on this device.';$('load').textContent='Reload model';$('run-status').textContent='Ready. Make a prediction, then run the comparison.';locks();if(state.pending){state.pending=false;run();}
  }else if(m.type==='error'){
   const loading=state.loading;state.loading=false;state.busy=false;state.pending=false;$('download').hidden=true;if(loading){state.ready=false;$('runtime-badge').textContent='Model not loaded';$('runtime-status').textContent='Loading failed. Check your connection or choose the CPU option, then retry.';}$('run-status').textContent='The comparison did not complete.';err(m.text);locks();
  }else if(m.type==='result'){
   if(m.requestId!==state.requestId)return;
   state.busy=false;state.result={...m.result,inputs:state.runningInputs};state.stage=0;state.start=0;state.candidate=m.result.scores[0].id;state.stale=false;$('stale').hidden=true;$('run-status').textContent=`Computed in ${(m.result.elapsedMs/1000).toFixed(2)} s. Inspect the steps below.`;locks();renderResults();$('results').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  }
 };
}
function load(){if(state.loading||state.busy)return;state.loading=true;state.ready=false;err();$('runtime-badge').textContent='Loading both encoders';$('runtime-status').textContent='Connecting to the pinned model files…';$('download').hidden=false;$('download').removeAttribute('value');locks();if(!worker)initWorker();worker.postMessage({type:'load',backend:$('backend').value});}
function run(){
 if(state.busy||state.loading)return;let inputs;try{inputs=inputSnapshot();}catch(e){err(e.message);return;}
 if(!state.ready){state.pending=true;load();return;}
 err();state.busy=true;state.runningInputs=inputs;state.requestId++;locks();$('run-status').textContent='Encoding your inputs…';worker.postMessage({type:'run',requestId:state.requestId,...inputs});
}
function vector(id){return state.result.vectors.find(x=>x.id===id);}
function symbol(v){return v.kind==='text'?'v':v.kind==='change'?'d':'u';}
function picker(){return `<label for="candidate">Compare with</label><select id="candidate">${state.result.scores.map(s=>`<option value="${esc(s.id)}"${s.id===state.candidate?' selected':''}>${esc(s.label)}</option>`).join('')}</select>`;}
function miniSource(v){if(v.kind==='image'){const im=state.result.inputs.gallery.find(x=>x.id===v.imageId);return picture(im)+esc(v.label);}return v.kind==='change'?'Unit image after − unit image before':'“'+esc(v.label)+'”';}
function encoderPath(v){return `<div class="encoder-path"><div class="source">${miniSource(v)}</div><span class="arrow">→</span><div class="encoder-box">${v.kind==='image'?'Image encoder':v.kind==='change'?'Subtract the two unit vectors':'Text encoder'}<small>${v.kind==='image'?'ViT + learned projection':v.kind==='change'?'u_after − u_before':'Transformer + learned projection'}</small></div><span class="arrow">→</span><div class="vector-preview"><strong>${v.kind==='change'?'Δ':'z'} · 512 numbers</strong><br>${short(v.raw)}<div class="subtle">Length ${v.norm.toFixed(5)}${v.tokenCount?' · '+v.tokenCount+' text tokens, including SOT/EOT':''}</div></div></div>`;}
function renderOutputs(){
 const q=vector(state.result.queryId),c=vector(state.candidate);let body=`<p class="result-intro">Each encoder turns its input into 512 numbers. These are the <b>projected outputs</b>, before we make their lengths equal to one.</p><div style="max-width:480px">${picker()}</div>`;
 if(q.kind==='change')body=`<p class="result-intro">First encode both images and normalize each output. Then subtract <b>before from after</b>. The text still goes through its own encoder.</p><div style="max-width:480px">${picker()}</div>`+state.result.vectors.filter(v=>v.kind==='image').map(encoderPath).join('');
 return body+encoderPath(q)+encoderPath(c)+`<p class="callout">The model weights stay fixed. New inputs give new vectors; there is no training here.</p>`;
}
function strip(v){const max=Math.max(...v.map(Math.abs),.001);return `<div class="strip" role="img" aria-label="All ${v.length} vector coordinates; green is positive and red is negative">${v.map((x,i)=>`<i title="Coordinate ${i+1}: ${fmt(x,7)}" style="background:${x>=0?'#14735c':'#aa4c3d'};opacity:${.12+.88*Math.abs(x)/max}"></i>`).join('')}</div>`;}
function coordinateControl(){return `<div class="coordinate-controls"><label for="coordinate">Coordinates <span id="coordinate-label">${state.start+1}–${state.start+8}</span></label><input id="coordinate" type="range" min="0" max="504" step="8" value="${state.start}" aria-label="First coordinate in the eight-coordinate window"></div>`;}
function coordinateTable(q,c,products=false){return `<div class="table-wrap"><table><thead><tr><th>Coordinate</th><th>${symbol(q)}</th><th>${symbol(c)}</th>${products?'<th>Product</th>':''}</tr></thead><tbody>${q.unit.slice(state.start,state.start+8).map((x,j)=>{const i=j+state.start;return `<tr><td>${i+1}</td><td>${fmt(x,6)}</td><td>${fmt(c.unit[i],6)}</td>${products?`<td>${fmt(x*c.unit[i],7)}</td>`:''}</tr>`;}).join('')}</tbody></table></div>`;}
function renderUnits(){const q=vector(state.result.queryId),c=vector(state.candidate);return `<p class="result-intro">Divide every coordinate by the vector’s length. This preserves its direction and gives it length 1.</p><div class="inspector-top"><div><b>Query</b><p>${esc(q.label)}</p></div><div>${picker()}</div></div>${[q,c].map((v,i)=>`<div class="vector-lines"><strong>${i?'Comparison vector':'Query vector'} ${symbol(v)} · ${esc(v.label)}</strong><div class="equation">${v.kind==='change'?'d = Δ / ‖Δ‖ = Δ':'unit vector = z / ‖z‖ = z'} / ${v.norm.toFixed(5)}</div><p class="subtle">First coordinate: ${fmt(v.raw[0],6)} ÷ ${v.norm.toFixed(5)} = ${fmt(v.unit[0],6)} · resulting length ${norm(v.unit).toFixed(6)}</p>${strip(v.unit)}</div>`).join('')}<p class="subtle">All 512 coordinates, in order. Green: positive. Red: negative. Each strip scales its intensity separately. Individual coordinates do not have named meanings.</p>${coordinateControl()}<div id="coordinates">${coordinateTable(q,c)}</div><p class="callout">Now the dot product compares directions. A longer raw output cannot win just by having a larger magnitude.</p>`;}
function renderDotDetails(){const q=vector(state.result.queryId),c=vector(state.candidate),d=contributions(q.unit,c.unit,state.start,8);return `<h3>${esc(c.label)}</h3><p>Multiply corresponding coordinates, then add <b>all 512 products</b>.</p>${coordinateControl()}${coordinateTable(q,c,true)}<div class="sum">These 8 products: ${fmt(d.shown,7)}<br>Other 504 products: ${fmt(d.remaining,7)}<br><b>Dot product: ${fmt(d.total,7)}</b></div><p>Both vectors have length 1, so this dot product is their cosine. Scores range from −1 to +1; they are not probabilities.</p>`;}
function renderShares(){const values=shares(state.result.scores.map(s=>s.cosine),state.tau);return state.result.scores.map((s,i)=>`<div class="share-row"><span>${esc(s.label)}</span><b>${(values[i]*100).toFixed(2)}%</b></div>`).join('');}
function renderScores(){return `<p class="result-intro">${state.result.mode==='search'?'The same text vector is compared with each image vector.':state.result.mode==='difference'?'Compare the unit change direction with each supplied word’s unit vector.':'The same image vector is compared with each description’s vector.'} Select a result to inspect its calculation.</p><div class="score-layout"><div class="score-list">${state.result.scores.map((s,i)=>`<button class="score-row" data-score="${esc(s.id)}" aria-pressed="${state.candidate===s.id}"><span class="score-label">${s.imageId?picture(state.result.inputs.gallery.find(x=>x.id===s.imageId)):''}<span>${esc(s.label)}${i===0?'<small>Highest cosine in this comparison</small>':''}</span></span><b>${fmt(s.cosine,5)}</b></button>`).join('')}</div><div class="score-details" id="dot-details">${renderDotDetails()}</div></div>${state.result.mode!=='difference'?`<details class="temperature"><summary>What happens if we change temperature?</summary><p>These optional shares are softmax(cosine / τ), over <b>this candidate list only</b>. The slider is a teaching control, not the checkpoint’s learned scale. It changes how concentrated the shares are; it does not change the vectors or their ranking.</p><label for="temperature">Temperature τ = <span id="tau-label">${state.tau.toFixed(2)}</span></label><input id="temperature" type="range" min="0.01" max="1" step="0.01" value="${state.tau}"><div id="shares">${renderShares()}</div><p>A large share is not a calibrated probability that the description is true. An incomplete candidate list still has a winner.</p></details>`:''}`;}
function renderResults(){if(!state.result)return;$('results').hidden=false;$('result-runtime').textContent=`Live browser run · ${state.result.device==='webgpu'?'WebGPU':'CPU / WASM'} · ${state.result.dtype} · 512 dimensions`;
 document.querySelectorAll('[data-stage]').forEach(n=>n.setAttribute('aria-pressed',Number(n.dataset.stage)===state.stage));$('result-body').innerHTML=[renderOutputs,renderUnits,renderScores][state.stage]();$('previous-step').disabled=state.stage===0;$('next-step').disabled=state.stage===2;$('step-note').textContent=['First inspect the encoder outputs.','Next add the coordinate-by-coordinate products.','Change one input and try again.'][state.stage];}
function download(){if(!state.result)return;const clean={...state.result,inputs:{...state.result.inputs,images:state.result.inputs.images.map(cleanImage),gallery:state.result.inputs.gallery.map(cleanImage)},inputEditsPending:state.stale,note:'All vectors are live model outputs; cosine scores use normalized 512D vectors. Uploaded image bytes are not included.'};const url=URL.createObjectURL(new Blob([JSON.stringify(clean,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='clip-vectors-'+state.result.computedAt.replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function cleanImage(im){return {...im,url:im.url.startsWith('blob:')?'local upload; image not exported':im.url};}
$('load').onclick=load;$('run').onclick=run;$('export').onclick=download;$('question-select').onchange=e=>applyQuestion(e.target.value);$('descriptions').oninput=dirty;
$('swap').onclick=()=>{[state.image,state.second]=[state.second,state.image];dirty();renderInputs();};
$('previous-step').onclick=()=>{state.stage=Math.max(0,state.stage-1);renderResults();};$('next-step').onclick=()=>{state.stage=Math.min(2,state.stage+1);renderResults();};
document.addEventListener('click',e=>{
 const b=e.target.closest('button');if(!b)return;
 if(b.dataset.mode){state.mode=b.dataset.mode;state.slot='image';$('question-title').textContent='Try your own comparison';$('question-try').textContent='Choose the inputs, make a prediction, then inspect the vectors.';$('question-notice').textContent='Change one thing at a time so you can trace its effect.';if(state.mode==='search')$('descriptions').value='something you can drink';else if($('descriptions').value.split('\n').length===1)$('descriptions').value='a photo of a cat\na photo of a dog\na photo of a rocket';dirty();renderInputs();}
 if(b.dataset.image){if(state.mode==='search')return;state[state.slot]=b.dataset.image;dirty();renderInputs();}
 if(b.dataset.slot){state.slot=b.dataset.slot;renderInputs();$('gallery-details').open=true;}
 if(b.dataset.stage!==undefined){state.stage=Number(b.dataset.stage);renderResults();}
 if(b.dataset.score){state.candidate=b.dataset.score;renderResults();}
});
$('result-body').addEventListener('change',e=>{if(e.target.id==='candidate'){state.candidate=e.target.value;renderResults();}});
$('result-body').addEventListener('input',e=>{
 if(e.target.id==='temperature'){state.tau=Number(e.target.value);$('tau-label').textContent=state.tau.toFixed(2);$('shares').innerHTML=renderShares();}
 if(e.target.id==='coordinate'){state.start=Number(e.target.value);$('coordinate-label').textContent=`${state.start+1}–${state.start+8}`;const q=vector(state.result.queryId),c=vector(state.candidate);if(state.stage===1)$('coordinates').innerHTML=coordinateTable(q,c);else{const detail=$('dot-details'),d=contributions(q.unit,c.unit,state.start,8);detail.querySelector('.table-wrap').outerHTML=coordinateTable(q,c,true);detail.querySelector('.sum').innerHTML=`These 8 products: ${fmt(d.shown,7)}<br>Other 504 products: ${fmt(d.remaining,7)}<br><b>Dot product: ${fmt(d.total,7)}</b>`;}}
});
$('upload').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(uploads.length>=6)throw Error('This visit already has six uploads. Reload the page to start a fresh gallery.');if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>10*1024*1024)throw Error('Choose a PNG, JPEG or WebP image smaller than 10 MB.');const bmp=await createImageBitmap(file);if(bmp.width*bmp.height>40_000_000){bmp.close();throw Error('Choose an image smaller than 40 megapixels.');}bmp.close();const url=URL.createObjectURL(file);uploads.push(url);const im={id:'upload-'+uploads.length,title:file.name,credit:'Your local image · not uploaded',url};state.gallery.push(im);state[state.slot]=im.id;dirty();renderInputs();}catch(e){err(e.message);}finally{$('upload').value='';}};
addEventListener('pagehide',()=>{worker?.terminate();uploads.forEach(url=>URL.revokeObjectURL(url));});
try{const response=await fetch('../data/gallery.json');if(!response.ok)throw Error('The sample gallery could not be loaded. Refresh to retry.');state.gallery=(await response.json()).map(x=>({...x,url:new URL('../images/'+x.file,location.href).href}));$('question-select').innerHTML=questions.map(q=>`<option value="${q.id}">${esc(q.name)}</option>`).join('');applyQuestion(location.hash.slice(1));$('question-select').addEventListener('change',()=>history.replaceState(null,'','#'+$('question-select').value));}catch(e){err(e.message);$('run').disabled=true;$('load').disabled=true;}
window.clipPlayground={getState:()=>({...state}),setStage:s=>{state.stage=s;renderResults();}};
