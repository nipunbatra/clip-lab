export function norm(v) { return Math.hypot(...v); }
export function unit(v) { const n=norm(v); if(!Number.isFinite(n)||n<1e-8)throw Error('These images produce no stable change direction. Choose a different pair.'); return v.map(x=>x/n); }
export function dot(a,b) { if(a.length!==b.length||!a.length)throw Error('Vector lengths must match.'); return a.reduce((s,x,i)=>s+x*b[i],0); }
export function difference(before,after) { if(before.length!==after.length)throw Error('Vector lengths must match.'); return after.map((x,i)=>x-before[i]); }
export function shares(scores,tau) { if(!(tau>0))throw Error('Temperature must be positive.'); const logits=scores.map(s=>s/tau),m=Math.max(...logits),e=logits.map(s=>Math.exp(s-m)),z=e.reduce((a,b)=>a+b,0); return e.map(x=>x/z); }
export function contributions(a,b,start=0,count=8) { const terms=a.map((x,i)=>x*b[i]),shown=terms.slice(start,start+count).reduce((a,b)=>a+b,0),total=dot(a,b); return {terms,shown,remaining:total-shown,total}; }
