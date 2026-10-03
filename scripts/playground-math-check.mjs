import assert from 'node:assert/strict';
import {norm,unit,dot,difference,shares,contributions} from '../playground/math.js';
const a=unit([3,4]),b=unit([-4,3]);assert.equal(norm(a),1);assert(Math.abs(dot(a,b))<1e-15);assert.equal(dot(a,a),1);
const forward=unit(difference(a,b)),reverse=unit(difference(b,a));assert(forward.every((x,i)=>Math.abs(x+reverse[i])<1e-15));assert.throws(()=>unit(difference(a,a)));assert.throws(()=>dot(a,[1]));
const x=unit(Array.from({length:512},(_,i)=>Math.sin(i))),y=unit(Array.from({length:512},(_,i)=>Math.cos(i*.3)));
for(let start=0;start<=504;start+=8){const c=contributions(x,y,start,8);assert(Math.abs(c.shown+c.remaining-dot(x,y))<1e-12);assert.equal(c.terms.length,512);}
for(const tau of [.01,.07,1]){const s=shares([.29,.23,-.1],tau);assert(Math.abs(s.reduce((a,b)=>a+b)-1)<1e-12);assert(s[0]>s[1]&&s[1]>s[2]);}assert.throws(()=>shares([1,2],0));
console.log('Checked normalization, zero differences, 512D contribution windows, swap symmetry and temperature shares.');
