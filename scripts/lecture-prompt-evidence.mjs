import fs from 'node:fs/promises';
import * as lib from '@huggingface/transformers';
import {createEngine,dot,MODEL,REVISION} from '../engine.js';
const prompts=['a photo of a Newfoundland','a photo of a Persian cat','a photo of a pug','a photo of a dog','a photo of an animal','dog','a dog'];
const engine=await createEngine(lib,{device:'cpu',dtype:'q8'});
try {
const image=await engine.image({url:new URL('../images/newfoundland_31.jpg',import.meta.url).pathname});
const vectors=await engine.text(prompts);
await fs.writeFile(new URL('../data/lecture-prompts.json',import.meta.url),JSON.stringify({created:new Date().toISOString(),model:MODEL,revision:REVISION,runtime:'Transformers.js 3.8.1 / Node CPU / q8',image:'newfoundland_31.jpg',note:'Fixed image encoding and fixed text batch; all cosines computed from normalized 512D vectors.',image_vector:image,prompts:prompts.map((text,i)=>({text,vector:vectors[i],cosine:dot(image,vectors[i])}))},null,2));
console.log(prompts.map((t,i)=>[t,dot(image,vectors[i])]))
} finally {await engine.dispose();}
