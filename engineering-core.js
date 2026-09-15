// SPDX-License-Identifier: GPL-3.0-only
/* Shared by the browser, local server, and deployed Worker. */
(function(root){
'use strict';
const E=typeof module!=='undefined'&&module.exports?require('./dog-engine.js'):root.DogModelEngine;
const DEFAULTS={temperature:1,topK:0,topP:1,maxTokens:4,seed:42,mode:'greedy',useCache:true};
function settings(input={}){
 const s={...DEFAULTS,...input};
 for(const [key,min,max,integer] of [['temperature',0,2,false],['topK',0,55,true],['topP',.05,1,false],['maxTokens',1,8,true],['seed',0,4294967295,true]]){
  if(typeof s[key]!=='number'||!Number.isFinite(s[key])||s[key]<min||s[key]>max||(integer&&!Number.isInteger(s[key])))throw Error('Invalid '+key+': expected '+(integer?'an integer':'a number')+' from '+min+' to '+max+'.');
 }
 if(!['greedy','sample'].includes(s.mode))throw Error('mode must be greedy or sample.');
 if(typeof s.useCache!=='boolean')throw Error('useCache must be a boolean.');
 return s;
}
function probabilities(logits,input={}){
 const s=settings(input),raw=E.softmax(logits,1),temperature=E.softmax(logits,s.temperature),order=temperature.map((p,id)=>({id,p})).sort((a,b)=>b.p-a.p||a.id-b.id);
 const limited=s.topK?order.slice(0,s.topK):order;
 const total=limited.reduce((sum,x)=>sum+x.p,0);
 let mass=0;const kept=[];
 for(const x of limited){if(kept.length&&mass>=s.topP)break;kept.push(x);mass+=x.p/total;}
 const keptMass=kept.reduce((sum,x)=>sum+x.p,0),final=logits.map(()=>0);kept.forEach(x=>final[x.id]=x.p/keptMass);
 return{raw,temperature,final,kept:kept.map(x=>x.id),settings:s};
}
function random(seed){let x=seed>>>0;return()=>{x=(Math.imul(x,1664525)+1013904223)>>>0;return x/4294967296;};}
function select(logits,distribution,s,rng){if(s.mode==='greedy'||s.temperature===0)return logits.indexOf(Math.max(...logits));let n=rng();for(let i=0;i<distribution.length;i++){n-=distribution[i];if(n<0)return i;}return distribution.lastIndexOf(Math.max(...distribution));}
function generate(payload,request={}){
 if(typeof request.context!=='string'||request.context.length>4096)throw Error('context must be text, at most 4096 characters.');
 const s=settings(request.settings),question="What is my dog's name?",prompt=[request.context.trim(),question].filter(Boolean).join(' '),model=new E.Transformer(payload),ids=model.encode(prompt);
 if(ids.length+s.maxTokens>model.config.max_length)throw Error('Input plus max new tokens exceeds the 64-position model limit.');
 const started=performance.now(),events=[],inputTokens=['<bos>',...E.tokenize(prompt),'<answer>'];
 let t=performance.now();model.begin(ids);events.push({name:'Embed tokens',ms:performance.now()-t});
 const readouts=[];
 for(let l=0;l<model.config.layers;l++){t=performance.now();model.forwardLayer();readouts.push(model.project(model.hidden.at(-1)));events.push({name:'Transformer block '+(l+1),ms:performance.now()-t});}
 const initialLogits=model.logits(),prefillMs=performance.now()-started;
 const rng=random(s.seed),generated=[],selectedProbabilities=[],generationSteps=[];
 let running=model.fork(),positionsComputed=ids.length,stop='max_tokens';
 const decodeStart=performance.now();
 for(let i=0;i<s.maxTokens;i++){
  const logits=running.logits(),p=probabilities(logits,s),id=select(logits,p.final,s,rng);
  generated.push(id);selectedProbabilities.push(p.final[id]);generationSteps.push({token:running.vocab[id],id,probability:p.final[id]});
  if(id===running.ids['<eos>']){stop='eos';break;}
  if(i+1<s.maxTokens){if(s.useCache){running.append([id]);positionsComputed++;}else{running=new E.Transformer(payload).append([...ids,...generated]);positionsComputed+=ids.length+generated.length;}}
 }
 const decodeMs=performance.now()-decodeStart;
 const trace={tokens:inputTokens,ids,cache:model.cache,readouts,initialLogits};
 return{model:'name-recall-5l',parameters:payload.metadata.parameters,settings:s,text:generated.filter(id=>payload.metadata.vocab[id]!=='<eos>').map(id=>payload.metadata.vocab[id]).join(' '),generated:generationSteps,stop,probabilities:probabilities(initialLogits,s),timings:{prefillMs,decodeMs,totalMs:performance.now()-started},events,usage:{inputTokens:ids.length,outputTokens:generated.length,positionsComputed,layerPositionEvaluations:positionsComputed*5,cachePositions:running.tokens.length,cacheNumbers:running.tokens.length*5*2*32},trace};
}
function tensorInfo(name,payload){
 const v=payload.weights[name],shape=Array.isArray(v[0])?[v.length,v[0].length]:[v.length],count=shape.reduce((a,b)=>a*b,1),layer=name.match(/^blocks\.(\d+)\./)?.[1];
 let description,operation;
 if(name.startsWith('token_embedding')){description='One learned 32-number vector for each vocabulary token. The input token ID selects a row.';operation='token ID → learned vector';}
 else if(name.startsWith('position_embedding')){description='A learned vector for each of the 64 sequence positions. It is added to the token embedding.';operation='token vector + position vector';}
 else if(name.startsWith('head.')){description='Projects the final normalized hidden vector into 55 vocabulary scores, one score per possible next token.';operation='[32] → [55] logits';}
 else if(name.includes('norm')||name.includes('.ln')){description=name.endsWith('weight')?'Learned scale applied after centering and normalizing a hidden vector. Each dimension has its own scale.':'Learned offset applied after normalization. Each dimension has its own offset.';operation='normalize → scale → shift';}
 else if(name.includes('.qkv.')){description='Produces queries, keys, and values. Its 96 output dimensions are split into three 32-number vectors, then into four attention heads.';operation='[32] → Q[32] + K[32] + V[32]';}
 else if(name.includes('.proj.')){description='Mixes the outputs of the four attention heads back into the 32-dimensional residual stream.';operation='4 heads × 8 values → [32]';}
 else if(name.includes('.fc1.')){description='Expands each token vector from 32 to 64 dimensions before the GELU activation.';operation='[32] → [64] → GELU';}
 else{description='Compresses the MLP representation from 64 dimensions back to 32 so it can be added to the residual stream.';operation='[64] → [32]';}
 if(name.endsWith('.bias')&&!name.includes('norm')&&!name.includes('.ln'))description+=' This tensor is the learned additive bias.';
 return{name,shape,count,layer:layer===undefined?null:Number(layer)+1,description,operation};
}
function blockInternals(payload,trace,layer,position,head=0){
 const w=payload.weights,p=`blocks.${layer}.`,cfg=payload.metadata.config;
 const x=layer?trace.cache[layer-1].hidden[position]:w['token_embedding.weight'][trace.ids[position]].map((v,i)=>v+w['position_embedding.weight'][position][i]);
 const normalized=E.layerNorm(x,w[p+'ln1.weight'],w[p+'ln1.bias'],cfg.epsilon);
 const qkv=E.linear(normalized,w[p+'qkv.weight'],w[p+'qkv.bias']);
 const c=trace.cache[layer],a=c.attention[position][head],query=qkv.slice(head*8,head*8+8),keys=c.k.slice(0,position+1).map(k=>k.slice(head*8,head*8+8));
 const scores=keys.map(k=>k.reduce((s,v,i)=>s+v*query[i],0)/Math.sqrt(8));
 const mixed=[];for(let h=0;h<4;h++)for(let j=0;j<8;j++)mixed.push(c.attention[position][h].reduce((s,v,i)=>s+v*c.v[i][h*8+j],0));
 const projected=E.linear(mixed,w[p+'proj.weight'],w[p+'proj.bias']),residual=x.map((v,i)=>v+projected[i]);
 const normalized2=E.layerNorm(residual,w[p+'ln2.weight'],w[p+'ln2.bias'],cfg.epsilon);
 const ff1=E.linear(normalized2,w[p+'fc1.weight'],w[p+'fc1.bias']);
 const activated=ff1.map(v=>.5*v*(1+Math.tanh(Math.sqrt(2/Math.PI)*(v+.044715*v**3))));
 return{input:x,normalized,qkv,query,key:c.k[position].slice(head*8,head*8+8),value:c.v[position].slice(head*8,head*8+8),scores,attention:a,mixed,projected,residual,normalized2,expanded:ff1,activated,output:c.hidden[position]};
}
const api={DEFAULTS,settings,probabilities,random,generate,tensorInfo,blockInternals};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.EngineeringCore=api;
})(typeof window==='undefined'?globalThis:window);
