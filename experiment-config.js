// SPDX-License-Identifier: GPL-3.0-only
(function(root){
'use strict';
const E=typeof module!=='undefined'&&module.exports?require('./dog-engine.js'):root.DogModelEngine;
const DEFAULT={initialization:'normal',std:.02,gain:1,layers:5,heads:4,activation:'gelu',optimizer:'adamw',loss:'cross_entropy',smoothing:.1,momentum:.9,beta1:.9,beta2:.999,optimizerEpsilon:1e-8,weightDecay:.01,gradientClip:1,schedule:'cosine'};
const SPECIAL=['<pad>','<bos>','<eos>','<answer>','<unk>'];
function settings(input={}){
 const s={...DEFAULT,...input};
 for(const [k,values] of Object.entries({initialization:['normal','xavier','he','zeros'],layers:[1,2,3,4,5],heads:[1,2,4,8],activation:['gelu','relu'],optimizer:['adamw','sgd','momentum'],loss:['cross_entropy','smoothed_ce','brier'],schedule:['cosine','constant','linear']}))if(!values.includes(s[k]))throw Error('Choose a valid '+k+'.');
 for(const [k,min,max] of [['std',.0001,.5],['gain',.1,3],['smoothing',0,.5],['momentum',0,.99],['beta1',0,.99],['beta2',0,.9999],['optimizerEpsilon',1e-10,.001],['weightDecay',0,.2],['gradientClip',0,10]])if(typeof s[k]!=='number'||!Number.isFinite(s[k])||s[k]<min||s[k]>max)throw Error(k+' must be between '+min+' and '+max+'.');
 return s;
}
function normalizeData(data,baseVocab){
 if(!data||!Array.isArray(data.train)||!Array.isArray(data.test)||data.train.length<1||data.train.length>256||data.test.length>128)throw Error('Provide 1–256 training rows and 0–128 evaluation rows.');
 const clean=(row,i,split)=>{
  if(!row||typeof row.prompt!=='string'||typeof row.response!=='string'||!row.prompt.trim()||!row.response.trim())throw Error(split+' row '+(i+1)+' needs a prompt and response.');
  const prompt=row.prompt.trim(),response=row.response.trim(),a=E.tokenize(prompt),b=E.tokenize(response);
  if(prompt.length>2000||response.length>300||a.length>54||b.length>7)throw Error(split+' row '+(i+1)+': use at most 54 prompt tokens and 7 response tokens.');
  if([...a,...b].some(x=>SPECIAL.includes(x)))throw Error(split+' row '+(i+1)+': boundary tokens are added automatically; remove special tokens.');
  return{prompt,response};
 };
 const train=data.train.map((r,i)=>clean(r,i,'Training')),test=data.test.map((r,i)=>clean(r,i,'Evaluation'));
 const vocab=[...baseVocab],set=new Set(vocab);for(const r of train)for(const t of E.tokenize(r.prompt+' '+r.response))if(!set.has(t)){set.add(t);vocab.push(t);}
 if(vocab.length>160)throw Error('This browser lesson supports at most 160 vocabulary tokens. Reduce the number of unique words.');
 const trainKeys=new Set(train.map(r=>r.prompt));if(test.some(r=>trainKeys.has(r.prompt)))throw Error('Evaluation prompts must be separate from training prompts.');
 const warnings=[],labels=new Map();for(const r of train){if(labels.has(r.prompt)&&labels.get(r.prompt)!==r.response)warnings.push('Conflicting responses for the same prompt: the model must divide its probability.');labels.set(r.prompt,r.response);}
 const unseen=[...new Set(test.flatMap(r=>E.tokenize(r.prompt+' '+r.response)).filter(x=>!set.has(x)))];if(unseen.length)warnings.push('Evaluation-only tokens map to <unk>: '+unseen.join(', '));
 return{name:String(data.name||'Custom lesson').slice(0,80),train,test,vocab,names:[],warnings:[...new Set(warnings)],vocabularySize:vocab.length,split:'Training rows update weights. Evaluation rows only measure predictions; their new words do not expand the vocabulary.'};
}
function preset(name,legacyLesson){
 if(name==='dog')return{name:'Dog-name recall',train:legacyLesson.train.map(r=>({prompt:r.context+" What is my dog's name?",response:r.target})),test:legacyLesson.test.map(r=>({prompt:r.context+" What is my dog's name?",response:r.target}))};
 if(name==='facts')return{name:'Tiny facts',train:[{prompt:'The capital of France is',response:'Paris'},{prompt:'The capital of Japan is',response:'Tokyo'},{prompt:'The capital of Italy is',response:'Rome'},{prompt:'France has a capital named',response:'Paris'},{prompt:'Japan has a capital named',response:'Tokyo'},{prompt:'Italy has a capital named',response:'Rome'}],test:[{prompt:'What is the capital of France?',response:'Paris'},{prompt:'What is the capital of Japan?',response:'Tokyo'}]};
 if(name==='completion')return{name:'Sentence completion',train:[{prompt:'The red fox',response:'runs through the forest.'},{prompt:'The blue bird',response:'flies across the sky.'},{prompt:'The small fish',response:'swims in the water.'},{prompt:'A red fox',response:'runs through the forest.'},{prompt:'A blue bird',response:'flies across the sky.'},{prompt:'A small fish',response:'swims in the water.'}],test:[]};
 if(name==='conflict')return{name:'Conflicting labels',train:[{prompt:'My dog is named',response:'Milo'},{prompt:'My dog is named',response:'Luna'},{prompt:'My dog is named',response:'Max'},{prompt:'My dog is named',response:'Bella'}],test:[]};
 throw Error('Unknown lesson preset.');
}
function createPayload(base,data,input={}){
 const experiment=settings(input),lesson=normalizeData(data,base.metadata.vocab),v=lesson.vocab.length,w={};
 for(const [name,value] of Object.entries(base.weights)){
  const block=name.match(/^blocks\.(\d+)/);if(block&&Number(block[1])>=experiment.layers)continue;
  if(name==='token_embedding.weight'||name==='head.weight')w[name]=Array.from({length:v},(_,i)=>value[i]?[...value[i]]:Array(32).fill(0));else w[name]=Array.isArray(value[0])?value.map(r=>[...r]):[...value];
 }
 const parameters=Object.values(w).reduce((s,x)=>s+x.flat().length,0);
 return{payload:{metadata:{...base.metadata,config:{...base.metadata.config,layers:experiment.layers,heads:experiment.heads,activation:experiment.activation},vocab:lesson.vocab,parameters,experiment},weights:w},data:lesson,experiment};
}
function scheduleRate(peak,progress,schedule){const t=Math.max(0,Math.min(1,progress));return peak*(schedule==='constant'?1:schedule==='linear'?1-.9*t:.1+.9*(1+Math.cos(Math.PI*t))/2);}
function tokenLoss(logits,target,kind='cross_entropy',smoothing=.1){
 const p=E.softmax(logits),v=p.length,y=p.map((_,i)=>(kind==='smoothed_ce'?smoothing/v:0)+(i===target?(kind==='smoothed_ce'?1-smoothing:1):0));
 if(kind==='brier'){const d=p.map((x,i)=>x-y[i]),dot=d.reduce((s,x,i)=>s+x*p[i],0);return{loss:d.reduce((s,x)=>s+.5*x*x,0),gradient:p.map((x,i)=>x*(d[i]-dot)),probabilities:p};}
 const max=Math.max(...logits),logZ=max+Math.log(logits.reduce((s,x)=>s+Math.exp(x-max),0));return{loss:y.reduce((s,x,i)=>s-x*(logits[i]-logZ),0),gradient:p.map((x,i)=>x-y[i]),probabilities:p};
}
const api={DEFAULT,settings,normalizeData,preset,createPayload,scheduleRate,tokenLoss};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ExperimentConfig=api;
})(typeof window==='undefined'?globalThis:window);
