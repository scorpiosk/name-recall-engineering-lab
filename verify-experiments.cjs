// SPDX-License-Identifier: GPL-3.0-only
const assert=require('node:assert/strict'),fs=require('node:fs');
const X=require('./experiment-config'),E=require('./dog-engine'),C=require('./engineering-core'),{Trainer}=require('./train-engine'),base=require('./dog-model.json');
const data={name:'Two invented codes',train:[{prompt:'The code for the moon is',response:'silver lantern'},{prompt:'The code for the sun is',response:'golden compass'}],test:[]};
const make=(recipe={},seed=1337)=>{const r=X.createPayload(base,data,{layers:1,...recipe});return new Trainer(r.payload,seed,true,{experiment:r.experiment,data:r.data});};
const close=(actual,expected,tolerance=1e-10)=>assert(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);
const report={};
let maxLossError=0,maxNetworkError=0;
for(const kind of ['cross_entropy','smoothed_ce','brier']){
 const logits=[-1.7,.3,2.1,-.4],part=X.tokenLoss(logits,1,kind,.2),h=1e-5;
 for(let j=0;j<logits.length;j++){const a=[...logits],b=[...logits];a[j]+=h;b[j]-=h;const error=Math.abs((X.tokenLoss(a,1,kind,.2).loss-X.tokenLoss(b,1,kind,.2).loss)/(2*h)-part.gradient[j]);maxLossError=Math.max(maxLossError,error);assert(error<1e-8);}
 close(part.gradient.reduce((s,x)=>s+x,0),0);assert(Number.isFinite(X.tokenLoss([1000,-1000,0],1,kind).loss));
 // Exercise multi-token teacher forcing and ReLU through attention, norm, MLP and embeddings.
 const t=make({loss:kind,smoothing:.2,activation:'relu',initialization:'xavier',heads:2});t.zeroGrad();t.example(data.train[0]);
 for(const [name,i] of [['token_embedding.weight',t.ids.The*32+3],['position_embedding.weight',8],['blocks.0.qkv.weight',33],['blocks.0.proj.weight',57],['blocks.0.fc1.weight',188],['blocks.0.ln1.weight',9],['head.weight',t.ids.silver*32+5]]){
  const p=t.params[name],old=p.data[i],analytic=p.grad[i];p.data[i]=old+h;const a=t.example(data.train[0],1,false);p.data[i]=old-h;const b=t.example(data.train[0],1,false);p.data[i]=old;const error=Math.abs((a-b)/(2*h)-analytic);maxNetworkError=Math.max(maxNetworkError,error);assert(error<2e-6,name+' '+kind+' gradient mismatch');
 }
}
report.gradients={objectives:3,lossFiniteDifferenceMaxError:maxLossError,multiTokenReluNetworkMaxError:maxNetworkError};
for(const initialization of ['normal','xavier','he','zeros']){
 const t=make({initialization}),same=make({initialization});assert.deepEqual(t.payload().weights,same.payload().weights);
 assert(t.params['final_norm.weight'].data.every(x=>x===1));assert(t.params['blocks.0.fc1.bias'].data.every(x=>x===0));
 if(initialization==='zeros'){const initial=t.example(data.train[0],1,false);for(let i=0;i<3;i++)t.trainStep({batchSize:2});close(t.example(data.train[0],1,false),initial);assert(t.params['head.weight'].data.every(x=>x===0));}
 else assert.notDeepEqual(t.payload().weights,make({initialization},1338).payload().weights);
}
const normal=make(),xavier=make({initialization:'xavier'}),he=make({initialization:'he'});
close(xavier.params['head.weight'].data[0]/normal.params['head.weight'].data[0],Math.sqrt(2/(normal.vocab.length+32))/.02);
close(he.params['head.weight'].data[0]/normal.params['head.weight'].data[0],Math.sqrt(2/32)/.02);
report.initialization='Four deterministic modes; fan scaling, bias/norm defaults, seed changes, and zero-matrix symmetry failure';
for(const optimizer of ['sgd','momentum','adamw']){
 const t=make({optimizer,weightDecay:.07,gradientClip:.01,momentum:.8,beta1:.7,beta2:.8,optimizerEpsilon:1e-6}),lr=.017;
 for(let step=1;step<=2;step++){
  const old=structuredClone(t.payload().weights['head.weight']);const oldMoment=[...t.params['head.weight'].m],oldSecond=[...t.params['head.weight'].v];
  const result=t.trainStep({learningRate:lr,batchSize:2}),p=t.params['head.weight'];assert(result.clipScale<1);
  for(let i=0;i<p.data.length;i++){
   const g=p.grad[i]*result.clipScale,w=old[Math.floor(i/32)][i%32];let direction=g;
   if(optimizer==='momentum')direction=.8*oldMoment[i]+g;
   if(optimizer==='adamw'){const m=(.7*oldMoment[i]+.3*g)/(1-.7**step),v=(.8*oldSecond[i]+.2*g*g)/(1-.8**step);direction=m/(Math.sqrt(v)+1e-6);}
   close(p.data[i],w-lr*direction-lr*.07*w);
  }
  close(result.probe.before-result.probe.after,result.probe.delta);close(result.probe.after,p.data[result.probe.row*32]);
 }
}
report.optimizers='SGD, momentum and AdamW match independent update equations across two clipped updates with decoupled decay';
for(const heads of [1,2,4,8]){
 const t=make({heads,layers:2,activation:'relu'}),ids=t.encode(data.train[0].prompt,true),a=Array.from(t.forward(ids).data.slice(-t.vocab.length)),b=new E.Transformer(t.payload()).append(ids).logits();a.forEach((x,i)=>close(x,b[i]));
 const cached=C.generate(t.payload(),{prompt:data.train[0].prompt,settings:{maxTokens:4,useCache:true}}),full=C.generate(t.payload(),{prompt:data.train[0].prompt,settings:{maxTokens:4,useCache:false}});assert.deepEqual(cached.generated.map(x=>x.id),full.generated.map(x=>x.id));
}
const normalized=X.normalizeData({...data,test:[{prompt:'A comet code is',response:'unseenword'}]},base.metadata.vocab);assert(normalized.vocab.includes('silver'));assert(!normalized.vocab.includes('unseenword'));assert(normalized.warnings.length);
for(const invalid of [{...data,train:[]},{...data,train:[{prompt:'x',response:''}]},{...data,train:[{prompt:'<answer>',response:'x'}]},{...data,test:[data.train[0]]},{...data,train:[{prompt:'a '.repeat(55),response:'b'}]},{...data,train:[{prompt:'x',response:'y '.repeat(8)}]}])assert.throws(()=>X.normalizeData(invalid,base.metadata.vocab));
assert.throws(()=>X.settings({heads:3}));assert.throws(()=>X.settings({beta2:1}));assert.throws(()=>X.settings({loss:'invented'}));
assert(X.normalizeData(X.preset('conflict'),base.metadata.vocab).warnings.length);
for(const schedule of ['cosine','linear','constant']){close(X.scheduleRate(.003,0,schedule),.003);close(X.scheduleRate(.003,1,schedule),schedule==='constant'?.003:.0003);}
report.configuration='Variable depth/head inference parity, train-only vocabulary growth, invalid materials/settings, conflict warnings and schedule endpoints';
const t=make();for(let i=0;i<250;i++)t.trainStep({learningRate:.003,batchSize:4});
const predictions=data.train.map(ex=>C.generate(t.payload(),{prompt:ex.prompt,settings:{maxTokens:8}}).text);assert.deepEqual(predictions,data.train.map(ex=>ex.response));assert(t.last.loss<.01);
report.customLearning={steps:250,batchSize:4,loss:t.last.loss,vocabularySize:t.vocab.length,predictions};
const template=fs.readFileSync('experiment.template.html','utf8'),ui=fs.readFileSync('experiment-ui.js','utf8');new Function(ui);new Function(fs.readFileSync('experiment-config.js','utf8'));
const ids=[...template.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);for(const [,id]of ui.matchAll(/\$\('([^']+)'\)/g))assert(ids.includes(id),'Missing experiment element '+id);
fs.writeFileSync('experiment-verification.json',JSON.stringify(report,null,2));console.log('PASS experiments',JSON.stringify(report));
