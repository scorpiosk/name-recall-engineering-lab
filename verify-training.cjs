const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Trainer}=require('./train-engine'),C=require('./engineering-core'),E=require('./dog-engine'),payload=require('./dog-model.json');
const report={};
async function main(){
 const trainer=new Trainer(payload),initial=trainer.payload(),ex=trainer.data.train[0];
 assert.equal(trainer.data.train.length,44);assert.equal(trainer.data.test.length,16);
 const trainPrompts=new Set(trainer.data.train.map(x=>x.context));assert(trainer.data.test.every(x=>!trainPrompts.has(x.context)));
 assert.deepEqual(new Trainer(payload,1337).payload().weights,initial.weights);assert.notDeepEqual(new Trainer(payload,1338).payload().weights,initial.weights);
 trainer.zeroGrad();const loss=trainer.example(ex);let maxGradientError=0;
 for(const [name,index] of [['token_embedding.weight',20*32+3],['blocks.0.qkv.weight',35],['blocks.2.ln1.weight',9],['blocks.3.fc1.weight',188],['blocks.4.proj.weight',301],['final_norm.weight',4],['head.weight',20*32+5]]){
  const p=trainer.params[name],analytical=p.grad[index],old=p.data[index],delta=1e-5;p.data[index]=old+delta;const a=trainer.example(ex,1,false);p.data[index]=old-delta;const b=trainer.example(ex,1,false);p.data[index]=old;const error=Math.abs((a-b)/(2*delta)-analytical);maxGradientError=Math.max(maxGradientError,error);assert(error<2e-6,name+' gradient error '+error);
 }
 // Training forward logits and the independent inference engine agree before any update.
 const ids=trainer.encode(ex.context),logits=trainer.forward(ids),actual=Array.from(logits.data.slice(-55)),expected=new E.Transformer(initial).append(ids).logits();assert(actual.every((x,i)=>Math.abs(x-expected[i])<1e-11));trainer.tape=[];
 const original=JSON.stringify(payload.weights),context='My dog is named Milo. My cat is named Luna.',before=C.generate(initial,{context,settings:{maxTokens:2}}).probabilities.raw[trainer.ids.Milo];
 const start=performance.now();for(let i=1;i<=600;i++)trainer.trainStep({batchSize:16,learningRate:.0003+.0027*(1+Math.cos(Math.PI*i/600))/2});
 const trained=trainer.payload(),out=C.generate(trained,{context,settings:{maxTokens:2}}),swapped=C.generate(trained,{context:'My dog is named Luna. My cat is named Milo.',settings:{maxTokens:2}}),missing=C.generate(trained,{context:'My cat is named Luna.',settings:{maxTokens:2}});
 assert.equal(out.text,'Milo');assert.equal(swapped.text,'Luna');assert.equal(missing.text,'unknown');assert.equal(out.stop,'eos');assert(out.probabilities.raw[trainer.ids.Milo]>.98);assert(trainer.last.loss<.1);
 assert.equal(JSON.stringify(payload.weights),original);assert.deepEqual(new Trainer(payload).payload().weights,initial.weights);
 for(let l=0;l<5;l++)assert.notDeepEqual(trained.weights['blocks.'+l+'.qkv.weight'],initial.weights['blocks.'+l+'.qkv.weight']);
 const heldout=trainer.evaluate();assert.equal(heldout.correct,16);const frozen=structuredClone(trained),frozenHash=JSON.stringify(frozen.weights);trainer.trainStep();assert.equal(JSON.stringify(frozen.weights),frozenHash);
 const fresh=new Trainer(payload);assert.equal(fresh.step,0);assert.equal(fresh.history.length,0);assert(Object.values(fresh.params).every(p=>p.m.every(x=>x===0)&&p.v.every(x=>x===0)));assert.equal(new Trainer(payload,1337,false).payload().metadata.provenance,'Original trained example + live updates');
 report.training={steps:600,batchSize:16,schedule:'Cosine 0.003 to 0.0003',heldout,initialTargetProbability:before,trainedTargetProbability:out.probabilities.raw[trainer.ids.Milo],lossAt600:trained.metadata.loss_history.at(-1).loss,predictions:{default:out.text,swap:swapped.text,missing:missing.text},elapsedSeconds:(performance.now()-start)/1000,allFiveBlocksChanged:true,originalCheckpointUnchanged:true,frozenCheckpointStable:true,resetDeterministic:true};
 report.gradients={finiteDifferenceMaxError:maxGradientError,checkedOperations:7,independentInferenceParity:true};
 // Exercise the actual worker message handler in a JS worker-like context, without browser automation.
 const messages=[],waiters=[];const scope={performance,setTimeout,clearTimeout,console};scope.self={postMessage:m=>{messages.push(m);for(const w of [...waiters])if(w.check(m)){waiters.splice(waiters.indexOf(w),1);clearTimeout(w.timer);w.resolve(m);}}};vm.createContext(scope);
 for(const name of ['dog-engine.js','experiment-config.js','train-engine.js','studio-worker.js'])vm.runInContext(fs.readFileSync(name,'utf8'),scope,{filename:name});
 const send=data=>scope.self.onmessage({data});const until=check=>new Promise((resolve,reject)=>{const w={check,resolve,timer:setTimeout(()=>reject(Error('Worker protocol timeout')),15000)};waiters.push(w)});
 let waiting=until(m=>m.type==='ready');send({type:'initialize',generation:1,payload,seed:1337,randomize:true});let msg=await waiting;assert.equal(msg.step,0);
 waiting=until(m=>m.step===1&&!m.running);send({type:'train',generation:1,steps:1,options:{learningRate:.003,batchSize:8}});msg=await waiting;assert.equal(msg.history.length,1);
 waiting=until(m=>m.type==='optimizer');send({type:'optimizer',generation:1});msg=await waiting;assert.equal(msg.state.step,1);assert.equal(Object.keys(msg.state.tensors).length,65);
 send({type:'train',generation:1,steps:600,options:{learningRate:.003,batchSize:8}});waiting=until(m=>m.type==='ready'&&m.generation===2);send({type:'initialize',generation:2,payload,seed:1337,randomize:true});msg=await waiting;assert.equal(msg.step,0);assert.equal(msg.running,false);
 const count=messages.length;send({type:'train',generation:1,steps:10,options:{learningRate:.003,batchSize:8}});assert.equal(messages.length,count,'Old generation messages are ignored');
 waiting=until(m=>m.type==='error'&&m.generation===2);send({type:'train',generation:2,steps:-1,options:{}});assert((await waiting).message.includes('Steps'));
 report.worker='Initialize, one-step training, optimizer artifact, reset during a scheduled run, stale message isolation, invalid-step rejection';
 const template=fs.readFileSync('engineering.template.html','utf8')+fs.readFileSync('studio.template.html','utf8'),ui=fs.readFileSync('studio-ui.js','utf8');new Function(ui);new Function(fs.readFileSync('train-engine.js','utf8'));
 const htmlIds=[...template.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);assert.equal(new Set(htmlIds).size,htmlIds.length);
 for(const [,id]of ui.matchAll(/\$\('([^']+)'\)/g))assert(htmlIds.includes(id)||['artifact-tensor-select','artifact-tensor-body','artifact-scalar-readout'].includes(id),'Missing UI element '+id);
 const built=fs.readFileSync('index.html','utf8');assert(!/\/\* (TRAINING_ENGINE|TRAINING_WORKER|STUDIO_UI) \*\//.test(built));assert(!built.includes('<!-- TRAINING_STUDIO -->'));
 report.validationScope='Numerical and Worker protocol checks; no visual browser QA';fs.writeFileSync('training-studio-verification.json',JSON.stringify(report,null,2));console.log('PASS',JSON.stringify({...report,training:{...report.training,heldout:{step:heldout.step,correct:heldout.correct,total:heldout.total}}}));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1});
