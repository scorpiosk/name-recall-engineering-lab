/* Full-gradient CPU training of the same five-block transformer. No fitted-answer rules. */
(function(root){
'use strict';
const E=typeof module!=='undefined'&&module.exports?require('./dog-engine.js'):root.DogModelEngine;
function rng(seed){let x=seed>>>0;const next=()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296;next.state=()=>x;return next;}
const matrix=(rows,cols)=>({rows,cols,data:new Float64Array(rows*cols),grad:new Float64Array(rows*cols)});
function lesson(vocab){
 const names=['Milo','Luna','Max','Bella'],train=[],test=[];
 for(const dog of names)for(const cat of names)if(dog!==cat){
  for(const word of ['named','called']){
   const a=`My dog is ${word} ${dog}.`,b=`My cat is named ${cat}.`;
   const holdOrder=(names.indexOf(dog)+names.indexOf(cat))%2;
   for(let order=0;order<2;order++){const ex={context:order?b+' '+a:a+' '+b,target:dog};(word==='called'&&order===holdOrder?test:train).push(ex);}
  }
 }
 for(const name of names){train.push({context:`My dog is named ${name}.`,target:name},{context:`My cat is named ${name}.`,target:'unknown'});test.push({context:`My cat is called ${name}.`,target:'unknown'});}
 return {name:'Four-name classroom lesson',names,train,test,split:'44 training prompts; 16 held-out combinations of wording and order. All four names and sentence lengths occur in training.',vocabularySize:vocab.length};
}
class Trainer{
 constructor(payload,seed=1337,initialize=true){
  this.base=payload;this.origin=initialize?'random':'pretrained';this.config=payload.metadata.config;this.vocab=payload.metadata.vocab;this.ids=Object.fromEntries(this.vocab.map((v,i)=>[v,i]));this.params={};this.seed=seed>>>0;this.random=rng(this.seed^0x9e3779b9);this.step=0;this.history=[];this.data=lesson(this.vocab);this.last=null;
  const random=rng(seed),normal=()=>Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random())*.02;
  for(const [name,value] of Object.entries(payload.weights)){
   const isMatrix=Array.isArray(value[0]),p=matrix(isMatrix?value.length:1,isMatrix?value[0].length:value.length);p.shape=isMatrix?[p.rows,p.cols]:[p.cols];p.data.set(value.flat());p.m=new Float64Array(p.data.length);p.v=new Float64Array(p.data.length);
   if(initialize)for(let i=0;i<p.data.length;i++)p.data[i]=name.endsWith('bias')?0:((name.includes('.ln')||name.startsWith('final_norm'))?1:normal());
   this.params[name]=p;
  }
 }
 node(rows,cols,back){const n=matrix(rows,cols);if(back)this.tape.push(()=>back(n));return n;}
 embed(ids){const d=this.config.width,t=this.params['token_embedding.weight'],p=this.params['position_embedding.weight'];const n=this.node(ids.length,d,o=>{for(let r=0;r<ids.length;r++)for(let j=0;j<d;j++){t.grad[ids[r]*d+j]+=o.grad[r*d+j];p.grad[r*d+j]+=o.grad[r*d+j];}});for(let r=0;r<ids.length;r++)for(let j=0;j<d;j++)n.data[r*d+j]=t.data[ids[r]*d+j]+p.data[r*d+j];return n;}
 linear(x,name){const w=this.params[name+'.weight'],b=this.params[name+'.bias'],di=x.cols,doo=w.rows;const out=this.node(x.rows,doo,n=>{for(let r=0;r<x.rows;r++)for(let k=0;k<doo;k++){const g=n.grad[r*doo+k];if(b)b.grad[k]+=g;for(let j=0;j<di;j++){x.grad[r*di+j]+=g*w.data[k*di+j];w.grad[k*di+j]+=g*x.data[r*di+j];}}});for(let r=0;r<x.rows;r++)for(let k=0;k<doo;k++){let v=b?b.data[k]:0;for(let j=0;j<di;j++)v+=x.data[r*di+j]*w.data[k*di+j];out.data[r*doo+k]=v;}return out;}
 norm(x,name){const w=this.params[name+'.weight'],b=this.params[name+'.bias'],d=x.cols,z=new Float64Array(x.data.length),inv=new Float64Array(x.rows);const out=this.node(x.rows,d,n=>{for(let r=0;r<x.rows;r++){let s=0,sz=0;for(let j=0;j<d;j++){const i=r*d+j,g=n.grad[i];w.grad[j]+=g*z[i];b.grad[j]+=g;s+=g*w.data[j];sz+=g*w.data[j]*z[i];}for(let j=0;j<d;j++){const i=r*d+j;x.grad[i]+=inv[r]*(n.grad[i]*w.data[j]-s/d-z[i]*sz/d);}}});for(let r=0;r<x.rows;r++){let mean=0,variance=0;for(let j=0;j<d;j++)mean+=x.data[r*d+j]/d;for(let j=0;j<d;j++)variance+=(x.data[r*d+j]-mean)**2/d;inv[r]=1/Math.sqrt(variance+this.config.epsilon);for(let j=0;j<d;j++){const i=r*d+j;z[i]=(x.data[i]-mean)*inv[r];out.data[i]=z[i]*w.data[j]+b.data[j];}}return out;}
 add(a,b){const n=this.node(a.rows,a.cols,o=>{for(let i=0;i<o.data.length;i++){a.grad[i]+=o.grad[i];b.grad[i]+=o.grad[i];}});for(let i=0;i<n.data.length;i++)n.data[i]=a.data[i]+b.data[i];return n;}
 gelu(x){const c=Math.sqrt(2/Math.PI);const n=this.node(x.rows,x.cols,o=>{for(let i=0;i<x.data.length;i++){const v=x.data[i],t=Math.tanh(c*(v+.044715*v*v*v));x.grad[i]+=o.grad[i]*(.5*(1+t)+.5*v*(1-t*t)*c*(1+3*.044715*v*v));}});for(let i=0;i<n.data.length;i++){const v=x.data[i];n.data[i]=.5*v*(1+Math.tanh(c*(v+.044715*v*v*v)));}return n;}
 attention(qkv){const d=this.config.width,h=this.config.heads,hd=d/h,t=qkv.rows,scale=1/Math.sqrt(hd),a=new Float64Array(h*t*t);const out=this.node(t,d,n=>{const da=new Float64Array(t);for(let head=0;head<h;head++)for(let r=0;r<t;r++){let avg=0;for(let k=0;k<=r;k++){let g=0;for(let j=0;j<hd;j++){const z=head*hd+j;g+=n.grad[r*d+z]*qkv.data[k*3*d+2*d+z];qkv.grad[k*3*d+2*d+z]+=a[(head*t+r)*t+k]*n.grad[r*d+z];}da[k]=g;avg+=g*a[(head*t+r)*t+k];}for(let k=0;k<=r;k++){const g=a[(head*t+r)*t+k]*(da[k]-avg)*scale;for(let j=0;j<hd;j++){const z=head*hd+j;qkv.grad[r*3*d+z]+=g*qkv.data[k*3*d+d+z];qkv.grad[k*3*d+d+z]+=g*qkv.data[r*3*d+z];}}}});for(let head=0;head<h;head++)for(let r=0;r<t;r++){let max=-Infinity,total=0;for(let k=0;k<=r;k++){let dot=0;for(let j=0;j<hd;j++){const z=head*hd+j;dot+=qkv.data[r*3*d+z]*qkv.data[k*3*d+d+z];}a[(head*t+r)*t+k]=dot*scale;max=Math.max(max,dot*scale);}for(let k=0;k<=r;k++){const i=(head*t+r)*t+k;a[i]=Math.exp(a[i]-max);total+=a[i];}for(let k=0;k<=r;k++){const i=(head*t+r)*t+k;a[i]/=total;for(let j=0;j<hd;j++){const z=head*hd+j;out.data[r*d+z]+=a[i]*qkv.data[k*3*d+2*d+z];}}}return out;}
 encode(context){return [this.ids['<bos>'],...E.tokenize(context.trim()+" What is my dog's name?").map(x=>this.ids[x]??this.ids['<unk>']),this.ids['<answer>']];}
 forward(ids){this.tape=[];let x=this.embed(ids);for(let l=0;l<5;l++){const p='blocks.'+l+'.';x=this.add(x,this.linear(this.attention(this.linear(this.norm(x,p+'ln1'),p+'qkv')),p+'proj'));x=this.add(x,this.linear(this.gelu(this.linear(this.norm(x,p+'ln2'),p+'fc1')),p+'fc2'));}return this.linear(this.norm(x,'final_norm'),'head');}
 example(example,scale=1,backward=true){const ids=this.encode(example.context),target=this.ids[example.target];if(target===undefined)throw Error('Target outside vocabulary.');const logits=this.forward([...ids,target]),targets=[target,this.ids['<eos>']];let loss=0;for(let n=0;n<2;n++){const pos=ids.length-1+n,start=pos*this.vocab.length,raw=Array.from(logits.data.slice(start,start+this.vocab.length)),p=E.softmax(raw);loss-=Math.log(Math.max(p[targets[n]],1e-300))/2;for(let k=0;k<p.length;k++)logits.grad[start+k]=(p[k]-(k===targets[n]?1:0))*scale/2;}
 if(backward)for(let i=this.tape.length-1;i>=0;i--)this.tape[i]();this.tape=[];return loss;}
 zeroGrad(){for(const p of Object.values(this.params))p.grad.fill(0);}
 trainStep({learningRate=.003,batchSize=16}={}){
  if(!Number.isFinite(learningRate)||learningRate<.0001||learningRate>.02||!Number.isInteger(batchSize)||batchSize<1||batchSize>32)throw Error('Invalid learning rate or batch size.');
  const start=performance.now();this.zeroGrad();let loss=0,first;
  for(let i=0;i<batchSize;i++){const ex=this.data.train[Math.floor(this.random()*this.data.train.length)];if(!first)first=ex;loss+=this.example(ex,1/batchSize)/batchSize;}
  let norm2=0;for(const p of Object.values(this.params))for(const g of p.grad)norm2+=g*g;const norm=Math.sqrt(norm2),clip=Math.min(1,1/(norm||1));
  if(!Number.isFinite(loss)||!Number.isFinite(norm))throw Error('Non-finite gradient; reset the model and lower the learning rate.');
  const step=this.step+1,b1=.9,b2=.999,updates=Array(5).fill(0),counts=Array(5).fill(0);let delta2=0;
  for(const [name,p] of Object.entries(this.params)){const block=name.match(/^blocks\.(\d+)/);for(let i=0;i<p.data.length;i++){const g=p.grad[i]*clip;p.m[i]=b1*p.m[i]+(1-b1)*g;p.v[i]=b2*p.v[i]+(1-b2)*g*g;const delta=learningRate*((p.m[i]/(1-b1**step))/(Math.sqrt(p.v[i]/(1-b2**step))+1e-8)+.01*p.data[i]);p.data[i]-=delta;delta2+=delta*delta;if(block){updates[Number(block[1])]+=delta*delta;counts[Number(block[1])]++;}}}
  this.step=step;this.last={step,loss,gradientNorm:norm,clipScale:clip,updateNorm:Math.sqrt(delta2),blockUpdates:updates.map((v,i)=>Math.sqrt(v/counts[i])),example:first,batchSize,learningRate,ms:performance.now()-start};this.history.push({...this.last});return this.last;
 }
 payload(){const weights={};for(const [name,p] of Object.entries(this.params))weights[name]=p.shape.length===1?Array.from(p.data):Array.from({length:p.rows},(_,r)=>Array.from(p.data.slice(r*p.cols,(r+1)*p.cols)));return{metadata:{config:{...this.config},vocab:[...this.vocab],names:[...this.data.names],parameters:48352,seed:this.seed,steps:this.step,task:'Live four-name classroom lesson',training_examples:this.data.train.length,loss_history:this.history.map(x=>({step:x.step,loss:x.loss})),provenance:this.origin==='pretrained'?'Original trained example + live updates':this.step?'Live browser training':'Randomly initialized weights'},weights};}
 optimizer(){const state={step:this.step,beta1:.9,beta2:.999,epsilon:1e-8,weightDecay:.01,gradientClip:1,seed:this.seed,samplerState:this.random.state(),gradientValues:'Before global-norm clipping',tensors:{}};for(const [name,p] of Object.entries(this.params))state.tensors[name]={shape:p.shape,firstMoment:Array.from(p.m),secondMoment:Array.from(p.v),gradient:Array.from(p.grad)};return state;}
 evaluate(){const payload=this.payload(),rows=this.data.test.map(ex=>{const m=new E.Transformer(payload).append(this.encode(ex.context)),p=E.softmax(m.logits()),id=p.indexOf(Math.max(...p));return{...ex,prediction:this.vocab[id],correct:this.vocab[id]===ex.target,probability:p[this.ids[ex.target]]};});return{step:this.step,correct:rows.filter(x=>x.correct).length,total:rows.length,rows};}
}
const api={Trainer,lesson,rng};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.TrainingEngine=api;
})(typeof window==='undefined'?globalThis:window);
