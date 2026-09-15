/* Runs locally in a dedicated browser Worker so training never locks the page. */
let trainer=null,running=false,generation=0,remaining=0,options={},evaluation=null,loopVersion=0,plannedSteps=0,completedSteps=0;
function snapshot(type='snapshot'){
 self.postMessage({type,generation,running,remaining,step:trainer.step,last:trainer.last,history:trainer.history,payload:trainer.payload(),evaluation});
}
function advance(version){
 if(!running||version!==loopVersion)return;
 try{
  completedSteps++;const ratio=plannedSteps>1?(0.1+0.9*(1+Math.cos(Math.PI*completedSteps/plannedSteps))/2):1;trainer.trainStep({...options,learningRate:options.learningRate*ratio});remaining--;
  if(trainer.step%25===0||remaining===0)evaluation=trainer.evaluate();
  if(remaining<=0)running=false;
  if(trainer.step%5===0||!running)snapshot();
  if(running)setTimeout(()=>advance(version),0);
 }catch(error){running=false;self.postMessage({type:'error',generation,message:error.message});snapshot();}
}
self.onmessage=({data:m})=>{
 try{
  if(m.type==='initialize'){
   generation=m.generation;running=false;loopVersion++;remaining=0;
   trainer=new TrainingEngine.Trainer(m.payload,m.seed,m.randomize);evaluation=trainer.evaluate();snapshot('ready');return;
  }
  if(m.generation!==generation||!trainer)return;
  if(m.type==='pause'){running=false;loopVersion++;evaluation=trainer.evaluate();snapshot();}
  else if(m.type==='train'){
   if(running)return;
   if(!Number.isInteger(m.steps)||m.steps<1||m.steps>1000)throw Error('Steps must be from 1 to 1000.');
   options=m.options;remaining=m.steps;plannedSteps=m.steps;completedSteps=0;running=true;const version=++loopVersion;snapshot();setTimeout(()=>advance(version),0);
  }else if(m.type==='optimizer')self.postMessage({type:'optimizer',generation,state:trainer.optimizer()});
  else if(m.type==='evaluate'){evaluation=trainer.evaluate();snapshot();}
 }catch(error){self.postMessage({type:'error',generation,message:error.message});}
};
