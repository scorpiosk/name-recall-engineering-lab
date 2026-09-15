// SPDX-License-Identifier: GPL-3.0-only
let trainer=null,running=false,generation=0,remaining=0,options={},evaluation=null,loopVersion=0,plannedSteps=0,completedSteps=0;
function snapshot(type='snapshot'){self.postMessage({type,generation,running,remaining,step:trainer.step,last:trainer.last,history:trainer.history,payload:trainer.payload(),evaluation,data:trainer.data});}
function advance(version){
 if(!running||version!==loopVersion)return;
 try{completedSteps++;const learningRate=plannedSteps===1?options.learningRate:ExperimentConfig.scheduleRate(options.learningRate,completedSteps/plannedSteps,trainer.experiment.schedule);trainer.trainStep({...options,learningRate});remaining--;if(trainer.step%25===0||remaining===0)evaluation=trainer.evaluate();if(remaining<=0)running=false;if(trainer.step%5===0||!running)snapshot();if(running)setTimeout(()=>advance(version),0);}
 catch(error){running=false;snapshot();self.postMessage({type:'error',generation,message:error.message});}
}
self.onmessage=({data:m})=>{
 try{
  if(m.type==='initialize'){
   const prepared=m.data?ExperimentConfig.createPayload(m.payload,m.data,m.experiment):{payload:m.payload,experiment:ExperimentConfig.settings(m.experiment)},candidate=new TrainingEngine.Trainer(prepared.payload,m.seed,m.randomize,prepared),candidateEvaluation=candidate.evaluate();
   generation=m.generation;running=false;loopVersion++;remaining=0;trainer=candidate;evaluation=candidateEvaluation;snapshot('ready');return;
  }
  if(m.generation!==generation||!trainer)return;
  if(m.type==='pause'){running=false;loopVersion++;evaluation=trainer.evaluate();snapshot();}
  else if(m.type==='train'){if(running)return;if(!Number.isInteger(m.steps)||m.steps<1||m.steps>1000)throw Error('Steps must be from 1 to 1000.');if(!m.options||!Number.isFinite(m.options.learningRate)||m.options.learningRate<.00001||m.options.learningRate>.2||!Number.isInteger(m.options.batchSize)||m.options.batchSize<1||m.options.batchSize>32)throw Error('Use learning rate 0.00001–0.2 and batch size 1–32.');options=m.options;remaining=m.steps;plannedSteps=m.steps;completedSteps=0;running=true;const version=++loopVersion;snapshot();setTimeout(()=>advance(version),0);}
  else if(m.type==='optimizer')self.postMessage({type:'optimizer',generation,state:trainer.optimizer()});
  else if(m.type==='evaluate'){evaluation=trainer.evaluate();snapshot();}
 }catch(error){self.postMessage({type:'error',generation:m.generation,message:error.message});}
};
