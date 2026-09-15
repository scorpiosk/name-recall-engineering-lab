"""Check the browser trainer's full backward pass against independent PyTorch autograd."""
import json
import subprocess
from pathlib import Path
import torch
from train_dog_model import DogTransformer

ROOT = Path(__file__).resolve().parent
script = """
const {Trainer}=require('./train-engine'),payload=require('./dog-model.json');
const t=new Trainer(payload),example=t.data.train[0];t.zeroGrad();const loss=t.example(example);
process.stdout.write(JSON.stringify({payload:t.payload(),ids:[...t.encode(example.context),t.ids[example.target]],targets:[t.ids[example.target],t.ids['<eos>']],loss,gradients:Object.fromEntries(Object.entries(t.params).map(([name,p])=>[name,Array.from(p.grad)]))}));
"""
p = json.loads(subprocess.check_output(['node', '-e', script], cwd=ROOT, text=True))
model = DogTransformer().double()
model.load_state_dict({k: torch.tensor(v, dtype=torch.double) for k, v in p['payload']['weights'].items()})
logits = model(torch.tensor([p['ids']]))[0, -2:]
loss = torch.nn.functional.cross_entropy(logits, torch.tensor(p['targets']))
loss.backward()
errors = {k: float((v.grad.flatten() - torch.tensor(p['gradients'][k], dtype=torch.double)).abs().max()) for k, v in model.named_parameters()}
report = {'lossError': abs(loss.item() - p['loss']), 'maximumGradientError': max(errors.values()), 'tensorsCompared': len(errors), 'reference': 'PyTorch autograd, float64', 'gradientErrorsByTensor': errors}
assert report['lossError'] < 1e-10
assert report['maximumGradientError'] < 1e-8
(ROOT / 'training-gradient-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'gradientErrorsByTensor'}))
