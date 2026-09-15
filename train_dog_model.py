"""Train and export a real five-block causal transformer for contextual name recall.

Local synthetic data only. Browser inference has no answer lookup or name extraction.
Run: .venv/bin/python train_dog_model.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SEED = 20260915
CONFIG = dict(layers=5, width=32, heads=4, ff_width=64, max_length=64, epsilon=1e-5)
NAMES = "Milo Luna Max Bella Coco Charlie Daisy Rocky Ruby Leo Toby Nala Oscar Willow Finn Rosie Teddy Pepper Scout Bruno Hazel Ziggy Archie Olive".split()
SPECIAL = ["<pad>", "<bos>", "<eos>", "<answer>", "<unk>"]
QUESTION_TEMPLATES = ["What is my dog's name ?", "My dog's name is", "Tell me my dog's name ."]
DOG_TEMPLATES = ["My dog is named {name} .", "My dog is called {name} .", "The name of my dog is {name} ."]
CAT_TEMPLATES = ["My cat is named {name} .", "My cat is called {name} ."]
NOISE = ["", "It is sunny today .", "We live near the park ."]


def tokenize(text):
    return re.findall(r"<[^>]+>|[A-Za-z]+|[^\sA-Za-z]", text)


VOCAB = SPECIAL + sorted(set(tokenize(" ".join(NAMES + QUESTION_TEMPLATES + [x.format(name="Milo") for x in DOG_TEMPLATES + CAT_TEMPLATES] + NOISE + ["unknown"]))))
IDS = {s: i for i, s in enumerate(VOCAB)}


def split_for(prompt):
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 10 == 0


def make_example(rng):
    name = rng.choice(NAMES)
    cat = rng.choice([n for n in NAMES if n != name])
    missing = rng.random() < .16
    dog = "" if missing else rng.choice(DOG_TEMPLATES).format(name=name)
    other = rng.choice(CAT_TEMPLATES).format(name=cat) if rng.random() < .66 else ""
    facts = [dog, other]
    if rng.random() < .5:
        facts.reverse()
    facts.insert(rng.randrange(3), rng.choice(NOISE))
    context = " ".join(f for f in facts if f)
    prompt = " ".join(x for x in [context, rng.choice(QUESTION_TEMPLATES)] if x)
    return prompt, "unknown" if missing else name


def dataset(size, heldout):
    rng = random.Random(SEED + int(heldout))
    examples = {}
    while len(examples) < size:
        prompt, target = make_example(rng)
        if split_for(prompt) == heldout:
            examples[prompt] = target
    return list(examples.items())


def encode(prompt, answer=None):
    tokens = ["<bos>"] + tokenize(prompt) + ["<answer>"]
    if answer is not None:
        tokens += [answer, "<eos>"]
    return [IDS.get(t, IDS["<unk>"]) for t in tokens]


def batch(examples):
    rows = [encode(p, a) for p, a in examples]
    length = max(map(len, rows)) - 1
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        x[i, :len(row) - 1] = torch.tensor(row[:-1])
        # Teacher forcing: predict the answer token and then the stop token.
        y[i, len(row) - 3:len(row) - 1] = torch.tensor(row[-2:])
    return x, y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        d, f = CONFIG["width"], CONFIG["ff_width"]
        self.ln1 = nn.LayerNorm(d, eps=CONFIG["epsilon"])
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=CONFIG["epsilon"])
        self.fc1 = nn.Linear(d, f)
        self.fc2 = nn.Linear(f, d)

    def forward(self, x):
        b, t, d = x.shape
        h = CONFIG["heads"]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class DogTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["width"]
        self.token_embedding = nn.Embedding(len(VOCAB), d)
        self.position_embedding = nn.Embedding(CONFIG["max_length"], d)
        self.blocks = nn.ModuleList([Block() for _ in range(CONFIG["layers"])])
        self.final_norm = nn.LayerNorm(d, eps=CONFIG["epsilon"])
        self.head = nn.Linear(d, len(VOCAB), bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[1]))
        for block in self.blocks:
            z = block(z)
        return self.head(self.final_norm(z))


@torch.no_grad()
def evaluate(model, examples):
    model.eval()
    correct = 0
    loss = 0.0
    for start in range(0, len(examples), 128):
        rows = examples[start:start + 128]
        x, y = batch(rows)
        logits = model(x)
        for j, (p, a) in enumerate(rows):
            answer_index = len(encode(p)) - 1
            correct += int(logits[j, answer_index].argmax().item() == IDS[a])
        loss += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item() * len(rows)
    model.train()
    return dict(correct=correct, total=len(examples), accuracy=correct / len(examples), loss=loss / len(examples))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    train = dataset(12000, False)
    test = dataset(1200, True)
    model = DogTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    rng = random.Random(SEED + 2)
    history = []
    started = time.time()
    steps = 900
    for step in range(1, steps + 1):
        x, y = batch(rng.choices(train, k=96))
        lr = .0003 + .0027 * (1 + math.cos(math.pi * step / steps)) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = dict(step=step, loss=loss.item(), seconds=round(time.time() - started, 2))
            history.append(row)
            print(json.dumps(row), flush=True)
    heldout = evaluate(model, test)
    counterfactuals = [(f"My dog is named {n} . What is my dog's name ?", n) for n in NAMES]
    counterfactuals += [(f"My dog is named {n} . My cat is named {NAMES[(i + 1) % len(NAMES)]} . What is my dog's name ?", n) for i, n in enumerate(NAMES)]
    counterfactuals += [(f"My cat is named {n} . What is my dog's name ?", "unknown") for n in NAMES]
    checks = evaluate(model, counterfactuals)
    model.eval()
    reference = []
    with torch.no_grad():
        for prompt, target in counterfactuals[:4] + counterfactuals[24:28] + counterfactuals[48:52]:
            ids = encode(prompt)
            logits = model(torch.tensor([ids]))[0, -1]
            reference.append(dict(prompt=prompt, target=target, ids=ids, logits=logits.tolist()))
    meta = dict(config=CONFIG, vocab=VOCAB, names=NAMES, seed=SEED, parameters=sum(p.numel() for p in model.parameters()), training_examples=len(train), steps=steps, batch_size=96, training_seconds=round(time.time() - started, 2), heldout=heldout, counterfactuals=checks, loss_history=history, task="Predict the dog's name, then EOS; unknown if no dog fact is present.", split="Unique prompt strings split by SHA-256 modulo 10; 10% held out. All name tokens are seen in training.", torch_version=torch.__version__)
    checkpoint = {"metadata": meta, "weights": {k: v.detach().tolist() for k, v in model.state_dict().items()}, "reference": reference}
    (ROOT / "dog-model.json").write_text(json.dumps(checkpoint, separators=(",", ":")))
    (ROOT / "training-report.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), ROOT / "dog-model.pt")
    print(json.dumps(dict(heldout=heldout, counterfactuals=checks, parameters=meta["parameters"], seconds=meta["training_seconds"])), flush=True)


if __name__ == "__main__":
    main()
