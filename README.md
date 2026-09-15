# Name Recall Engineering Lab

[Live interactive demo](https://name-recall-engineering-lab.apon-2369.chatgpt.site) · [Source repository](https://github.com/scorpiosk/name-recall-engineering-lab) · [GPL-3.0 license](LICENSE)

An interactive engineering walkthrough of a **real trained five-block causal transformer**. Open `index.html` for self-contained browser inference, or run the local server to use the prediction API as well.

## Run

```sh
npm run build
npm run dev
```

Open http://127.0.0.1:8766/. No npm dependencies or external model service are required. Building needs Python 3; serving and validation need a modern Node.js with the Fetch API.

## Train and infer side by side

The opening **Train & infer** studio trains an independent causal transformer in a local browser Web Worker. Choose **Edit materials & recipe** to design an experiment:

| Area | Editable controls | What changes |
| --- | --- | --- |
| Training materials | Prompt/response rows, training/evaluation split, JSON or JSONL import | The actual examples and training vocabulary |
| Initialization | Small normal, Xavier normal, He normal, all-zero matrices; standard deviation or gain | Starting values of every weight matrix |
| Learning rule | SGD, momentum, AdamW; momentum coefficient, Adam betas and epsilon | The numerical weight update |
| Objective | Cross entropy, smoothed cross entropy, half squared probability error; smoothing | Loss values and their gradients |
| Regularization and schedule | Decoupled weight decay, gradient clipping, constant/linear/cosine learning rate | Update magnitude over time |
| Transformer | 1–5 blocks, 1/2/4/8 attention heads, GELU or ReLU | Actual architecture and computations |
| Run and decoding | Learning rate, batch size, steps, reset seed; temperature, top-k, top-p, greedy/sampling, output limit and sampling seed | Training run and token selection |

Every control has a plain-language explanation. Initialization and learning-rule panels show the equations. **Apply recipe & reset model** validates the staged recipe, creates its vocabulary and tensors, and clears the previous training run and frozen checkpoint. **Reset weights** repeats the active initializer with the selected seed; normalization scales start at 1 and biases at 0. The deliberate all-zero recipe demonstrates failure to break symmetry.

Materials can contain new words and multi-token responses. The tokenizer retains the 55 base tokens and adds words/punctuation found in **training rows only**. Limits are 160 vocabulary entries, 256 training rows, 128 evaluation rows, 54 prompt tokens and 7 response tokens. The model has 64 sequence positions. Evaluation-only new words map to `<unk>` and produce a warning; exact repeated training prompts are rejected from the evaluation split. This is a small, case-sensitive word/punctuation tokenizer, not a frontier model’s subword tokenizer.

Four starter lessons cover dog-name recall, facts, sentence completion and conflicting labels. You can edit every row and export/import the materials and experiment settings as JSON. A JSONL file accepts one `{ "prompt": "...", "response": "...", "split": "train" }` object per line; `split: "test"` marks evaluation rows. Exported recipes contain data and model/learning-rule settings; run controls and reset seed are shown separately in the studio and recorded in model/log artifacts.

**Train model** performs forward passes, response-token and stop-token loss, backpropagation through every selected block, optional global-norm clipping, and the chosen optimizer’s update. Loss is averaged per response sequence, then across the batch. Decoupled weight decay applies to all tensors with all three optimizers. The default dog lesson uses 44 training prompts, batches of 16, and 600 updates. Its cosine schedule falls from the selected peak learning rate to 10% within each run. **1 step** uses the peak rate. Pausing preserves weights and optimizer state; another run adds updates with a fresh schedule.

The right panel receives the latest checkpoint every five updates and at the end of a run. Its entire inference prompt is editable. A live panel connects the actual gradient to one weight’s before/after values; block bars and the loss chart use measured updates. **Freeze checkpoint** preserves a model copy for inference comparison while training continues. **Load trained example** restores the original five-block dog model, default lesson and recipe. Other engineering tabs and the server API use that original reference checkpoint, as identified in their banner.

The artifact sidebar opens actual dataset pairs, tokenizer IDs, recipe, weights, gradients and optimizer moments, per-step logs, evaluation results, portable checkpoints, inference traces, and KV cache contents. Weight/gradient matrices have clickable heatmaps. Downloads preserve snapshots; browser state is lost on reload. A checkpoint supports inference; exact training continuation also needs optimizer and sampler state.

At seed 1337 with the default dog settings, the measured run answered all 16 held-out wording/order combinations correctly. New lessons can memorize training rows while failing on different wording. The examples and evaluation results demonstrate a tiny teaching model, not broad language ability.

## Explore

- **Playground:** edit the dog's name, swap the cat distractor, remove the dog fact, replay all five blocks, and inspect predictions and attention.
- **Decoding controls:** temperature, top-k, top-p, greedy/sampling, seed, maximum new tokens, and KV-cache reuse. Each control has a plain-language explanation and a concrete example.
- **Model internals:** query/key/value projections, normalization, causal masking, attention, residual additions, and GELU MLP. Inspect each head and token position, plus all 65 weight/bias tensors and their 48,352 scalar values.
- **Training:** the original loss measurements, architecture and optimizer settings, evaluation scope, and a portable checkpoint download.
- **Deployment:** interactive browser/server paths, request-state ownership, release lifecycle, and a working `POST /api/predict` console.

The Play trace button replays a completed numerical forward pass. Timing counters measure inference, not the replay animation. Intermediate vocabulary readouts are diagnostic projections; only block five produces the trained final distribution.

## Actual implementation

`engineering.template.html`, `engineering.css`, and `engineering-ui.js` implement the reference views. `experiment-config.js` validates recipes and implements loss functions; `experiment-ui.js` and its template/CSS provide the editor. `train-engine.js` implements full backpropagation and optimizers, while `studio-worker.js` and `studio-ui.js` connect training to the interface. `dog-engine.js` implements attention and transformer inference. `engineering-core.js` supplies decoding, input validation, timing, cache comparison, and numerical inspection. `build_engineering.py` bundles the same engine and rounded checkpoint into both `index.html` and the Worker at `dist/server/index.js`.

`api-worker.template.js` serves the page, `/api/health`, and `/api/predict`. Each prediction gets a separate model instance and KV cache. The application does not persist or log prompts. Hosted access is managed by Sites. The tiny model runs on CPU; no GPU, database, or external LLM API is used.

Example API body:

```json
{"context":"My dog is named Milo. My cat is named Luna.","settings":{"mode":"sample","temperature":1,"topK":0,"topP":1,"seed":42,"maxTokens":4,"useCache":true}}
```

The API returns the generated tokens, processed probabilities, stop reason, measured timings, and work counters. Full internal tensor traces stay in the browser inspector and are omitted from the API response.

## What is trained

48,352 parameters; five distinct transformer blocks; four heads of width eight; hidden width 32; MLP width 64. The network uses learned token and position embeddings, causal attention, pre-layer normalization (epsilon 1e-5), residual connections, a tanh-approximation GELU MLP, final normalization, and a 55-token vocabulary head.

This is a narrow synthetic recall task. It is useful for teaching transformer mechanics, and is not a frontier-scale or general-purpose language model. Training used 12,000 unique prompts and cross-entropy loss on the name and following `<eos>` token. A deterministic SHA-256 split excluded 1,200 test prompts from training; all were answered correctly. All 24 name tokens appeared during training. Unfamiliar names or wording can fail. The tokenizer is case-sensitive and supports 64 sequence positions.

## Reproduce training

```sh
uv venv .venv
uv pip install --python .venv/bin/python torch==2.14.0 numpy
.venv/bin/python train_dog_model.py
npm run build
npm test
```

`dog-model.pt` is the PyTorch checkpoint. `dog-model.json` contains portable weights, metadata, and reference logits. Browser/server bundles round weights to eight decimal places. The prior PyTorch/JavaScript parity check agreed within 0.0001 per logit.

## Validation

`npm test` checks 96 known-name/order/distractor/absence cases, seeded sampling, 80 decoding combinations, filter normalization, full-prefix/cache equivalence, causal attention calculations, all tensor shapes/counts, unchanged weights, Worker routing, API validation, concurrent request isolation, and HTML/JavaScript source contracts. Results are in `engineering-verification.json`. `verify-training.cjs` additionally validates full-gradient training, finite differences, a complete 600-step lesson, deterministic reset, frozen checkpoint stability, and Worker messages (including reset during training). It writes `training-studio-verification.json`. Run `.venv/bin/python verify-training-pytorch.py` to independently compare every gradient tensor with PyTorch autograd; its report is `training-gradient-verification.json`. `verify-experiments.cjs` checks all three loss gradients, multi-token ReLU backpropagation, four initializers, three optimizer update equations, variable-head inference/cache parity, material validation, and actual learning of two new multi-token responses. It writes `experiment-verification.json`. Targeted local browser checks also exercised material edits, new vocabulary, alternative architecture/initialization/optimizer/loss choices, applying/resetting, and a live training step.

A feature-detected `document.modelContext` tool, `run_name_prediction`, uses the same local Playground action. Its supported-browser WebMCP registration/execution contract has not been verified in this environment; this optional capability is not required for the ordinary interface or API.

The earlier scripted explainer remains in `explorer.html`. The prior five-layer template and browser checks are retained as historical source; `build_dog_demo.py` would replace the current entry point with that older demo, so use `build_engineering.py` for this lab.

## License

Copyright (C) 2026 Sandip Kumar and contributors.

This project is licensed under the GNU General Public License, version 3 only (`GPL-3.0-only`). See [LICENSE](LICENSE) for the full text. The project source, synthetic lesson data, and included model checkpoints are provided under this license. External dependencies retain their own licenses.

## Hosting a fork

The public demo is hosted on Sites. The `.openai/hosting.json` in this repository identifies that original deployment; register your own Site and replace its project ID before deploying a fork. Source repository credentials are never stored in this project. Local browser training and the development server work without Sites credentials.
