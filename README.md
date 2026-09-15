# Name Recall Engineering Lab

An interactive engineering walkthrough of a **real trained five-block causal transformer**. Open `index.html` for self-contained browser inference, or run the local server to use the prediction API as well.

## Run

```sh
npm run build
npm run dev
```

Open http://127.0.0.1:8766/. No npm dependencies or external model service are required. Building needs Python 3; serving and validation need a modern Node.js with the Fetch API.

## Explore

- **Playground:** edit the dog's name, swap the cat distractor, remove the dog fact, replay all five blocks, and inspect predictions and attention.
- **Decoding controls:** temperature, top-k, top-p, greedy/sampling, seed, maximum new tokens, and KV-cache reuse. Each control has a plain-language explanation and a concrete example.
- **Model internals:** query/key/value projections, normalization, causal masking, attention, residual additions, and GELU MLP. Inspect each head and token position, plus all 65 weight/bias tensors and their 48,352 scalar values.
- **Training:** the original loss measurements, architecture and optimizer settings, evaluation scope, and a portable checkpoint download.
- **Deployment:** interactive browser/server paths, request-state ownership, release lifecycle, and a working `POST /api/predict` console.

The Play trace button replays a completed numerical forward pass. Timing counters measure inference, not the replay animation. Intermediate vocabulary readouts are diagnostic projections; only block five produces the trained final distribution.

## Actual implementation

`engineering.template.html`, `engineering.css`, and `engineering-ui.js` implement the interface. `dog-engine.js` implements attention and transformer inference. `engineering-core.js` supplies decoding, input validation, timing, cache comparison, and numerical inspection. `build_engineering.py` bundles the same engine and rounded checkpoint into both `index.html` and the Worker at `dist/server/index.js`.

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

`npm test` checks 96 known-name/order/distractor/absence cases, seeded sampling, 80 decoding combinations, filter normalization, full-prefix/cache equivalence, causal attention calculations, all tensor shapes/counts, unchanged weights, Worker routing, API validation, concurrent request isolation, and HTML/JavaScript source contracts. Results are in `engineering-verification.json`. These checks do not constitute visual browser QA.

A feature-detected `document.modelContext` tool, `run_name_prediction`, uses the same local Playground action. Its supported-browser WebMCP registration/execution contract has not been verified in this environment; this optional capability is not required for the ordinary interface or API.

The earlier scripted explainer remains in `explorer.html`. The prior five-layer template and browser checks are retained as historical source; `build_dog_demo.py` would replace the current entry point with that older demo, so use `build_engineering.py` for this lab.
