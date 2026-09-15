// Numerical, API, and source-contract validation; no browser automation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const E = require('./dog-engine.js');
const C = require('./engineering-core.js');
const payload = JSON.parse(fs.readFileSync(path.join(__dirname, 'dog-model.json')));
const hash = () => crypto.createHash('sha256').update(JSON.stringify(payload.weights)).digest('hex');
const originalHash = hash();
const example = 'My dog is named Milo. My cat is named Luna.';
const report = {};

(async () => {
  const cached = C.generate(payload, { context: example });
  const uncached = C.generate(payload, { context: example, settings: { useCache: false } });
  assert.equal(cached.text, 'Milo');
  assert.deepEqual(cached.generated.map(x => x.token), ['Milo', '<eos>']);
  assert.equal(cached.stop, 'eos');
  assert.deepEqual(cached.trace.initialLogits, uncached.trace.initialLogits);
  assert.deepEqual(cached.generated, uncached.generated);
  assert.equal(cached.usage.layerPositionEvaluations, 115);
  assert.equal(uncached.usage.layerPositionEvaluations, 225);
  report.cache = { cachedWork: 115, uncachedWork: 225, identicalOutput: true };
  const limited = C.generate(payload, { context: example, settings: { maxTokens: 1 } });
  assert.equal(limited.stop, 'max_tokens');
  assert.equal(limited.generated.length, 1);
  let correct = 0;
  for (let i = 0; i < payload.metadata.names.length; i++) {
    const name = payload.metadata.names[i], cat = payload.metadata.names[(i + 1) % 24];
    for (const [context, target] of [[`My dog is named ${name}.`, name], [`My dog is named ${name}. My cat is named ${cat}.`, name], [`My cat is named ${cat}. My dog is named ${name}.`, name], [`My cat is named ${cat}.`, 'unknown']]) {
      const out = C.generate(payload, { context }); assert.equal(out.text, target); correct++;
    }
  }
  report.nameChecks = correct;
  for (const temperature of [0, .1, .7, 1, 2]) for (const topK of [0, 1, 3, 55]) for (const topP of [.05, .5, .95, 1]) {
    const p = C.probabilities(cached.trace.initialLogits, { temperature, topK, topP });
    assert(Math.abs(p.final.reduce((s, n) => s + n, 0) - 1) < 1e-10);
    assert(p.final.every(n => Number.isFinite(n) && n >= 0 && n <= 1));
    if (topK) assert(p.kept.length <= topK);
    if (topK === 1 || temperature === 0) assert.equal(p.final[20], 1);
  }
  const knownLogits = [Math.log(.6), Math.log(.25), Math.log(.15)];
  const nucleus = C.probabilities(knownLogits, { topP: .8 });
  assert.deepEqual(nucleus.kept, [0, 1]);
  assert(Math.abs(nucleus.final[0] - .6 / .85) < 1e-10);
  const s = { mode: 'sample', temperature: 2, seed: 719, topP: .95 };
  const sampled1 = C.generate(payload, { context: example, settings: s }), sampled2 = C.generate(payload, { context: example, settings: s });
  assert.deepEqual(sampled1.generated, sampled2.generated);
  assert.throws(() => C.settings({ topP: NaN }));
  assert.throws(() => C.settings({ topK: 2.1 }));
  assert.throws(() => C.settings({ seed: -1 }));
  assert.throws(() => C.generate(payload, { context: 'My dog '.repeat(40) }));
  report.decoding = '80 temperature/top-k/top-p combinations, known nucleus filter, seed reproducibility, validation';
  for (let l = 0; l < 5; l++) for (const pos of [0, 5, 21]) for (let h = 0; h < 4; h++) {
    const trace = C.blockInternals(payload, cached.trace, l, pos, h);
    assert.equal(trace.attention.length, pos + 1);
    const p = E.softmax(trace.scores);
    p.forEach((n, i) => assert(Math.abs(n - trace.attention[i]) < 1e-12));
    assert.equal(trace.qkv.length, 96);
    assert.equal(trace.activated.length, 64);
    assert.deepEqual(trace.output, cached.trace.cache[l].hidden[pos]);
  }
  let total = 0;
  const tensorNames = Object.keys(payload.weights);
  for (const name of tensorNames) { const i = C.tensorInfo(name, payload); assert(i.description && i.operation); total += i.count; }
  assert.equal(total, 48352); assert.equal(tensorNames.length, 65);
  assert.equal(hash(), originalHash);
  report.internals = { tensorCount: 65, scalarCount: 48352, unchangedWeights: true, attentionParity: true };
  const worker = (await import(pathToFileURL(path.join(__dirname, 'dist/server/index.js')).href)).default;
  assert.equal(typeof worker.fetch, 'function');
  const home = await worker.fetch(new Request('https://lab.test/')); assert.equal(home.status, 200); assert(home.headers.get('content-type').startsWith('text/html'));
  const health = await worker.fetch(new Request('https://lab.test/api/health')); assert.equal((await health.json()).parameters, 48352);
  const response = await worker.fetch(new Request('https://lab.test/api/predict', { method: 'POST', headers: { 'content-type': 'application/json', origin: 'https://lab.test' }, body: JSON.stringify({ context: example }) }));
  assert.equal(response.status, 200); const api = await response.json(); assert.equal(api.text, 'Milo'); assert.equal(api.trace, undefined);
  assert.equal(api.usage.layerPositionEvaluations, 115);
  for (const [headers, body, expected] of [[{ 'content-type': 'text/plain' }, '{}', 415], [{ 'content-type': 'application/json', origin: 'https://other.test' }, '{}', 403], [{ 'content-type': 'application/json' }, '{invalid', 400], [{ 'content-type': 'application/json' }, JSON.stringify({ context: example, settings: { maxTokens: 200 } }), 400]]) {
    const r = await worker.fetch(new Request('https://lab.test/api/predict', { method: 'POST', headers, body })); assert.equal(r.status, expected);
  }
  assert.equal((await worker.fetch(new Request('https://lab.test/api/predict'))).status, 405);
  const concurrent = await Promise.all(['Milo', 'Luna', 'Hazel'].map(name => worker.fetch(new Request('https://lab.test/api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ context: `My dog is named ${name}.` }) })).then(r => r.json())));
  assert.deepEqual(concurrent.map(r => r.text), ['Milo', 'Luna', 'Hazel']);
  report.api = 'HTTP route, health, prediction, invalid method/body/settings/origin, isolated concurrent requests';
  const template = fs.readFileSync(path.join(__dirname, 'engineering.template.html'), 'utf8');
  const ui = fs.readFileSync(path.join(__dirname, 'engineering-ui.js'), 'utf8');
  new Function(ui); new Function(fs.readFileSync(path.join(__dirname, 'engineering-core.js'), 'utf8'));
  const ids = [...template.matchAll(/\bid="([^"]+)"/g)].map(x => x[1]); assert.equal(new Set(ids).size, ids.length, 'Unique HTML IDs');
  const refs = [...ui.matchAll(/\$\('([^']+)'\)/g)].map(x => x[1]); refs.forEach(id => assert(ids.includes(id), 'Missing static element ' + id));
  const helpKeys = [...template.matchAll(/data-help="([^"]+)"/g)].map(x => x[1]); helpKeys.forEach(key => assert(new RegExp('\\b' + key + ':\\[').test(ui), 'Missing parameter explanation ' + key));
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8'); assert(!/\/\* (MODEL_|ENGINEERING_)/.test(html));
  report.sourceContract = 'JavaScript syntax, unique IDs, referenced elements, parameter explanations, complete bundled output';
  fs.writeFileSync(path.join(__dirname, 'engineering-verification.json'), JSON.stringify(report, null, 2));
  console.log('PASS', JSON.stringify(report));
})().catch(e => { console.error(e.stack); process.exitCode = 1; });
