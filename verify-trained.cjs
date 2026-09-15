const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, reducedMotion: 'reduce' });
    const errors = [], externalRequests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) externalRequests.push(r.url()); });
    await page.goto(pathToFileURL(path.join(__dirname, 'index.html')).href);
    const numerical = await page.evaluate(() => {
      const { Transformer, softmax } = dogDemo.engine, payload = dogDemo.payload;
      const before = JSON.stringify(payload.weights);
      let maxLogitError = 0, maxCacheError = 0, maxProbabilitySumError = 0;
      for (const row of payload.reference) {
        const full = new Transformer(payload).append(row.ids);
        const logits = full.logits();
        logits.forEach((v, i) => maxLogitError = Math.max(maxLogitError, Math.abs(v - row.logits[i])));
        const incremental = new Transformer(payload);
        row.ids.forEach(id => incremental.append([id]));
        full.cache.forEach((c, l) => {
          for (const kind of ['k', 'v']) c[kind].forEach((v, i) => v.forEach((n, d) => { maxCacheError = Math.max(maxCacheError, Math.abs(n - incremental.cache[l][kind][i][d])); }));
          c.attention.forEach((heads, pos) => heads.forEach(a => {
            if (a.length !== pos + 1) throw Error('A token can see future positions.');
            maxProbabilitySumError = Math.max(maxProbabilitySumError, Math.abs(a.reduce((s, n) => s + n, 0) - 1));
          }));
        });
      }
      const cases = [];
      payload.metadata.names.forEach((name, i) => {
        const cat = payload.metadata.names[(i + 1) % payload.metadata.names.length];
        cases.push([`My dog is named ${name}. What is my dog's name?`, name]);
        cases.push([`My dog is named ${name}. My cat is named ${cat}. What is my dog's name?`, name]);
        cases.push([`My cat is named ${cat}. My dog is named ${name}. What is my dog's name?`, name]);
        cases.push([`My cat is named ${cat}. What is my dog's name?`, 'unknown']);
      });
      let correct = 0;
      for (const [prompt, target] of cases) {
        const model = new Transformer(payload); model.append(model.encode(prompt));
        const logits = model.logits(), id = logits.indexOf(Math.max(...logits));
        if (model.vocab[id] === target) correct++;
        model.append([id]);
        const next = model.logits();
        if (model.vocab[next.indexOf(Math.max(...next))] !== '<eos>') throw Error('Missing learned EOS.');
      }
      return { maxLogitError, maxCacheError, maxProbabilitySumError, browserChecksCorrect: correct, browserChecksTotal: cases.length, weightsUnchanged: before === JSON.stringify(payload.weights) };
    });
    assert(numerical.maxLogitError < 1e-4, 'Bundled JavaScript logits match PyTorch');
    assert(numerical.maxCacheError < 1e-10, 'Incremental KV cache matches full prefill');
    assert(numerical.maxProbabilitySumError < 1e-10, 'Attention normalizes correctly');
    assert(numerical.weightsUnchanged, 'Inference must never change trained weights');
    assert.equal(numerical.browserChecksCorrect, numerical.browserChecksTotal, 'Known-name context changes work');
    const state = () => page.evaluate(() => dogDemo.inspect());
    await page.locator('#step').click();
    assert.equal((await state()).phase, 1);
    await page.locator('#step').click();
    assert.deepEqual((await state()).cacheLengths, [22, 0, 0, 0, 0]);
    await page.locator('[data-query="4"]').click();
    assert.equal(await page.locator('.prompt-token.masked').count(), 17);
    await page.locator('#head').selectOption('2');
    await page.locator('.kv-values summary').click();
    assert.match(await page.locator('#kv-values').textContent(), /Head 3/);
    await page.locator('[data-query="21"]').click();
    await page.locator('#finish').click();
    let s = await state();
    assert.deepEqual(s.prediction, ['Milo', '<eos>']);
    assert(s.distribution[20] > .98, 'Expected initial Milo probability');
    assert.deepEqual(s.cacheLengths, [23, 23, 23, 23, 23]);
    assert.equal(s.phase, 9);
    await page.locator('#head').selectOption('mean');
    await page.locator('.kv-values summary').click();
    await page.screenshot({ path: path.join(__dirname, 'dog-prediction-desktop.png'), fullPage: true });
    for (const [preset, expected] of [['swap', 'Luna'], ['missing', 'unknown'], ['reverse', 'Milo']]) {
      await page.locator(`[data-preset="${preset}"]`).click();
      assert.deepEqual((await state()).cacheLengths, [0, 0, 0, 0, 0]);
      await page.locator('#finish').click();
      assert.deepEqual((await state()).prediction, [expected, '<eos>']);
    }
    await page.locator('#context').fill('My dog is called Hazel. We live near the park.');
    await page.locator('#finish').click();
    assert.deepEqual((await state()).prediction, ['Hazel', '<eos>']);
    await page.locator('#temperature').fill('0');
    await page.locator('#sample').click();
    assert.deepEqual((await state()).prediction, ['Hazel', '<eos>']);
    await page.locator('#temperature').fill('2');
    assert(Math.abs((await state()).distribution.reduce((s, x) => s + x, 0) - 1) < 1e-10);
    await page.locator('#context').fill('My dog is named Pixel.');
    assert.match(await page.locator('#input-status').textContent(), /<unk>: Pixel/);
    await page.locator('#context').fill('My dog '.repeat(40));
    assert.equal((await state()).valid, false);
    assert(await page.locator('#finish').isDisabled());
    await page.locator('[data-preset="milo"]').click();
    await page.locator('#temperature').fill('1');
    await page.locator('#speed').selectOption('300');
    await page.locator('#play').click();
    await page.waitForFunction(() => dogDemo.inspect().phase > 0);
    await page.locator('#play').click();
    const paused = (await state()).phase;
    await page.waitForTimeout(400);
    assert.equal((await state()).phase, paused);
    await page.locator('#finish').click();
    await page.locator('#training-proof summary').click();
    for (const width of [320, 390, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 950 });
      const size = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert(size.scroll <= size.viewport, `Horizontal overflow at ${width}: ${size.scroll}`);
    }
    await page.locator('#training-proof summary').click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(__dirname, 'dog-prediction-mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#fullscreen').click();
    assert(await page.evaluate(() => !!document.fullscreenElement));
    await page.screenshot({ path: path.join(__dirname, 'dog-prediction-presentation.png'), fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(externalRequests, [], 'The self-contained demo works offline');
    const report = { ...numerical, browserRuntimeErrors: errors.length, externalRequests: externalRequests.length, uiChecks: 'Presets, edited context, masking, head selection, KV display, layer stepping, learned EOS, sampling, context length, playback, mobile layout, full screen' };
    fs.writeFileSync(path.join(__dirname, 'verification-report.json'), JSON.stringify(report, null, 2));
    console.log('PASS', JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
