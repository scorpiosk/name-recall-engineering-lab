// Run with Node and Playwright installed. CHROME_PATH can select a local Chrome binary.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, reducedMotion: 'reduce' });
    const errors = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
    await page.goto(pathToFileURL(path.join(__dirname, 'explorer.html')).href);
    const read = () => page.evaluate(() => frontierDemo.inspect());
    const weights = (await read()).weightsFingerprint;
    async function complete() {
      let count = 0;
      while (!(await read()).done && count++ < 150) await page.locator('#step').click();
      const s = await read();
      assert(s.done, 'Scenario must complete');
      assert(s.cacheLengths.every(n => n === s.tokens.length), 'All layers cache every processed token');
      assert(s.tokens.length <= 512, 'Demo fits the declared context budget');
      assert.equal(s.weightsFingerprint, weights, 'Inference leaves weights fixed');
      s.attention.forEach(a => assert(Math.abs(a.reduce((s, x) => s + x, 0) - 1) < 1e-10));
      return s;
    }
    // Full-batch causal prefill must equal appending tokens individually.
    const numerical = await page.evaluate(() => {
      const { ToyTransformer } = frontierDemo;
      const tokens = ['My', 'dog', 'is', 'named', 'Milo', '.'];
      const batch = new ToyTransformer(), incremental = new ToyTransformer();
      batch.append(tokens); tokens.forEach(t => incremental.append([t]));
      let maxError = 0;
      batch.cache.forEach((c, l) => ['k', 'v'].forEach(key => c[key].forEach((v, i) => v.forEach((n, d) => { maxError = Math.max(maxError, Math.abs(n - incremental.cache[l][key][i][d])); }))));
      return maxError;
    });
    assert(numerical < 1e-10, 'Incremental cache matches full causal prefill');
    await page.locator('#step').click();
    await page.locator('[data-token="4"]').click();
    assert.match(await page.locator('#token-info').textContent(), /Position 4/);
    for (let i = 0; i < 5; i++) await page.locator('#step').click();
    assert.deepEqual((await read()).cacheLengths, Array(5).fill((await read()).tokens.length));
    await page.locator('[data-layer="2"]').click();
    assert.equal(await page.locator('#attention-layer').textContent(), 'Layer 3');
    await page.screenshot({ path: path.join(__dirname, 'preview-transformer.png'), fullPage: true });
    const briefing = await complete();
    assert.match(briefing.output, /QA/);
    assert.match(briefing.output, /Thursday/);
    await page.locator('#reset').click();
    assert.deepEqual((await read()).cacheLengths, [0, 0, 0, 0, 0]);
    await page.locator('#tools-toggle').uncheck();
    assert.match((await complete()).output, /need access/);
    await page.locator('#scenario').selectOption('recall');
    assert.match((await complete()).output, /Milo/);
    await page.locator('#history-toggle').uncheck();
    assert.doesNotMatch((await complete()).output, /Milo/);
    await page.locator('#scenario').selectOption('calculate');
    await page.locator('#tools-toggle').check();
    assert.match((await complete()).output, /^408/);
    await page.locator('[data-view="memory"]').click();
    for (let i = 0; i < 5; i++) {
      await page.locator(`[data-memory="${i}"]`).click();
      assert(await page.locator('#memory-detail h3').textContent());
    }
    await page.locator('#life-end').click();
    assert.equal((await read()).life, 'ended');
    assert.match(await page.locator('#memory-detail').textContent(), /records remain/);
    await page.locator('[data-memory="2"]').click();
    assert.match(await page.locator('#memory-detail').textContent(), /cache is released/);
    await page.locator('#life-restore').click();
    await page.locator('#life-next').click();
    assert.equal((await read()).life, 'next');
    await page.locator('[data-view="sampling"]').click();
    await page.locator('#temperature').fill('0');
    assert.deepEqual((await read()).distribution, [1, 0, 0, 0]);
    await page.locator('#sample-many').click();
    assert.match(await page.locator('#sample-result').textContent(), /Milo: 20/);
    await page.locator('#temperature').fill('2');
    const p = (await read()).distribution;
    assert(p[0] < .6 && p[3] > .05);
    assert(Math.abs(p.reduce((s, n) => s + n, 0) - 1) < 1e-10);
    await page.locator('#sample-once').click();
    assert.equal(await page.locator('.sample-chip').count(), 1);
    await page.locator('[data-view="inference"]').click();
    await page.locator('#reset').click();
    await page.locator('#speed').selectOption('350');
    await page.locator('#play').click();
    await page.waitForFunction(() => frontierDemo.inspect().stage > 0);
    await page.locator('#play').click();
    const pausedStage = (await read()).stage;
    await page.waitForTimeout(450);
    assert.equal((await read()).stage, pausedStage);
    await page.locator('#play').click();
    await page.locator('[data-view="memory"]').click();
    assert.equal((await read()).playing, false, 'Switching views pauses playback');
    // Check every screen for horizontal overflow at phone and tablet sizes.
    for (const width of [320, 390, 768, 1024]) {
      await page.setViewportSize({ width, height: 950 });
      for (const tab of ['inference', 'memory', 'sampling', 'notes']) {
        await page.locator(`[data-view="${tab}"]`).click();
        const size = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, width: innerWidth }));
        assert(size.scroll <= size.width, `${tab} overflows at ${width}px (${size.scroll}px)`);
      }
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('[data-view="inference"]').click();
    await page.screenshot({ path: path.join(__dirname, 'preview-mobile.png'), fullPage: true });
    assert.deepEqual(errors, [], 'No browser runtime errors');
    assert.deepEqual(requests, [], 'The demo makes no external requests');
    console.log('PASS: all scenarios; cache equivalence; fixed weights; context toggles; layer/token inspection; memory lifecycle; sampling; playback; responsive layouts; offline operation.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
