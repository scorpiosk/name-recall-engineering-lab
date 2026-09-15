import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { stat } from 'node:fs/promises';

const port = Number(process.env.PORT || 8766);
const workerUrl = new URL('./dist/server/index.js', import.meta.url);
let worker, modified;
async function handler() {
  const stamp = (await stat(fileURLToPath(workerUrl))).mtimeMs;
  if (!worker || stamp !== modified) {
    worker = (await import(workerUrl.href + '?version=' + stamp)).default;
    modified = stamp;
  }
  return worker;
}
const server = http.createServer(async (req, res) => {
  try {
    let body = Buffer.alloc(0);
    for await (const chunk of req) {
      body = Buffer.concat([body, chunk]);
      if (body.length > 12000) { res.writeHead(413); res.end('Request too large'); return; }
    }
    const request = new Request(`http://127.0.0.1:${port}${req.url}`, { method: req.method, headers: req.headers, ...(!['GET', 'HEAD'].includes(req.method) ? { body } : {}) });
    const result = await (await handler()).fetch(request);
    res.writeHead(result.status, Object.fromEntries(result.headers));
    res.end(Buffer.from(await result.arrayBuffer()));
  } catch (error) { res.writeHead(500, { 'content-type': 'text/plain' }); res.end('Server error: ' + error.message); }
});
server.listen(port, '127.0.0.1', () => console.log(`Local: http://127.0.0.1:${port}/`));
