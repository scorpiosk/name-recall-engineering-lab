# SPDX-License-Identifier: GPL-3.0-only
"""Build the self-contained engineering lab and Worker from the same checkpoint."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
payload = json.loads((ROOT / 'dog-model.json').read_text())


def compact(v):
    if isinstance(v, float):
        return round(v, 8)
    if isinstance(v, list):
        return [compact(x) for x in v]
    if isinstance(v, dict):
        return {k: compact(x) for k, x in v.items()}
    return v


model_code = 'const MODEL = ' + json.dumps(compact(payload), separators=(',', ':')) + ';'
engine = (ROOT / 'dog-engine.js').read_text()
core = (ROOT / 'engineering-core.js').read_text()
ui = (ROOT / 'engineering-ui.js').read_text()
training_engine = (ROOT / 'train-engine.js').read_text()
experiment_config = (ROOT / 'experiment-config.js').read_text()
training_worker = engine + '\n' + experiment_config + '\n' + training_engine + '\n' + (ROOT / 'studio-worker.js').read_text()
worker_source = 'const TRAINING_WORKER_SOURCE = ' + json.dumps(training_worker) + ';'
html = (ROOT / 'engineering.template.html').read_text().replace('<!-- TRAINING_STUDIO -->', (ROOT / 'studio.template.html').read_text() + '\n' + (ROOT / 'experiment.template.html').read_text())
for marker, source in [('ENGINEERING_CSS', (ROOT / 'engineering.css').read_text() + '\n' + (ROOT / 'studio.css').read_text() + '\n' + (ROOT / 'experiment.css').read_text()), ('MODEL_PAYLOAD', model_code), ('MODEL_ENGINE', engine), ('ENGINEERING_CORE', core), ('ENGINEERING_UI', ui), ('EXPERIMENT_CONFIG', experiment_config), ('TRAINING_ENGINE', training_engine), ('TRAINING_WORKER', worker_source), ('EXPERIMENT_UI', (ROOT / 'experiment-ui.js').read_text()), ('STUDIO_UI', (ROOT / 'studio-ui.js').read_text())]:
    html = html.replace('/* ' + marker + ' */', source)
(ROOT / 'index.html').write_text(html)
worker = (ROOT / 'api-worker.template.js').read_text()
worker = worker.replace('/* ENGINE_BUNDLE */', engine + '\n' + core)
worker = worker.replace('/* PAYLOAD_BUNDLE */', model_code)
worker = worker.replace('/* PAGE_BUNDLE */', 'const PAGE = ' + json.dumps(html) + ';')
server = ROOT / 'dist/server'
server.mkdir(parents=True, exist_ok=True)
(server / 'index.js').write_text(worker)
(server / 'package.json').write_text('{"type":"module"}\n')
print(f'Built browser page: {len(html.encode()):,} bytes; Worker: {len(worker.encode()):,} bytes.')
