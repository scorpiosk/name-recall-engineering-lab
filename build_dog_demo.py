# SPDX-License-Identifier: GPL-3.0-only
"""Bundle the trained weights, pure JavaScript inference, and UI into one HTML file."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
payload = json.loads((root / "dog-model.json").read_text())


def compact(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


template = (root / "dog-demo.template.html").read_text()
template = template.replace("/* BASE_STYLES */", (root / "observatory.css").read_text())
result = template.replace("/* MODEL_PAYLOAD */", "const MODEL = " + json.dumps(compact(payload), separators=(",", ":")) + ";")
result = result.replace("/* MODEL_ENGINE */", (root / "dog-engine.js").read_text())
(root / "index.html").write_text(result)
print(f"Built index.html: {len(result.encode()):,} bytes; {payload['metadata']['parameters']:,} trained parameters.")
