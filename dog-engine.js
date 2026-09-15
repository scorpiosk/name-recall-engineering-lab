/* Five-block trained decoder inference. No domain-specific answer logic. */
(function (root) {
  'use strict';
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const linear = (x, weights, bias) => weights.map((row, i) => row.reduce((s, w, j) => s + w * x[j], bias ? bias[i] : 0));
  const softmax = (logits, temperature = 1) => {
    const max = Math.max(...logits);
    if (temperature === 0) {
      const count = logits.filter(x => x === max).length;
      return logits.map(x => x === max ? 1 / count : 0);
    }
    const es = logits.map(x => Math.exp((x - max) / temperature)), total = sum(es);
    return es.map(x => x / total);
  };
  const layerNorm = (x, weight, bias, epsilon) => {
    const mean = sum(x) / x.length;
    const scale = 1 / Math.sqrt(sum(x.map(v => (v - mean) ** 2)) / x.length + epsilon);
    return x.map((v, i) => (v - mean) * scale * weight[i] + bias[i]);
  };
  const gelu = x => .5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + .044715 * x ** 3)));
  const tokenize = text => text.replaceAll('’', "'").match(/<[^>]+>|[A-Za-z]+|[^\sA-Za-z]/g) || [];
  class Transformer {
    constructor(payload) {
      this.payload = payload;
      this.config = payload.metadata.config;
      this.weights = payload.weights;
      this.vocab = payload.metadata.vocab;
      this.ids = Object.fromEntries(this.vocab.map((s, i) => [s, i]));
      this.cache = Array.from({ length: this.config.layers }, () => ({ k: [], v: [], queries: [], attention: [], hidden: [] }));
      this.tokens = [];
      this.hidden = [];
      this.next = 0;
    }
    encode(text) { return [this.ids['<bos>'], ...tokenize(text).map(t => this.ids[t] ?? this.ids['<unk>']), this.ids['<answer>']]; }
    begin(ids) {
      if (this.tokens.length && this.next !== this.config.layers) throw Error('Complete all five blocks before appending tokens.');
      if (this.tokens.length + ids.length > this.config.max_length) throw Error('The model has a 64-token context limit.');
      if (ids.some(id => !Number.isInteger(id) || id < 0 || id >= this.vocab.length)) throw Error('Invalid token ID.');
      const start = this.tokens.length;
      this.hidden = ids.map((id, i) => add(this.weights['token_embedding.weight'][id], this.weights['position_embedding.weight'][start + i]));
      this.tokens.push(...ids);
      this.next = 0;
      return this;
    }
    forwardLayer() {
      if (this.next >= this.config.layers) return null;
      const l = this.next, prefix = `blocks.${l}.`, w = this.weights, c = this.cache[l];
      const d = this.config.width, h = this.config.heads, hd = d / h;
      this.hidden = this.hidden.map(x => {
        const z = layerNorm(x, w[prefix + 'ln1.weight'], w[prefix + 'ln1.bias'], this.config.epsilon);
        const qkv = linear(z, w[prefix + 'qkv.weight'], w[prefix + 'qkv.bias']);
        const q = qkv.slice(0, d), k = qkv.slice(d, 2 * d), v = qkv.slice(2 * d);
        c.k.push(k); c.v.push(v); c.queries.push(q);
        const heads = [], attention = [];
        for (let head = 0; head < h; head++) {
          const offset = head * hd;
          const scores = c.k.map(key => { let dot = 0; for (let j = 0; j < hd; j++) dot += q[offset + j] * key[offset + j]; return dot / Math.sqrt(hd); });
          const a = softmax(scores);
          attention.push(a);
          for (let j = 0; j < hd; j++) heads.push(a.reduce((s, weight, i) => s + weight * c.v[i][offset + j], 0));
        }
        let y = add(x, linear(heads, w[prefix + 'proj.weight'], w[prefix + 'proj.bias']));
        const z2 = layerNorm(y, w[prefix + 'ln2.weight'], w[prefix + 'ln2.bias'], this.config.epsilon);
        const ff = linear(z2, w[prefix + 'fc1.weight'], w[prefix + 'fc1.bias']).map(gelu);
        y = add(y, linear(ff, w[prefix + 'fc2.weight'], w[prefix + 'fc2.bias']));
        c.attention.push(attention); c.hidden.push(y);
        return y;
      });
      this.next++;
      return l;
    }
    append(ids) { this.begin(ids); while (this.next < this.config.layers) this.forwardLayer(); return this; }
    project(vector) {
      const w = this.weights;
      const z = layerNorm(vector, w['final_norm.weight'], w['final_norm.bias'], this.config.epsilon);
      return linear(z, w['head.weight']);
    }
    logits() {
      if (this.next !== this.config.layers) throw Error('The prediction requires all five transformer blocks.');
      return this.project(this.hidden[this.hidden.length - 1]);
    }
    fork() {
      const clone = new Transformer(this.payload);
      clone.cache = JSON.parse(JSON.stringify(this.cache));
      clone.tokens = this.tokens.slice();
      clone.hidden = this.hidden.map(x => x.slice());
      clone.next = this.next;
      return clone;
    }
  }
  const api = { Transformer, softmax, tokenize, linear, layerNorm };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DogModelEngine = api;
})(typeof window === 'undefined' ? globalThis : window);
