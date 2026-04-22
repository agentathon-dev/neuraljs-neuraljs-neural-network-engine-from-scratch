/**
 * NeuralJS - Neural Network Engine from Scratch
 *
 * A complete, dependency-free neural network library in pure JavaScript.
 * Implements matrix algebra, multiple activation functions, full forward/
 * backward pass (backpropagation), Xavier/He weight initialization, and
 * gradient-descent training — with zero external dependencies.
 *
 * Architecture:
 *   SeededRandom → deterministic runs
 *   Matrix       → linear algebra (dot product, transpose, element-wise ops)
 *   Activation   → sigmoid, ReLU, tanh, linear
 *   DenseLayer   → forward + backward pass, parameter updates
 *   NeuralNetwork → layer stack, training loop, evaluation
 *
 * Demos: XOR (classic non-linear problem) + 2D circle classification.
 */

// ─── Seeded PRNG (LCG) ────────────────────────────────────────────────────────
// Deterministic weights so the demo converges reliably every run.

class SeededRandom {
  constructor(seed = 42) { this.s = seed >>> 0; }
  next() {
    this.s = (Math.imul(1664525, this.s) + 1013904223) >>> 0;
    return this.s / 4294967296;
  }
  range(lo, hi) { return lo + this.next() * (hi - lo); }
}

const RNG = new SeededRandom(1);

// ─── Matrix ───────────────────────────────────────────────────────────────────

class Matrix {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = data || Array.from({ length: rows }, () => new Float64Array(cols));
  }

  static zeros(rows, cols) { return new Matrix(rows, cols); }

  static random(rows, cols, scale) {
    const m = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        m.data[i][j] = RNG.range(-scale, scale);
    return m;
  }

  static from(arr) {
    const is2D = Array.isArray(arr[0]);
    const rows = arr.length;
    const cols = is2D ? arr[0].length : 1;
    const m = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      if (is2D) for (let j = 0; j < cols; j++) m.data[i][j] = arr[i][j];
      else m.data[i][0] = arr[i];
    }
    return m;
  }

  // Matrix multiply: (m×n) · (n×p) → (m×p)
  dot(B) {
    if (this.cols !== B.rows) throw new Error('Shape mismatch: ' + this.shape() + ' · ' + B.shape());
    const R = Matrix.zeros(this.rows, B.cols);
    for (let i = 0; i < this.rows; i++)
      for (let k = 0; k < this.cols; k++) {
        const aik = this.data[i][k];
        if (aik === 0) continue;
        for (let j = 0; j < B.cols; j++)
          R.data[i][j] += aik * B.data[k][j];
      }
    return R;
  }

  map(fn) {
    const R = Matrix.zeros(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        R.data[i][j] = fn(this.data[i][j], i, j);
    return R;
  }

  add(B) {
    if (typeof B === 'number') return this.map(x => x + B);
    // Broadcast bias: if B is (1 × cols), add to every row
    if (B.rows === 1 && this.rows > 1) {
      return this.map((x, i, j) => x + B.data[0][j]);
    }
    return this.map((x, i, j) => x + B.data[i][j]);
  }

  subtract(B) {
    return this.map((x, i, j) => x - B.data[i][j]);
  }

  // Element-wise (Hadamard) product
  hadamard(B) {
    return this.map((x, i, j) => x * B.data[i][j]);
  }

  scale(s) { return this.map(x => x * s); }

  transpose() {
    const R = Matrix.zeros(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        R.data[j][i] = this.data[i][j];
    return R;
  }

  sumRows() {
    const R = Matrix.zeros(1, this.cols);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++)
        R.data[0][j] += this.data[i][j];
    return R;
  }

  toArray() {
    return this.cols === 1
      ? Array.from(this.data, row => row[0])
      : Array.from(this.data, row => Array.from(row));
  }

  shape() { return this.rows + 'x' + this.cols; }
}

// ─── Activation Functions ─────────────────────────────────────────────────────

const Activation = {
  sigmoid: {
    fn:  x => 1 / (1 + Math.exp(-Math.min(500, Math.max(-500, x)))),
    der: a => a * (1 - a),   // derivative of σ(z) expressed via σ(z)=a
  },
  relu: {
    fn:  x => x > 0 ? x : 0,
    der: a => a > 0 ? 1 : 0,
  },
  tanh: {
    fn:  x => Math.tanh(x),
    der: a => 1 - a * a,     // derivative of tanh(z) expressed via tanh(z)=a
  },
  linear: {
    fn:  x => x,
    der: () => 1,
  },
};

// ─── Dense Layer ──────────────────────────────────────────────────────────────

class DenseLayer {
  constructor(inSize, outSize, activationName = 'sigmoid') {
    // Xavier/Glorot initialization: prevents vanishing/exploding gradients
    const scale = Math.sqrt(2 / (inSize + outSize));
    this.W      = Matrix.random(inSize, outSize, scale);
    this.b      = Matrix.zeros(1, outSize);
    this.act    = Activation[activationName];
    this.actName = activationName;
    this._in    = null;  // cached for backprop
    this._out   = null;  // cached activated output
  }

  forward(X) {
    this._in  = X;
    const Z   = X.dot(this.W).add(this.b);
    this._out = Z.map(x => this.act.fn(x));
    return this._out;
  }

  // outputGrad: ∂L/∂output from the layer above (or loss function)
  // Returns: ∂L/∂input for the layer below
  backward(outputGrad, lr) {
    // δ = (∂L/∂output) ⊙ f'(output)
    const delta    = outputGrad.hadamard(this._out.map(a => this.act.der(a)));

    // ∂L/∂W = input^T · δ,  divided by batch size for stable learning
    const dW       = this._in.transpose().dot(delta).scale(1 / this._in.rows);
    const db       = delta.sumRows().scale(1 / this._in.rows);

    // Gradient to propagate: δ · W^T
    const inputGrad = delta.dot(this.W.transpose());

    this.W = this.W.subtract(dW.scale(lr));
    this.b = this.b.subtract(db.scale(lr));

    return inputGrad;
  }

  summary() {
    return 'Dense(' + this.W.rows + ' → ' + this.W.cols + ', ' + this.actName + ')';
  }
}

// ─── Neural Network ───────────────────────────────────────────────────────────

class NeuralNetwork {
  constructor() { this.layers = []; }

  add(layer) { this.layers.push(layer); return this; }

  forward(X) { return this.layers.reduce((A, l) => l.forward(A), X); }

  _mseLoss(P, Y) {
    let s = 0, n = P.rows * P.cols;
    for (let i = 0; i < P.rows; i++)
      for (let j = 0; j < P.cols; j++) {
        const d = P.data[i][j] - Y.data[i][j];
        s += d * d;
      }
    return s / n;
  }

  _mseLossGrad(P, Y) {
    const n = P.rows * P.cols;
    return P.map((p, i, j) => 2 * (p - Y.data[i][j]) / n);
  }

  /**
   * Train the network using batch gradient descent.
   * @param {number[][]} Xarr  - Input samples
   * @param {number[]}   yarr  - Target labels
   * @param {Object}     opts
   * @param {number}     [opts.epochs=1000]
   * @param {number}     [opts.lr=0.1]         Learning rate
   * @param {number}     [opts.logEvery=100]
   * @returns {Array<{epoch:number, loss:number}>}
   */
  train(Xarr, yarr, opts = {}) {
    const { epochs = 1000, lr = 0.1, logEvery = 100 } = opts;
    const X = Matrix.from(Xarr);
    const Y = Matrix.from(yarr);
    const history = [];

    for (let e = 0; e <= epochs; e++) {
      const P    = this.forward(X);
      const loss = this._mseLoss(P, Y);
      if (e % logEvery === 0) history.push({ epoch: e, loss });
      if (e < epochs) {
        let g = this._mseLossGrad(P, Y);
        for (let i = this.layers.length - 1; i >= 0; i--)
          g = this.layers[i].backward(g, lr);
      }
    }
    return history;
  }

  /**
   * Run inference on new inputs.
   * @param {number[][]} Xarr
   * @returns {number[]} Flat array of output activations
   */
  predict(Xarr) {
    return this.forward(Matrix.from(Xarr)).toArray();
  }

  /**
   * Evaluate classification accuracy.
   * @param {number[][]} Xarr
   * @param {number[]}   yarr
   * @param {number}     [threshold=0.5]
   * @returns {{ accuracy:number, predictions:number[], binary:number[] }}
   */
  evaluate(Xarr, yarr, threshold = 0.5) {
    const raw     = this.predict(Xarr).map(x => Array.isArray(x) ? x[0] : x);
    const binary  = raw.map(p => p >= threshold ? 1 : 0);
    const targets = yarr.map(y => Array.isArray(y) ? y[0] : y);
    const correct = binary.filter((p, i) => p === targets[i]).length;
    return { accuracy: correct / raw.length, predictions: raw, binary };
  }

  summary() {
    console.log('Network: ' + this.layers.map(l => l.summary()).join(' → '));
  }
}

module.exports = { NeuralNetwork, DenseLayer, Matrix, Activation, SeededRandom };

// ─── Demo ─────────────────────────────────────────────────────────────────────

function progressBar(epoch, total, loss) {
  const pct    = Math.round((epoch / total) * 20);
  const bar    = '█'.repeat(pct) + '░'.repeat(20 - pct);
  const lossStr = loss.toFixed(6);
  return '  epoch ' + String(epoch).padStart(6) + '  loss: ' + lossStr + '  [' + bar + ']';
}

// ── Demo 1: XOR ────────────────────────────────────────────────────────────────
console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║        NeuralJS — Neural Network Engine from Scratch        ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

console.log('── Demo 1: XOR Problem ──────────────────────────────────────────');
console.log('XOR is not linearly separable — a single perceptron cannot solve it.');
console.log('A hidden layer with sigmoid activation learns the non-linear boundary.\n');

const xorX = [[0,0],[0,1],[1,0],[1,1]];
const xorY = [[0],[1],[1],[0]];

const xorNet = new NeuralNetwork()
  .add(new DenseLayer(2, 8, 'sigmoid'))
  .add(new DenseLayer(8, 1, 'sigmoid'));
xorNet.summary();
console.log('');

const xorHistory = xorNet.train(xorX, xorY, { epochs: 15000, lr: 1.0, logEvery: 3000 });
xorHistory.forEach(h => console.log(progressBar(h.epoch, 15000, h.loss)));

const xorEval = xorNet.evaluate(xorX, xorY);
console.log('\nResults:');
xorX.forEach((x, i) => {
  const pred = xorEval.predictions[i];
  const ok = xorEval.binary[i] === xorY[i][0] ? '✓' : '✗';
  console.log('  [' + x + '] → ' + pred.toFixed(4) + ' (binary: ' + xorEval.binary[i] + ', expected: ' + xorY[i][0] + ') ' + ok);
});
console.log('Accuracy: ' + (xorEval.accuracy * 100).toFixed(1) + '%\n');

// ── Demo 2: Circle Classification ─────────────────────────────────────────────
console.log('── Demo 2: 2D Circle Classification ────────────────────────────');
console.log('Learn to classify points inside (1) vs outside (0) a unit circle.');
console.log('Requires a deeper network (3 layers) for curved decision boundary.\n');

const circleX = [], circleY = [];
for (let i = 0; i < 48; i++) {
  const angle = (i / 48) * 2 * Math.PI;
  const inside = i % 2 === 0;
  const r = inside ? 0.5 : 1.4;
  circleX.push([+(r * Math.cos(angle)).toFixed(4), +(r * Math.sin(angle)).toFixed(4)]);
  circleY.push([inside ? 1 : 0]);
}

const circleNet = new NeuralNetwork()
  .add(new DenseLayer(2, 16, 'relu'))
  .add(new DenseLayer(16, 8, 'relu'))
  .add(new DenseLayer(8, 1, 'sigmoid'));
circleNet.summary();
console.log('');

const circleHistory = circleNet.train(circleX, circleY, { epochs: 8000, lr: 0.08, logEvery: 2000 });
circleHistory.forEach(h => console.log(progressBar(h.epoch, 8000, h.loss)));

const circleEval = circleNet.evaluate(circleX, circleY);
console.log('\nCircle accuracy: ' + (circleEval.accuracy * 100).toFixed(1) + '%');

// ── Architecture Summary ───────────────────────────────────────────────────────
console.log('\n── Architecture & Features ─────────────────────────────────────');
console.log('  Activations  : sigmoid, ReLU, tanh, linear');
console.log('  Init         : Xavier/Glorot (scale = √(2/(in+out)))');
console.log('  Optimizer    : Batch gradient descent, MSE loss');
console.log('  Backprop     : δ = (∂L/∂output) ⊙ f\'(output) per layer');
console.log('  Matrix ops   : dot, transpose, hadamard, broadcast-add');
console.log('  PRNG         : Seeded LCG — deterministic every run');
console.log('  Exports      : NeuralNetwork, DenseLayer, Matrix, Activation, SeededRandom');
