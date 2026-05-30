/* ============================================================
   D3PM — Data Layer: states, β-schedule, transition matrices
   ============================================================ */

const D3PM = (() => {
  // --- Configuration ---
  const K = 10;                 // Number of states
  const T = 8;                  // Number of timesteps
  const GRID_ROWS = 4;
  const GRID_COLS = 4;

  // State colors (ordered: red → orange → yellow → green → cyan → blue → violet → pink)
  const STATE_COLORS = [
    '#E74C3C', // 0: Red
    '#E67E22', // 1: Orange
    '#F1C40F', // 2: Yellow
    '#A8E635', // 3: Lime
    '#2ECC71', // 4: Green
    '#1ABC9C', // 5: Cyan
    '#3498DB', // 6: Blue
    '#9B59B6', // 7: Indigo
    '#8E44AD', // 8: Violet
    '#E91E63', // 9: Pink
  ];

  const STATE_NAMES = [
    'Red', 'Orange', 'Yellow', 'Lime', 'Green',
    'Cyan', 'Blue', 'Indigo', 'Violet', 'Pink'
  ];

  // β-schedule: discretized Gaussian schedule
  // T=8, linearly increasing from near-zero to near-uniform
  const BETA = [0.008, 0.025, 0.060, 0.120, 0.210, 0.350, 0.560, 0.850];

  // --- Transition matrix computation (discretized Gaussian) ---
  // Q_t[i][j] = softmax_j( -(i-j)^2 / β_t )
  function buildTransitionMatrices(betas) {
    return betas.map(beta => {
      const Q = [];
      for (let i = 0; i < K; i++) {
        const row = [];
        let sum = 0;
        for (let j = 0; j < K; j++) {
          const w = Math.exp(-((i - j) ** 2) / beta);
          row.push(w);
          sum += w;
        }
        // Normalize to probabilities
        for (let j = 0; j < K; j++) {
          row[j] /= sum;
        }
        Q.push(row);
      }
      return Q;
    });
  }

  // Cumulative matrices: Q̄_t = Q_1 · Q_2 · ... · Q_t
  function buildCumulativeMatrices(Qts) {
    const cum = [];
    let acc = Qts[0];
    cum.push(acc);
    for (let t = 1; t < Qts.length; t++) {
      // Matrix multiply: acc = acc × Q_t
      const next = [];
      for (let i = 0; i < K; i++) {
        const row = [];
        for (let j = 0; j < K; j++) {
          let s = 0;
          for (let k = 0; k < K; k++) {
            s += acc[i][k] * Qts[t][k][j];
          }
          row.push(s);
        }
        next.push(row);
      }
      acc = next;
      cum.push(acc);
    }
    return cum;
  }

  const Q_ONESTEP = buildTransitionMatrices(BETA);
  const Q_CUMULATIVE = buildCumulativeMatrices(Q_ONESTEP);

  // --- Forward path (deterministic, visually interesting) ---
  // Starts at state 0, wanders through the spectrum
  const FORWARD_PATH = [0, 0, 1, 2, 4, 6, 8, 5];

  // --- Reverse path: the model's denoising trajectory ---
  // The "predicted x₀" distribution converges back to state 0
  // At each reverse step r (from T down to 0), the model's belief about x₀
  // becomes more concentrated on the original state.
  // We model this as a softmax with decreasing temperature.
  function computeReverseProbs(step) {
    // step goes from T down to 0
    // At step=T: uniform noise (high temperature)
    // At step=0: concentrated at state 0
    const alpha = step / T; // 0 at T, 1 at 0
    const temperature = Math.max(0.05, 2.0 * (1 - alpha) + 0.05);
    const dist = [];
    let sum = 0;
    for (let i = 0; i < K; i++) {
      const w = Math.exp(-(i * i) / temperature);
      dist.push(w);
      sum += w;
    }
    for (let i = 0; i < K; i++) {
      dist[i] /= sum;
    }
    return dist;
  }

  // --- Entropy computation ---
  function entropy(probs) {
    let H = 0;
    for (const p of probs) {
      if (p > 1e-10) H -= p * Math.log2(p);
    }
    return H;
  }

  // --- Pixel grid initial data (4×4 array of state indices) ---
  function generateGrid() {
    // Seeded-like pattern for reproducibility
    const grid = [
      [0, 2, 5, 7],
      [3, 1, 4, 6],
      [8, 6, 2, 9],
      [4, 7, 1, 3],
    ];
    return grid;
  }

  // Original clean data state
  const X_0 = 0;

  // ---- Narrative: pixel's first-person account ----
  const NARRATIVE = {
    intro: "我是阵列左上角那颗红色像素。此刻世界清晰分明——我知道我是谁。但噪声正在远处酝酿……",
    forward: [
      "我是状态 0，红色。周围的同伴们颜色各异。前向扩散即将开始——我会漂向哪里？",
      "β=0.008，几乎察觉不到的扰动。我依然稳稳地站在红色里，但隐隐有什么在拉扯着我。",
      "β=0.025。我开始有些恍惚了——红色的记忆还在，但橙色的影子开始在边缘浮现。",
      "β=0.060……我变成了橙色，状态 1。这是我第一次真正漂移。",
      "β=0.120。黄色，状态 2。扩散在加速——我能感到自己正沿着色谱滑行，每一步都在远离原点。",
      "β=0.210。绿色，状态 4。我已经跳过了一个状态——跳跃的距离变大了。熵在增长，我在迷失。",
      "β=0.350。蓝色，状态 6。最初的红色已如隔世。我能去往任何方向——分布越来越均匀了。",
      "β=0.560。紫色，状态 8。几乎完全被噪声吞噬。我几乎均匀地分布在所有 10 种颜色上。",
      "β=0.850。青色，状态 5。这里是终点——纯噪声的国度。我已认不出自己。",
    ],
    reverse: [
      "我回来了。我是红色的，状态 0。阵列中的每一个像素都重新变得分明。扩散与去噪的轮回完成了。",
      "β 微小到几乎为零。尘埃落定——我就是红色，状态 0。",
      "红色！我几乎就要回到红色了！周边的世界也开始恢复色彩。",
      "第一个橙色的回音——状态 1。几乎就要触及那个最初的自我了。",
      "黄色，状态 2。我能感到噪声正在被剥离。熵在下降，确定性在回归。",
      "绿色，状态 4。模型的信心在增长。那条通往红色的路径越来越清晰。",
      "蓝色，状态 6。我在往回走！每一步都离最初的红色更近一些。",
      "紫色中浮现了某种倾向——模型正在收紧它的预测。概率分布开始向红色聚拢。",
      "混沌之中，模型开始尝试回忆我本来的样子。它说我可能来自……红色？",
    ],
    complete: "重建完成。我从噪声中被找回，变回了最初的自己。这就是 D3PM——离散去噪扩散概率模型。",
  };

  return {
    K, T, X_0, GRID_ROWS, GRID_COLS,
    STATE_COLORS, STATE_NAMES,
    BETA, Q_ONESTEP, Q_CUMULATIVE,
    FORWARD_PATH,
    NARRATIVE,
    computeReverseProbs,
    entropy,
    generateGrid,
  };
})();
