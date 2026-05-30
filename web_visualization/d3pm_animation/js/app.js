/* ============================================================
   D3PM — Main Application Controller
   ============================================================ */

class D3PMApp {
  constructor() {
    this.phase = 'intro';
    this.currentStep = 0;
    this.direction = 'forward';
    this.isPlaying = false;
    this.speed = 1;
    this.loop = false;
    this.transitioning = false;

    this.pixelGrid = null;
    this.stateGraph = null;
    this.spectrometer = null;
    this.timeline = null;
    this.refs = {};
  }

  // ---- Initialization ----

  init() {
    this._cacheRefs();
    this._initComponents();
    this._initControls();
    this._initBackground();
    this._startIntro();
  }

  _cacheRefs() {
    const ids = [
      'phase-badge', 't-display', 't-total',
      'pixel-grid', 'pixel-coord', 'pixel-state-id',
      'state-graph', 'center-state-dot', 'center-state-label',
      'info-t', 'info-beta', 'info-state-dot', 'info-state-id',
      'info-next-state', 'info-entropy', 'direction-arrow', 'direction-label',
      'spectrometer', 'timeline-track',
      'btn-play', 'btn-reset', 'btn-skip-back', 'btn-skip-fwd',
      'btn-loop', 'btn-toggle-mode',
      'scene-overlay', 'bg-canvas', 'narrative-text',
    ];
    for (const id of ids) {
      this.refs[id] = document.getElementById(id);
    }
  }

  _initComponents() {
    const gridData = D3PM.generateGrid();
    this.pixelGrid = new PixelGrid('pixel-grid');
    this.pixelGrid.init(gridData).highlight(0, 0);

    this.stateGraph = new StateGraph('state-graph');
    this.spectrometer = new Spectrometer('spectrometer').init();
    this.timeline = new Timeline('timeline-track');
    this.timeline.init(D3PM.T, (step) => this._jumpTo(step));
  }

  _initControls() {
    this.refs['btn-play'].addEventListener('click', () => this.togglePlay());
    this.refs['btn-reset'].addEventListener('click', () => this.reset());
    this.refs['btn-skip-back'].addEventListener('click', () => this._jumpTo(0));
    this.refs['btn-skip-fwd'].addEventListener('click', () => this._jumpTo(D3PM.T));
    this.refs['btn-loop'].addEventListener('click', () => {
      this.loop = !this.loop;
      this.refs['btn-loop'].style.color = this.loop ? 'var(--accent-cyan)' : '';
    });
    this.refs['btn-toggle-mode'].addEventListener('click', () => this._toggleDirection());

    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.speed = parseFloat(btn.dataset.speed);
      });
    });

    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      switch (e.key) {
        case ' ': e.preventDefault(); this.togglePlay(); break;
        case 'ArrowRight': e.preventDefault(); this.stepForward(); break;
        case 'ArrowLeft': e.preventDefault(); this.stepBackward(); break;
        case 'r': case 'R': this.reset(); break;
      }
    });
  }

  _initBackground() {
    const canvas = this.refs['bg-canvas'];
    const ctx = canvas.getContext('2d');
    let w, h;

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const count = Math.min(50, Math.floor(w * h / 30000));
    const particles = Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      r: Math.random() * 1.8 + 0.4,
      a: Math.random() * 0.25 + 0.04,
      speed: 0.3 + Math.random() * 0.5,
    }));

    let phase = 0;

    function draw() {
      ctx.clearRect(0, 0, w, h);

      // Ambient glow — warm light
      const cx = w * 0.5 + Math.sin(phase * 0.0003) * w * 0.1;
      const cy = h * 0.35 + Math.cos(phase * 0.0004) * h * 0.05;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, w * 0.55);
      grad.addColorStop(0, 'rgba(235, 215, 185, 0.30)');
      grad.addColorStop(0.4, 'rgba(225, 210, 195, 0.10)');
      grad.addColorStop(1, 'rgba(245, 242, 236, 0)');

      // Second glow — cool complement
      const cx2 = w * 0.7 + Math.cos(phase * 0.0002) * w * 0.05;
      const cy2 = h * 0.7 + Math.sin(phase * 0.0003) * h * 0.05;
      const grad2 = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, w * 0.35);
      grad2.addColorStop(0, 'rgba(195, 205, 230, 0.12)');
      grad2.addColorStop(1, 'rgba(245, 242, 236, 0)');

      // Floating particles — darker on light bg
      for (const p of particles) {
        p.x += p.vx * p.speed;
        p.y += p.vy * p.speed;
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10;
        if (p.y > h + 10) p.y = -10;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(80, 80, 130, ${p.a})`;
        ctx.fill();
      }

      phase++;
      requestAnimationFrame(draw);
    }
    draw();
  }

  // ---- State helpers ----

  /** Get the noisy state at time step t (t = 0..T) */
  _stateAt(t) {
    if (t === 0) return D3PM.X_0;
    return D3PM.FORWARD_PATH[t - 1];
  }

  /** 
   * Forward-looking transition distribution.
   * At step t, shows Q_{t+1}[x_t] — probability distribution 
   * over the next state given current state x_t.
   * At t=T (final), shows the stationary uniform distribution.
   */
  _forwardProbs(t) {
    if (t >= D3PM.T) {
      const p = new Array(D3PM.K).fill(0);
      p.fill(1 / D3PM.K);
      return p;
    }
    const currState = this._stateAt(t);
    return D3PM.Q_ONESTEP[t][currState];
  }

  // ---- Scenes ----

  async _startIntro() {
    this.phase = 'intro';
    this.refs['scene-overlay'].classList.remove('active');

    // Show clean data (one-hot at original state)
    const probs = new Array(D3PM.K).fill(0);
    probs[D3PM.X_0] = 1;
    const s = this._stateAt(0);
    this.stateGraph.update(s, probs, false);
    this.spectrometer.update(probs, s);
    this.spectrometer.animateIn(400);
    this.pixelGrid.animateIn(200);
    this.timeline.setActive(0, false);
    this._updateInfo(0, probs, false, s);
    this._updateNarrative(D3PM.NARRATIVE.intro);

    await this._delay(2200);
    if (this.phase === 'intro') {
      this.startForwardSequence();
    }
  }

  async startForwardSequence() {
    if (this.phase === 'complete') { this.reset(); return; }
    this.phase = 'forward';
    this.direction = 'forward';
    this.currentStep = 0;
    this.isPlaying = true;
    this._updateUIState();

    this.pixelGrid.dimOthers(true);
    this.refs['pixel-coord'].textContent = '聚焦 (0,0)';
    await this._delay(600);

    for (let t = 0; t <= D3PM.T; t++) {
      if (this.phase !== 'forward' || this.direction !== 'forward') break;
      await this._renderForward(t);
      if (t < D3PM.T) {
        // Animate pixel transition to next state
        this.pixelGrid.setCellState(0, 0, this._stateAt(t + 1));
        if (this._stateAt(t + 1) !== this._stateAt(t)) {
          this.pixelGrid.burst(0, 0, D3PM.STATE_COLORS[this._stateAt(t + 1)]);
        }
        await this._delay(1500 / this.speed);
      }
    }

    if (this.phase === 'forward') {
      await this._delay(1800);
      if (this.phase === 'forward') {
        this.startReverseSequence();
      }
    }
  }

  async startReverseSequence() {
    this.phase = 'reverse';
    this.direction = 'reverse';
    this.isPlaying = true;
    this._updateUIState();
    this._flashOverlay();

    for (let t = D3PM.T; t >= 0; t--) {
      if (this.phase !== 'reverse' || this.direction !== 'reverse') break;
      await this._renderReverse(t);
      if (t > 0) {
        this.pixelGrid.setCellState(0, 0, this._stateAt(t - 1));
        if (this._stateAt(t - 1) !== this._stateAt(t)) {
          this.pixelGrid.burst(0, 0, D3PM.STATE_COLORS[this._stateAt(t - 1)]);
        }
        await this._delay(1500 / this.speed);
      }
    }

    if (this.phase === 'reverse') {
      this._onComplete();
    }
  }

  async _renderForward(t) {
    this.currentStep = t;
    this.transitioning = true;

    const state = this._stateAt(t);
    const probs = this._forwardProbs(t);

    this.stateGraph.update(state, probs, false);
    this.spectrometer.update(probs, state);
    this.stateGraph.setCenterState(state);
    this.pixelGrid.setCellState(0, 0, state);
    this.timeline.setActive(t, false);
    this._updateInfo(t, probs, false, state);
    this._updateNarrative(D3PM.NARRATIVE.forward[t]);

    this.transitioning = false;
  }

  async _renderReverse(t) {
    this.currentStep = t;
    this.transitioning = true;

    const state = this._stateAt(t);
    const probs = D3PM.computeReverseProbs(t);

    this.stateGraph.update(state, probs, true);
    this.spectrometer.update(probs, -1);
    this.stateGraph.setCenterState(state);
    this.pixelGrid.setCellState(0, 0, state);
    this.timeline.setActive(t, true);
    this._updateInfo(t, probs, true, state);
    this._updateNarrative(D3PM.NARRATIVE.reverse[t]);

    this.transitioning = false;
  }

  _onComplete() {
    this.phase = 'complete';
    this.isPlaying = false;
    this.refs['phase-badge'].textContent = 'COMPLETE';
    this.refs['phase-badge'].className = 'phase-badge';

    this.pixelGrid.dimOthers(false);
    this.pixelGrid.highlight(0, 0);

    const orig = D3PM.X_0;
    const fp = new Array(D3PM.K).fill(0);
    fp[orig] = 1;
    this.stateGraph.update(orig, fp, false);
    this.spectrometer.update(fp, orig);
    this.timeline.setActive(0, false);
    this._updateInfo(0, fp, false, orig);
    this.stateGraph.setCenterState(orig);
    this.pixelGrid.setCellState(0, 0, orig);
    this.pixelGrid.burst(0, 0, D3PM.STATE_COLORS[orig]);
    this._updateNarrative(D3PM.NARRATIVE.complete);
    this._updatePlayButton();

    if (this.loop) {
      this._delay(2500).then(() => this.reset());
    }
  }

  // ---- Controls ----

  togglePlay() {
    this.isPlaying ? this.pause() : this.play();
  }

  play() {
    if (this.phase === 'complete') { this.reset(); return; }
    this.isPlaying = true;
    this._updatePlayButton();
    if (this.phase === 'intro') this.startForwardSequence();
    else if (this.phase === 'pause') {
      this.phase = this.direction;
      this._resumeSequence();
    } else this._resumeSequence();
  }

  pause() {
    this.isPlaying = false;
    if (['forward', 'reverse'].includes(this.phase)) this.phase = 'pause';
    this._updatePlayButton();
  }

  reset() {
    this.isPlaying = false;
    this.phase = 'intro';
    this.currentStep = 0;
    this.direction = 'forward';
    this._updatePlayButton();

    const gridData = D3PM.generateGrid();
    this.pixelGrid.init(gridData).highlight(0, 0).dimOthers(false);

    const initProbs = new Array(D3PM.K).fill(0);
    initProbs[D3PM.X_0] = 1;
    this.stateGraph.update(D3PM.X_0, initProbs, false);
    this.spectrometer.update(initProbs, D3PM.X_0);
    this.timeline.setActive(0, false);
    this._updateInfo(0, initProbs, false, D3PM.X_0);
    this.stateGraph.setCenterState(D3PM.X_0);

    const badge = this.refs['phase-badge'];
    badge.textContent = 'FORWARD';
    badge.className = 'phase-badge';

    this._delay(600).then(() => {
      if (!this.isPlaying && this.phase === 'intro') {
        this.startForwardSequence();
      }
    });
  }

  stepForward() {
    if (this.transitioning) return;
    if (this.direction === 'forward' && this.currentStep < D3PM.T) {
      this._renderForward(this.currentStep + 1);
      this.currentStep++;
    } else if (this.direction === 'reverse' && this.currentStep > 0) {
      this._renderReverse(this.currentStep - 1);
      this.currentStep--;
    }
  }

  stepBackward() {
    if (this.transitioning) return;
    if (this.direction === 'forward' && this.currentStep > 0) {
      this._renderForward(this.currentStep - 1);
      this.currentStep--;
    } else if (this.direction === 'reverse' && this.currentStep < D3PM.T) {
      this._renderReverse(this.currentStep + 1);
      this.currentStep++;
    }
  }

  _jumpTo(step) {
    if (this.transitioning) return;
    this.pause();
    this.currentStep = Math.max(0, Math.min(D3PM.T, step));
    if (this.direction === 'forward') this._renderForward(this.currentStep);
    else this._renderReverse(this.currentStep);
  }

  _toggleDirection() {
    if (this.transitioning) return;
    this.pause();
    this.direction = this.direction === 'forward' ? 'reverse' : 'forward';
    this.phase = this.direction;
    this._flashOverlay();
    if (this.direction === 'forward') this._renderForward(this.currentStep);
    else this._renderReverse(this.currentStep);
    this._updateUIState();
  }

  async _resumeSequence() {
    if (this.direction === 'forward') {
      for (let t = this.currentStep; t <= D3PM.T; t++) {
        if (!this.isPlaying || this.phase !== 'forward') break;
        await this._renderForward(t);
        if (t < D3PM.T) {
          this.pixelGrid.setCellState(0, 0, this._stateAt(t + 1));
          await this._delay(1500 / this.speed);
        }
      }
      if (this.isPlaying && this.phase === 'forward') {
        await this._delay(1800);
        if (this.isPlaying && this.phase === 'forward') {
          this.startReverseSequence();
        }
      }
    } else {
      for (let t = this.currentStep; t >= 0; t--) {
        if (!this.isPlaying || this.phase !== 'reverse') break;
        await this._renderReverse(t);
        if (t > 0) {
          this.pixelGrid.setCellState(0, 0, this._stateAt(t - 1));
          await this._delay(1500 / this.speed);
        }
      }
      if (this.isPlaying && this.phase === 'reverse') {
        this._onComplete();
      }
    }
  }

  // ---- UI Updates ----

  _updateUIState() {
    const badge = this.refs['phase-badge'];
    const isFwd = this.direction === 'forward';
    badge.textContent = isFwd ? 'FORWARD' : 'REVERSE';
    badge.className = 'phase-badge';
    badge.classList.add(isFwd ? 'active-forward' : 'active-reverse');
    this.refs['direction-arrow'].classList.toggle('reverse', !isFwd);
    this.refs['direction-label'].textContent = isFwd ? 'Noising' : 'Denoising';
  }

  _updatePlayButton() {
    this.refs['btn-play'].innerHTML = this.isPlaying
      ? '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><rect x="5" y="3" width="3.5" height="14" rx="1" fill="currentColor"/><rect x="11.5" y="3" width="3.5" height="14" rx="1" fill="currentColor"/></svg>'
      : '<svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M6 3.5v13l11-6.5L6 3.5z" fill="currentColor"/></svg>';
  }

  _updateInfo(t, probs, isReverse, state) {
    const beta = t > 0 && t <= D3PM.T ? D3PM.BETA[t - 1] : 0;

    this.refs['t-display'].textContent = t;
    this.refs['info-t'].textContent = t;
    this.refs['info-beta'].textContent = beta.toFixed(3);
    this.refs['info-state-dot'].style.background = D3PM.STATE_COLORS[state];
    this.refs['info-state-id'].textContent = state;
    this.refs['pixel-state-id'].textContent = state;

    if (isReverse) {
      const modeProbs = D3PM.computeReverseProbs(t);
      const maxIdx = modeProbs.indexOf(Math.max(...modeProbs));
      this.refs['info-next-state'].innerHTML =
        `<span class="state-dot-sm" style="background:${D3PM.STATE_COLORS[maxIdx]}"></span>` +
        `<span class="mono">${maxIdx} (pred.)</span>`;
    } else {
      const nextIdx = t < D3PM.T ? this._stateAt(t + 1) : -1;
      this.refs['info-next-state'].innerHTML = nextIdx >= 0
        ? `<span class="state-dot-sm" style="background:${D3PM.STATE_COLORS[nextIdx]}"></span><span class="mono">${nextIdx}</span>`
        : '<span class="mono" style="color:var(--text-dim)">—</span>';
    }

    this.refs['info-entropy'].textContent = D3PM.entropy(probs).toFixed(3);
  }

  /** Update the narrative text with a cross-fade transition */
  _updateNarrative(text) {
    const el = this.refs['narrative-text'];
    if (!el) return;
    if (el.textContent === text) return;
    el.classList.add('fading');
    setTimeout(() => {
      el.textContent = text;
      el.classList.remove('fading');
    }, 200);
  }

  _flashOverlay() {
    const overlay = this.refs['scene-overlay'];
    overlay.classList.add('active');
    setTimeout(() => overlay.classList.remove('active'), 400);
  }

  _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms / this.speed));
  }
}

// ---- Bootstrap ----
document.addEventListener('DOMContentLoaded', () => {
  const app = new D3PMApp();
  app.init();
  window.__D3PM_APP = app;
});
