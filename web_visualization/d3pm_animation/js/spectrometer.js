/* ============================================================
   D3PM — Probability Spectrometer Component
   ============================================================ */

class Spectrometer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.bars = [];
  }

  init() {
    this.container.innerHTML = '';
    this.bars = [];

    for (let i = 0; i < D3PM.K; i++) {
      const row = document.createElement('div');
      row.className = 'prob-bar-row';

      // State number label
      const label = document.createElement('span');
      label.className = 'prob-bar-label';
      label.textContent = i;
      label.dataset.index = i;
      row.appendChild(label);

      // Track
      const track = document.createElement('div');
      track.className = 'prob-bar-track';

      const fill = document.createElement('div');
      fill.className = 'prob-bar-fill';
      fill.style.background = D3PM.STATE_COLORS[i];
      fill.style.width = '0%';
      fill.dataset.index = i;
      track.appendChild(fill);

      row.appendChild(track);

      // Value
      const val = document.createElement('span');
      val.className = 'prob-bar-val';
      val.textContent = '0.000';
      val.dataset.index = i;
      row.appendChild(val);

      this.container.appendChild(row);
      this.bars.push({ row, label, fill, val });
    }

    return this;
  }

  update(probs, activeState = -1) {
    for (let i = 0; i < D3PM.K; i++) {
      const p = probs[i] || 0;
      const pct = Math.max(0.5, p * 100);
      this.bars[i].fill.style.width = `${pct}%`;
      this.bars[i].val.textContent = p.toFixed(3);

      // Highlight the active state
      this.bars[i].label.classList.toggle('active-state', i === activeState);
      this.bars[i].fill.classList.toggle('active-bar', i === activeState);
      if (i === activeState) {
        this.bars[i].fill.style.color = D3PM.STATE_COLORS[i];
      }
    }
  }

  animateIn(delay = 0) {
    this.bars.forEach((bar, i) => {
      bar.row.style.opacity = '0';
      bar.row.style.transform = 'translateX(-8px)';
      setTimeout(() => {
        bar.row.style.transition = 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
        bar.row.style.opacity = '1';
        bar.row.style.transform = 'translateX(0)';
        setTimeout(() => { bar.row.style.transition = ''; }, 500);
      }, delay + i * 30);
    });
  }
}
