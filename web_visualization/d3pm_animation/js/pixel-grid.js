/* ============================================================
   D3PM — Pixel Grid Component
   ============================================================ */

class PixelGrid {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.cells = [];
    this.gridData = null;
    this.activeRow = 0;
    this.activeCol = 0;
    this.isDimmed = false;
  }

  init(gridData) {
    this.gridData = gridData;
    const { GRID_ROWS, GRID_COLS, STATE_COLORS } = D3PM;

    this.container.innerHTML = '';
    this.cells = [];

    const grid = this.container;
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = `repeat(${GRID_COLS}, 36px)`;
    grid.style.gridTemplateRows = `repeat(${GRID_ROWS}, 36px)`;
    grid.style.gap = '4px';

    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const cell = document.createElement('div');
        cell.className = 'pixel-cell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        const state = gridData[r][c];
        cell.style.background = STATE_COLORS[state];
        this.cells.push(cell);
        grid.appendChild(cell);
      }
    }

    return this;
  }

  highlight(row, col) {
    this.activeRow = row;
    this.activeCol = col;
    this.cells.forEach((cell, i) => {
      const r = Math.floor(i / D3PM.GRID_COLS);
      const c = i % D3PM.GRID_COLS;
      cell.classList.remove('active');
      if (r === row && c === col) {
        cell.classList.add('active');
      }
    });
    return this;
  }

  dimOthers(activeOnly) {
    this.isDimmed = activeOnly;
    this.cells.forEach((cell, i) => {
      const r = Math.floor(i / D3PM.GRID_COLS);
      const c = i % D3PM.GRID_COLS;
      if (activeOnly) {
        if (r === this.activeRow && c === this.activeCol) {
          cell.classList.remove('dimmed');
          cell.classList.add('active');
        } else {
          cell.classList.add('dimmed');
          cell.classList.remove('active');
        }
      } else {
        cell.classList.remove('dimmed', 'active');
      }
    });
    return this;
  }

  setCellState(row, col, stateIndex) {
    const idx = row * D3PM.GRID_COLS + col;
    if (this.cells[idx]) {
      this.cells[idx].style.background = D3PM.STATE_COLORS[stateIndex];
      this.gridData[row][col] = stateIndex;
    }
    return this;
  }

  /**
   * Animate entrance with staggered delay
   */
  animateIn(delay = 0) {
    this.cells.forEach((cell, i) => {
      const row = Math.floor(i / D3PM.GRID_COLS);
      const col = i % D3PM.GRID_COLS;
      const d = (row + col) * 40 + delay;
      cell.style.opacity = '0';
      cell.style.transform = 'scale(0.5)';
      setTimeout(() => {
        cell.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
        cell.style.opacity = '1';
        cell.style.transform = 'scale(1)';
        setTimeout(() => {
          cell.style.transition = '';
        }, 600);
      }, d);
    });
    return this;
  }

  /**
   * Particle burst at a specific cell position
   */
  burst(row, col, color) {
    const idx = row * D3PM.GRID_COLS + col;
    const cell = this.cells[idx];
    if (!cell) return;
    const rect = cell.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    for (let i = 0; i < 12; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      const size = 2 + Math.random() * 3;
      const angle = (Math.PI * 2 * i) / 12 + (Math.random() - 0.5) * 0.3;
      const dist = 30 + Math.random() * 40;
      particle.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        background: ${color};
        left: ${cx}px;
        top: ${cy}px;
        border-radius: 50%;
        position: fixed;
        pointer-events: none;
        z-index: 50;
        box-shadow: 0 0 4px ${color};
      `;
      document.body.appendChild(particle);
      const tx = Math.cos(angle) * dist;
      const ty = Math.sin(angle) * dist;
      particle.animate([
        { transform: 'translate(0,0) scale(1)', opacity: 1 },
        { transform: `translate(${tx}px, ${ty}px) scale(0)`, opacity: 0 },
      ], {
        duration: 500 + Math.random() * 300,
        easing: 'cubic-bezier(0.25, 0.1, 0.25, 1)',
      }).onfinish = () => particle.remove();
    }
  }
}
