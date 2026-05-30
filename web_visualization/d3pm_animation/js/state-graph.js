/* ============================================================
   D3PM — State Radiation Graph (SVG)
   ============================================================ */

class StateGraph {
  constructor(svgId) {
    this.svg = document.getElementById(svgId);
    this.ns = 'http://www.w3.org/2000/svg';
    this.centerX = 0;
    this.centerY = 0;
    this.radius = 160;
    this.nodes = [];
    this.lines = [];
    this.labels = [];
    this.centerCircle = null;
    this.ringBg = null;
    this.currentState = 0;

    this._initSVG();
    this._drawBackground();
    this._drawNodes();
    this._drawCenter();
  }

  _initSVG() {
    // Defs for glow filter
    const defs = document.createElementNS(this.ns, 'defs');
    defs.innerHTML = `
      <filter id="node-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <filter id="center-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    `;
    this.svg.appendChild(defs);
  }

  _drawBackground() {
    // Faint concentric rings
    const ringRadii = [60, 100, 140];
    for (const r of ringRadii) {
      const circle = document.createElementNS(this.ns, 'circle');
      circle.setAttribute('cx', 0);
      circle.setAttribute('cy', 0);
      circle.setAttribute('r', r);
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', 'rgba(0,0,0,0.05)');
      circle.setAttribute('stroke-width', '0.5');
      this.svg.appendChild(circle);
    }

    // Subtle crosshairs
    for (const angle of [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4]) {
      const line = document.createElementNS(this.ns, 'line');
      const x = Math.cos(angle) * this.radius * 1.2;
      const y = Math.sin(angle) * this.radius * 1.2;
      line.setAttribute('x1', -x);
      line.setAttribute('y1', -y);
      line.setAttribute('x2', x);
      line.setAttribute('y2', y);
      line.setAttribute('stroke', 'rgba(0,0,0,0.03)');
      line.setAttribute('stroke-width', '0.5');
      this.svg.appendChild(line);
    }
  }

  _getNodePos(index) {
    const angle = (index / D3PM.K) * Math.PI * 2 - Math.PI / 2;
    return {
      x: Math.cos(angle) * this.radius,
      y: Math.sin(angle) * this.radius,
      angle,
    };
  }

  _drawNodes() {
    for (let i = 0; i < D3PM.K; i++) {
      const pos = this._getNodePos(i);
      const color = D3PM.STATE_COLORS[i];

      // Connection line (from center to node)
      const line = document.createElementNS(this.ns, 'line');
      line.setAttribute('x1', 0);
      line.setAttribute('y1', 0);
      line.setAttribute('x2', pos.x);
      line.setAttribute('y2', pos.y);
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', '1');
      line.setAttribute('opacity', '0.08');
      line.classList.add('connection-line');
      line.dataset.index = i;
      this.svg.appendChild(line);
      this.lines.push(line);

      // Node circle
      const circle = document.createElementNS(this.ns, 'circle');
      circle.setAttribute('cx', pos.x);
      circle.setAttribute('cy', pos.y);
      circle.setAttribute('r', '4');
      circle.setAttribute('fill', color);
      circle.setAttribute('opacity', '0.3');
      circle.classList.add('state-node');
      circle.dataset.index = i;
      this.svg.appendChild(circle);
      this.nodes.push(circle);

      // Label
      const text = document.createElementNS(this.ns, 'text');
      const labelDist = this.radius + 18;
      const lx = Math.cos(pos.angle) * labelDist;
      const ly = Math.sin(pos.angle) * labelDist;
      text.setAttribute('x', lx);
      text.setAttribute('y', ly);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('fill', 'rgba(255,255,255,0.15)');
      text.setAttribute('font-size', '6');
      text.setAttribute('font-family', "'JetBrains Mono', monospace");
      text.textContent = i;
      text.classList.add('state-label');
      text.dataset.index = i;
      this.svg.appendChild(text);
      this.labels.push(text);
    }
  }

  _drawCenter() {
    // The center node will be drawn via the CSS indicator overlay
    // But we add a subtle ring in SVG
    this.ringBg = document.createElementNS(this.ns, 'circle');
    this.ringBg.setAttribute('cx', 0);
    this.ringBg.setAttribute('cy', 0);
    this.ringBg.setAttribute('r', '22');
    this.ringBg.setAttribute('fill', 'none');
    this.ringBg.setAttribute('stroke', 'rgba(0,0,0,0.06)');
    this.ringBg.setAttribute('stroke-width', '1');
    this.svg.appendChild(this.ringBg);
  }

  /**
   * Update the radiation diagram for a given state and probability distribution
   * @param {number} currentState - current state index
   * @param {number[]} probs - probability distribution (length K)
   * @param {boolean} isReverse - whether in reverse mode
   */
  update(currentState, probs, isReverse = false) {
    this.currentState = currentState;
    const K = D3PM.K;
    const colors = D3PM.STATE_COLORS;

    // Update center indicator
    const dot = document.getElementById('center-state-dot');
    const label = document.getElementById('center-state-label');
    if (dot) {
      dot.style.background = colors[currentState];
      if (isReverse) {
        dot.style.borderColor = 'rgba(255, 107, 107, 0.4)';
        dot.style.boxShadow = '0 0 20px rgba(255, 107, 107, 0.2)';
      } else {
        dot.style.borderColor = 'rgba(0, 229, 255, 0.4)';
        dot.style.boxShadow = '0 0 20px rgba(0, 229, 255, 0.15)';
      }
    }
    if (label) {
      label.textContent = `s=${currentState}`;
    }

    // Update nodes and lines
    for (let i = 0; i < K; i++) {
      const p = probs[i] || 0;
      const minR = 2.5;
      const maxR = 20;
      const r = minR + p * (maxR - minR) * K * 0.5;

      // Node circle
      const circle = this.nodes[i];
      circle.setAttribute('r', Math.max(minR, Math.min(maxR, r)));
      circle.setAttribute('opacity', isReverse ? 0.25 + p * 0.7 : 0.15 + p * 0.75);

      if (p > 0.15) {
        circle.setAttribute('filter', 'url(#node-glow)');
        circle.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
      } else {
        circle.removeAttribute('filter');
      }

      // Connection line
      const line = this.lines[i];
      line.setAttribute('opacity', Math.max(0.02, p * 0.9));
      line.setAttribute('stroke-width', Math.max(0.5, p * 4 * K * 0.15));
      line.setAttribute('stroke', colors[i]);

      // Label
      const label = this.labels[i];
      if (p > 0.05) {
        label.setAttribute('fill', 'rgba(255,255,255,0.4)');
        label.setAttribute('font-size', '6.5');
      } else {
        label.setAttribute('fill', 'rgba(255,255,255,0.12)');
        label.setAttribute('font-size', '5.5');
      }
    }
  }

  /**
   * Particle burst effect on state change
   */
  burstEffect(fromState, toState) {
    const pos = this._getNodePos(toState);
    const svgRect = this.svg.getBoundingClientRect();
    const scaleX = svgRect.width / 440;
    const scaleY = svgRect.height / 440;
    const cx = svgRect.left + svgRect.width / 2;
    const cy = svgRect.top + svgRect.height / 2;
    const tx = cx + pos.x * scaleX;
    const ty = cy + pos.y * scaleY;
    const color = D3PM.STATE_COLORS[toState];

    for (let i = 0; i < 8; i++) {
      const particle = document.createElement('div');
      const size = 2 + Math.random() * 3;
      const angle = Math.random() * Math.PI * 2;
      const dist = 15 + Math.random() * 25;
      particle.className = 'particle';
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
        box-shadow: 0 0 6px ${color};
      `;
      document.body.appendChild(particle);

      const dx = Math.cos(angle) * (dist + Math.random() * 20);
      const dy = Math.sin(angle) * (dist + Math.random() * 20);

      particle.animate([
        { transform: 'translate(0,0) scale(1)', opacity: 0.8 },
        { transform: `translate(${dx}px, ${dy}px) scale(0)`, opacity: 0 },
      ], {
        duration: 600 + Math.random() * 400,
        easing: 'cubic-bezier(0.25, 0.1, 0.25, 1)',
      }).onfinish = () => particle.remove();
    }
  }

  /**
   * Update center state color (simple version for external call)
   */
  setCenterState(state) {
    this.currentState = state;
    const dot = document.getElementById('center-state-dot');
    const label = document.getElementById('center-state-label');
    if (dot) dot.style.background = D3PM.STATE_COLORS[state];
    if (label) label.textContent = `s=${state}`;
  }
}
