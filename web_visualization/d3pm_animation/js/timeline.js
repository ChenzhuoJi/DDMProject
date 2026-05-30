/* ============================================================
   D3PM — Timeline Component
   ============================================================ */

class Timeline {
  constructor(trackId) {
    this.track = document.getElementById(trackId);
    this.steps = [];
    this.activeIndex = 0;
    this.isReverse = false;
    this.onStepClick = null;
  }

  init(T, onClick) {
    this.track.innerHTML = '';
    this.steps = [];
    this.onStepClick = onClick;

    for (let i = 0; i <= T; i++) {
      const step = document.createElement('div');
      step.className = 'timeline-step';
      step.dataset.index = i;

      // Position
      const pct = (i / T) * 100;
      step.style.left = `${pct}%`;

      step.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onStepClick) this.onStepClick(i);
      });

      this.track.appendChild(step);
      this.steps.push(step);
    }

    // Gradient background
    this.track.style.background = 'linear-gradient(90deg, rgba(255,107,107,0.15), rgba(0,229,255,0.15))';

    return this;
  }

  setActive(index, isReverse = false) {
    this.activeIndex = index;
    this.isReverse = isReverse;

    const T = this.steps.length - 1;

    for (let i = 0; i <= T; i++) {
      const step = this.steps[i];
      step.classList.remove('active', 'active-reverse', 'past', 'past-reverse');

      if (i === index) {
        step.classList.add(isReverse ? 'active-reverse' : 'active');
      } else if (isReverse ? i > index : i < index) {
        step.classList.add(isReverse ? 'past-reverse' : 'past');
      }
    }
  }

  animateTo(index, isReverse = false) {
    return new Promise((resolve) => {
      this.setActive(index, isReverse);
      // Brief delay for visual feedback
      setTimeout(resolve, 200);
    });
  }
}
