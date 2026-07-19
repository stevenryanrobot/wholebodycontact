// Plain HTML/CSS HUD: detection bar, per-region probability bars,
// applied vs predicted readouts, pause/reset buttons, status line.

export class Hud {
  /**
   * @param regions string[] region names (bar order)
   * @param onPause () => boolean  toggles pause, returns new paused state
   * @param onReset () => void
   * @param onModelChange (name: string) => void  sensor model picker
   */
  constructor(regions, onPause, onReset, onModelChange = null) {
    this.el = {
      status: document.getElementById('status'),
      detVal: document.getElementById('det-val'),
      detBar: document.getElementById('det-bar'),
      regionBars: document.getElementById('region-bars'),
      applied: document.getElementById('applied-val'),
      predicted: document.getElementById('predicted-val'),
      pause: document.getElementById('btn-pause'),
      reset: document.getElementById('btn-reset'),
      modelSelect: document.getElementById('model-select'),
      dragLabel: document.getElementById('drag-label'),
      grabbing: document.getElementById('grabbing-val'),
    };

    // Bars for the localization heads. 5-region models -> 5 bars; 24-link
    // models -> one bar per link. Rebuilt on model switch via setBars().
    this.setBars(regions.map((r) => ({ key: r, label: r })));

    this.el.pause.addEventListener('click', () => {
      const paused = onPause();
      this.el.pause.textContent = paused ? 'Resume' : 'Pause';
    });
    this.el.reset.addEventListener('click', onReset);
    if (this.el.modelSelect && onModelChange) {
      this.el.modelSelect.addEventListener('change', (e) => onModelChange(e.target.value));
    }
  }

  /**
   * (Re)build the localization bar rows.
   * @param entries {key,label}[]  key = prob-dict key, label = display text
   */
  setBars(entries) {
    this.barKeys = entries.map((e) => e.key);
    this.el.regionBars.innerHTML = '';
    this.el.regionBars.classList.toggle('links', entries.length > 8);
    this.bars = {};
    for (const { key, label } of entries) {
      const row = document.createElement('div');
      row.className = 'region-row';
      row.innerHTML = `<span title="${key}">${label}</span><div class="bar-wrap"><div class="bar"></div></div><span class="pct">0%</span>`;
      this.el.regionBars.appendChild(row);
      this.bars[key] = { bar: row.querySelector('.bar'), pct: row.querySelector('.pct') };
    }
  }

  status(msg, isError = false) {
    this.el.status.textContent = msg;
    this.el.status.classList.toggle('error', isError);
    if (isError) console.error(msg);
  }

  hideStatus() { this.el.status.style.display = 'none'; }

  /** Show which MuJoCo body the drag currently grabs (null = not grabbing).
   * The hands hang next to the hips, so a "grab the hand" click easily lands on
   * a hip/thigh body instead — surfacing the grabbed body makes the resulting
   * region prediction (e.g. a leg) legible rather than looking like a mislocalize. */
  setGrabbed(bodyName) {
    if (!this.el.grabbing) return;
    this.el.grabbing.textContent = bodyName ?? '—';
    this.el.grabbing.classList.toggle('active', !!bodyName);
  }

  /** Reposition the det-bar threshold markers (per-model operating point). */
  setThresholds(lo, hi) {
    const loEl = document.querySelector('#det-wrap .th-marker.lo');
    const hiEl = document.querySelector('#det-wrap .th-marker:not(.lo)');
    if (loEl) { loEl.style.left = `${(lo * 100).toFixed(0)}%`; loEl.title = `th_lo ${lo}`; }
    if (hiEl) { hiEl.style.left = `${(hi * 100).toFixed(0)}%`; hiEl.title = `th_hi ${hi}`; }
  }

  /**
   * @param det number 0..1 (RAW sigmoid(det_logit) — shown even when the
   *        debounced gate is off; threshold markers sit at 0.35 / 0.5)
   * @param probs Record<region, number>
   * @param appliedN number
   * @param predictedN number (already gated by the debounced contact state)
   * @param contactOn debounced contact state (colors the det bar green)
   */
  update(det, probs, appliedN, predictedN, contactOn = false) {
    this.el.detVal.textContent = det.toFixed(2);
    this.el.detBar.style.width = `${(det * 100).toFixed(0)}%`;
    this.el.detBar.classList.toggle('contact-on', !!contactOn);
    for (const [r, { bar, pct }] of Object.entries(this.bars)) {
      const p = probs[r] ?? 0;
      bar.style.width = `${(p * 100).toFixed(0)}%`;
      pct.textContent = `${(p * 100).toFixed(0)}%`;
    }
    this.el.applied.textContent = `${appliedN.toFixed(1)} N`;
    this.el.predicted.textContent = `${predictedN.toFixed(1)} N`;
  }
}
