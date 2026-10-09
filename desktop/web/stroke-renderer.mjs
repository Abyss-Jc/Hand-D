/** Incremental SVG drawing. The durable Strokes model owns points and undo history.
 * Only the active path's 'd' attribute changes during pointer movement.
 * Browser repaint is scheduled at most once per animation frame.
 */
const SVG_NS = 'http://www.w3.org/2000/svg';

export class StrokeRenderer {
  constructor(svg, {
    scheduleFrame = callback => typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame(callback) : queueMicrotask(callback),
    createPath = () => document.createElementNS(SVG_NS, 'path'),
  } = {}) {
    this.svg = svg;
    this.scheduleFrame = scheduleFrame;
    this.createPath = createPath;
    this.elements = new Map();
    this.pending = new Set();
    this.scheduled = false;
  }

  sync(paths) {
    const keep = new Set(paths);
    for (const [stroke, entry] of this.elements) {
      if (!keep.has(stroke)) {
        entry.path.remove();
        this.elements.delete(stroke);
        this.pending.delete(stroke);
      }
    }
    for (const stroke of paths) {
      if (this.elements.has(stroke)) continue;
      const path = this.createPath();
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', stroke.tool === 'erase' ? '#fffef9'
        : stroke.tool === 'wiggly' ? '#5263e6' : '#1a1d1c');
      path.setAttribute('stroke-width', stroke.tool === 'erase' ? '32'
        : stroke.tool === 'wiggly' ? '5' : '4');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('pointer-events', 'none');
      this.svg.append(path);
      this.elements.set(stroke, {path, count: 0, geometry: '', animated: false});
      this.pointAdded(stroke);
    }
  }

  pointAdded(stroke) {
    if (!this.elements.has(stroke)) return;
    this.pending.add(stroke);
    if (this.scheduled) return;
    this.scheduled = true;
    this.scheduleFrame(() => this.flush());
  }

  flush() {
    this.scheduled = false;
    const pending = [...this.pending];
    this.pending.clear();
    for (const stroke of pending) {
      const entry = this.elements.get(stroke);
      if (!entry) continue; // stroke was undone or document cleared
      const count = stroke.points.length;
      if (count < entry.count) {
        entry.geometry = '';
        entry.count = 0;
      }
      if (count === entry.count) continue;
      for (let i = entry.count; i < count; i++) {
        const p = stroke.points[i];
        entry.geometry += (i === 0 ? 'M' : ' L')
          + (p.x * 1000).toFixed(2) + ' ' + (p.y * 600).toFixed(2);
      }
      entry.count = count;
      entry.path.setAttribute(
        'd', entry.geometry + (count === 1 ? ' l0.1 0.1' : '')
      );
      entry.animated = false;
    }
  }

  /** Display-only animation capped at 320 vertices per Wiggly stroke. */
  animateWiggly(phase, {enabled = true} = {}) {
    for (const [stroke, entry] of this.elements) {
      if (stroke.tool !== 'wiggly' || !stroke.points.length) continue;
      if (!enabled) {
        if (entry.animated) {
          entry.path.setAttribute('d', entry.geometry
            + (stroke.points.length === 1 ? ' l0.1 0.1' : ''));
          entry.animated = false;
        }
        continue;
      }
      const points = stroke.points;
      const stride = Math.max(1, Math.ceil(points.length / 320));
      const parts = [];
      for (let i = 0; i < points.length; i += stride) {
        const point = points[i];
        const offsetX = Math.sin(i * 1.3 + phase) * 1.8;
        const offsetY = Math.cos(i * 1.7 + phase * 1.1) * 1.6;
        parts.push((parts.length ? ' L' : 'M')
          + (point.x * 1000 + offsetX).toFixed(2)
          + ' ' + (point.y * 600 + offsetY).toFixed(2));
      }
      if ((points.length - 1) % stride !== 0) {
        const point = points[points.length - 1];
        parts.push(' L' + (point.x * 1000).toFixed(2)
          + ' ' + (point.y * 600).toFixed(2));
      }
      entry.path.setAttribute('d', parts.join('')
        + (points.length === 1 ? ' l0.1 0.1' : ''));
      entry.animated = true;
    }
  }
}
