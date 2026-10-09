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
      path.setAttribute('stroke', stroke.tool === 'erase' ? '#fffef9' : '#1a1d1c');
      path.setAttribute('stroke-width', stroke.tool === 'erase' ? '32' : '4');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('pointer-events', 'none');
      this.svg.append(path);
      this.elements.set(stroke, {path, count: 0, geometry: ''});
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
    }
  }
}
