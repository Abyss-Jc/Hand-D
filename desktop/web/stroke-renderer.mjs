/** Incremental SVG drawing. The durable Strokes model owns points and undo history.
 * Only the active path's 'd' attribute changes during pointer movement.
 * Browser repaint is scheduled at most once per animation frame.
 */
const SVG_NS = 'http://www.w3.org/2000/svg';
let nextEraserMaskId = 0;

export class StrokeRenderer {
  constructor(svg, {
    scheduleFrame = callback => typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame(callback) : queueMicrotask(callback),
    createPath = () => document.createElementNS(SVG_NS, 'path'),
    createSvg = tag => document.createElementNS(SVG_NS, tag),
  } = {}) {
    this.svg = svg;
    this.scheduleFrame = scheduleFrame;
    this.createPath = createPath;
    this.createSvg = createSvg;
    this.elements = new Map();
    this.pending = new Set();
    this.scheduled = false;
    this.order = [];
    this.defs = null;
  }

  sync(paths) {
    // Normal drawing only appends one new path; NEVER rebuild previous ink on
    // point updates. Undo/redo that removes an eraser rebuilds mask structure
    // while retaining the existing ink <path> nodes.
    const appendOnly = this.order.length <= paths.length
      && this.order.every((stroke, i) => stroke === paths[i]);
    const keep = new Set(paths);
    for (const [stroke, entry] of this.elements) {
      if (!keep.has(stroke)) {
        entry.path.remove();
        this.elements.delete(stroke);
        this.pending.delete(stroke);
      }
    }
    if (!appendOnly) {
      for (const entry of this.elements.values()) entry.path.remove();
      for (const child of [...this.svg.children]) child.remove();
      this.defs = null;
    }
    const newPaths = appendOnly ? paths.slice(this.order.length) : paths;
    for (const stroke of newPaths) {
      if (this.elements.has(stroke)) {
        this._attach(stroke, this.elements.get(stroke).path);
        continue;
      }
      const path = this.createPath();
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', stroke.tool === 'erase' ? 'black'
        : stroke.tool === 'wiggly' ? '#5263e6' : '#1a1d1c');
      path.setAttribute('stroke-width', stroke.tool === 'erase' ? '32'
        : stroke.tool === 'wiggly' ? '5' : '4');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('pointer-events', 'none');
      this._attach(stroke, path);
      this.elements.set(stroke, {path, count: 0, geometry: '', animated: false});
      this.pointAdded(stroke);
    }
    this.order = [...paths];
  }

  _attach(stroke, path) {
    if (stroke.tool !== 'erase') {
      // Ink drawn AFTER an eraser is on top of the previous masked layers.
      this.svg.append(path);
      return;
    }
    if (!this.defs) {
      this.defs = this.createSvg('defs');
      this.svg.append(this.defs);
    }
    const maskId = 'handd-erase-' + (++nextEraserMaskId);
    const mask = this.createSvg('mask');
    mask.setAttribute('id', maskId);
    mask.setAttribute('maskUnits', 'userSpaceOnUse');
    mask.setAttribute('maskContentUnits', 'userSpaceOnUse');
    mask.setAttribute('mask-type', 'luminance');
    mask.setAttribute('x', '0');
    mask.setAttribute('y', '0');
    mask.setAttribute('width', '1000');
    mask.setAttribute('height', '600');
    const opaque = this.createSvg('rect');
    opaque.setAttribute('x', '0');
    opaque.setAttribute('y', '0');
    opaque.setAttribute('width', '1000');
    opaque.setAttribute('height', '600');
    opaque.setAttribute('fill', 'white');
    mask.append(opaque);
    mask.append(path); // black removes *ink alpha*, not camera pixels
    this.defs.append(mask);

    const earlierInk = this.createSvg('g');
    earlierInk.setAttribute('mask', 'url(#' + maskId + ')');
    for (const layer of [...this.svg.children]) {
      if (layer !== this.defs) {
        layer.remove();
        earlierInk.append(layer);
      }
    }
    this.svg.append(earlierInk);
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
        // The original ~2px tremor was barely visible. More playful motion,
        // still bounded around the canonical stroke (never modifying points).
        const offsetX = Math.sin(i * 1.3 + phase) * 8;
        const offsetY = Math.cos(i * 1.7 + phase * 1.1) * 7;
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
