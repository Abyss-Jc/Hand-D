/** Transient gesture event gate. Never owns any Whiteboard document data. */
export class RuntimeGate {
  constructor() { this.session = null; this.seq = -1; this.timestamp = -1; }
  install(snapshot) {
    if (!snapshot || !snapshot.runtime_session_id || !Number.isInteger(snapshot.seq))
      throw new Error('Invalid sidecar READY snapshot');
    this.session = snapshot.runtime_session_id;
    this.seq = snapshot.seq;
    this.timestamp = snapshot.timestamp_ms ?? -1;
  }
  accept(event) {
    if (!event || event.type !== 'runtime.update' || !this.session
        || event.runtime_session_id !== this.session
        || !Number.isInteger(event.seq) || event.seq <= this.seq
        || !Number.isInteger(event.timestamp_ms)
        || event.timestamp_ms <= this.timestamp) return false;
    this.seq = event.seq;
    this.timestamp = event.timestamp_ms;
    return true;
  }
}

/** Self-contained document strokes; resilient to camera/sidecar restart. */
export class Strokes {
  constructor() { this.paths = []; this.redoStack = []; }
  add(points, tool = 'draw') {
    if (!Array.isArray(points) || !points.length) return;
    this.paths.push({points: points.map(p => ({x: p.x, y: p.y})), tool});
    this.redoStack = [];
  }
  undo() { if (this.paths.length) this.redoStack.push(this.paths.pop()); }
  redo() { if (this.redoStack.length) this.paths.push(this.redoStack.pop()); }
  clear() { this.paths = []; this.redoStack = []; }
  serialize() { return JSON.stringify(this.paths); }
}
