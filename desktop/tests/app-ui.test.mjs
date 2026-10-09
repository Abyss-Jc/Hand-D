import test from 'node:test';
import assert from 'node:assert/strict';

const names = ['drawing','undo','redo','clear','tool-pen','tool-eraser',
  'nav-whiteboard','nav-studio','whiteboard','studio','canvas-shell',
  'modal-canvas-slot','full-overlay','close-expand','expand','camera-fallback',
  'preview','footer-session','status','lamp','gesture','tracking','diagnostics',
  'cursor','restart','model-status'];
class Element {
  constructor(id) {
    this.id = id; this.hidden = false; this.listeners = {}; this.children = [];
    this.style = {}; this.attributes = {}; this.classList = {toggle() {}};
    this.parentElement = null;
  }
  addEventListener(key, cb) { this.listeners[key] = cb; }
  setAttribute(key, val) { this.attributes[key] = val; }
  removeAttribute(key) { delete this.attributes[key]; }
  append(node) { this.children.push(node); node.parentElement = this; }
  prepend(node) { this.children.unshift(node); node.parentElement = this; }
  replaceChildren() { this.children = []; }
  setPointerCapture() {}
  getBoundingClientRect() { return {left:0,top:0,width:1000,height:600}; }
  focus() {}
  emit(type, event = {}) { this.listeners[type]?.(event); }
}
const elements = Object.fromEntries(names.map(name => [name, new Element(name)]));
elements.drawing.parentElement = elements['canvas-shell'];
elements['canvas-shell'].parentElement = new Element('workspace');
globalThis.document = {
  getElementById(id) { if (!elements[id]) throw Error(id); return elements[id]; },
  createElementNS(_namespace, tag) { return new Element(tag); },
  addEventListener() {},
};
globalThis.window = {};
const timers = [];
globalThis.setInterval = callback => { timers.push(callback); };
const connections = [];
globalThis.WebSocket = class {
  static CLOSED = 3;
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    connections.push(this);
  }
  close() { this.readyState = 3; this.onclose?.(); }
  sendEvent(data) { this.onmessage?.({data: JSON.stringify(data)}); }
};
await import('../web/app.js');

test('mouse strokes support undo/redo and survive modal reparenting', () => {
  elements.drawing.emit('pointerdown', {button:0,pointerId:1,clientX:100,clientY:120});
  elements.drawing.emit('pointermove', {clientX:300,clientY:400});
  assert.equal(elements.drawing.children.length, 1);
  elements.undo.emit('click');
  assert.equal(elements.drawing.children.length, 0);
  elements.redo.emit('click');
  assert.equal(elements.drawing.children.length, 1);
  elements.expand.emit('click');
  assert.equal(elements['full-overlay'].hidden, false);
  elements['close-expand'].emit('click');
  assert.equal(elements['full-overlay'].hidden, true);
  assert.equal(elements.drawing.children.length, 1);
});

test('Whiteboard and Studio navigation does not erase canvas', () => {
  elements['nav-studio'].emit('click');
  assert.equal(elements.whiteboard.hidden, true);
  elements['nav-whiteboard'].emit('click');
  assert.equal(elements.whiteboard.hidden, false);
  assert.equal(elements.drawing.children.length, 1);
});

test('connection READY distinguishes legacy model from unavailable model', async () => {
  const meta = {port: 39001, token: 'synthetic-test-token', runtime_session_id: 'session-a'};
  window.__TAURI__ = {core: {invoke: async command => {
    assert.equal(command, 'sidecar_status');
    return meta;
  }}};
  await timers[0]();
  assert.equal(connections.length, 1);
  assert.match(connections[0].url, /127[.]0[.]0[.]1:39001[/]ws/);
  connections[0].sendEvent({type: 'runtime.ready', snapshot: {
    runtime_session_id:'session-a',seq:0,timestamp_ms:-1,
    health:{model:'legacy_unverified',camera:'starting'}
  }});
  assert.match(elements.status.textContent, /CONECTADO/);
  assert.match(elements['model-status'].textContent, /ANTIGUO ACTIVO/);
  await timers[0]();
  assert.equal(connections.length, 1, 'healthy WS should not reconnect each poll');

  connections[0].sendEvent({
    type:'runtime.update',runtime_session_id:'session-a',seq:1,timestamp_ms:10,
    payload:{drawing:{physical_hand:'Right',raw_gesture:'Index_Finger',
                      stable_gesture:'Index_Finger',action:null,pointer:null},
             modifier:{action:null},health:{model:'legacy_unverified',camera:'tracking'}},
  });
  assert.equal(elements.gesture.textContent, 'Index_Finger');
  assert.match(elements['model-status'].textContent, /SIN VALIDACIÓN/);
});
