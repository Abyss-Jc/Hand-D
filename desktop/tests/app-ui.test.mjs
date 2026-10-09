import test from 'node:test';
import assert from 'node:assert/strict';

const names = ['drawing','undo','redo','clear','tool-pen','tool-eraser',
  'nav-whiteboard','nav-studio','whiteboard','studio','canvas-shell',
  'modal-canvas-slot','full-overlay','close-expand','expand','camera-fallback',
  'preview','footer-session','status','lamp','gesture','tracking','diagnostics',
  'cursor','restart','model-status','camera-scene','hands-overlay',
  'view-camera','view-clean','toggle-hands','tool-wiggly',
  'workspace-path','select-workspace','workspace-message','dataset-summary',
  'review-sample','refresh-dataset','review-accept','review-reject',
  'review-drop','review-restore','build-snapshot','snapshot-result',
  'collect-participant','collect-gesture','collect-hand','collect-target',
  'collect-interval','collect-start','collect-pause','collect-resume',
  'collect-finish','collect-progress'];
class Element {
  constructor(id) {
    this.id = id; this.hidden = false; this.listeners = {}; this.children = [];
    this.value = ''; this.disabled = false;
    this.style = {}; this.attributes = {}; this.classList = {toggle() {}};
    this.parentElement = null;
  }
  addEventListener(key, cb) { this.listeners[key] = cb; }
  setAttribute(key, val) { this.attributes[key] = val; }
  removeAttribute(key) { delete this.attributes[key]; }
  getAttribute(key) { return this.attributes[key] ?? null; }
  append(node) { this.children.push(node); node.parentElement = this; }
  prepend(node) { this.children.unshift(node); node.parentElement = this; }
  replaceChildren() { this.children = []; }
  remove() {
    if (!this.parentElement) return;
    const list=this.parentElement.children;
    list.splice(list.indexOf(this),1);
    this.parentElement=null;
  }
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
  createElement(tag) { return new Element(tag); },
  addEventListener() {},
};
globalThis.window = {confirm: () => true};
const timers = [];
globalThis.setInterval = callback => { timers.push(callback); };
let reducedMotion = false;
globalThis.matchMedia = () => ({get matches(){return reducedMotion;}});
const connections = [];
globalThis.WebSocket = class {
  static CLOSED = 3;
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    connections.push(this);
  }
  close() { this.readyState = 3; this.onclose?.(); }
  send(data) { this.sent = [...(this.sent || []),JSON.parse(data)]; }
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

test('camera/clean mode, skeletal visibility and Wiggly style preserve strokes', async()=>{
  elements['view-clean'].emit('click');
  assert.equal(elements['canvas-shell'].classList.selected, undefined);
  assert.equal(elements['view-clean'].attributes['aria-pressed'],'true');
  elements['view-camera'].emit('click');
  assert.equal(elements['view-camera'].attributes['aria-pressed'],'true');
  elements['toggle-hands'].emit('click');
  assert.equal(elements['hands-overlay'].style.display,'none');
  elements['toggle-hands'].emit('click');
  assert.equal(elements['hands-overlay'].style.display,'');
  elements['tool-wiggly'].emit('click');
  assert.equal(elements['tool-wiggly'].attributes['aria-pressed'],'true');
  elements.drawing.emit('pointerdown',{button:0,pointerId:1,clientX:150,clientY:200});
  elements.drawing.emit('pointermove',{clientX:300,clientY:400});
  await Promise.resolve();
  assert.equal(elements.drawing.children.at(-1).attributes.stroke,'#5263e6');
  elements.undo.emit('click');
  elements.redo.emit('click');
  await Promise.resolve();
  assert.equal(elements.drawing.children.at(-1).attributes.stroke,'#5263e6');
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
  assert.match(elements.status.textContent, /CONNECTED/);
  assert.match(elements['model-status'].textContent, /LEGACY MODEL ACTIVE/);
  await timers[0]();
  assert.equal(connections.length, 1, 'healthy WS should not reconnect each poll');

  connections[0].sendEvent({
    type:'runtime.update',runtime_session_id:'session-a',seq:1,timestamp_ms:10,
    payload:{drawing:{physical_hand:'Right',raw_gesture:'Index_Finger',
                      stable_gesture:'Index_Finger',action:null,pointer:null},
             modifier:{action:null},health:{model:'legacy_unverified',camera:'tracking'}},
  });
  assert.equal(elements.gesture.textContent, 'Index_Finger');
  assert.match(elements['model-status'].textContent, /NOT INDEPENDENTLY VERIFIED/);
});

test('Modifier skeleton stays visible when Drawing Hand is absent',()=>{
  const points=Array.from({length:21},(_,i)=>({x:i/40,y:.4}));
  connections.at(-1).sendEvent({
    type:'runtime.update',runtime_session_id:'session-a',seq:2,timestamp_ms:20,
    payload:{
      drawing:{physical_hand:'Right',pointer:null,landmarks:null,action:null},
      modifier:{physical_hand:'Left',pointer:null,landmarks:points,action:null},
      health:{model:'legacy_unverified',camera:'tracking'},
    },
  });
  assert.equal(elements['hands-overlay'].children[0].style.display,'none');
  assert.equal(elements['hands-overlay'].children[1].style.display,'');
});

test('reduced-motion restores Wiggly to static path without altering strokes',async()=>{
  elements['tool-wiggly'].emit('click');
  elements.drawing.emit('pointerdown',{button:0,pointerId:1,clientX:130,clientY:110});
  for(let i=0;i<8;i++)
    elements.drawing.emit('pointermove',{clientX:155+i*16,clientY:160+i*7});
  await Promise.resolve();
  const path=elements.drawing.children.at(-1);
  const stable=path.attributes.d;
  timers[1](); // Wiggly presentation timer; transport poll is timers[0].
  assert.notEqual(path.attributes.d,stable);
  reducedMotion=true;
  timers[1]();
  assert.equal(path.attributes.d,stable);
  reducedMotion=false;
  elements.drawing.emit('pointerup',{});
});

test('Studio uses explicit workspace and manual review/immutable snapshot commands',async()=>{
  elements['workspace-path'].value='/tmp/handd-user-workspace';
  let selected=null;
  window.__TAURI__ = {core:{invoke:async (command,args)=>{
    if(command==='select_workspace'){selected=args.path;return args.path;}
    if(command==='sidecar_status') return {
      port:39009,token:'synthetic-token',runtime_session_id:'studio-session',
      workspace:'/tmp/handd-user-workspace',
    };
    throw Error(command);
  }}};
  elements['select-workspace'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(selected,'/tmp/handd-user-workspace');
  await timers[0]();
  const socket=connections.at(-1);
  socket.sendEvent({type:'runtime.ready',snapshot:{
    runtime_session_id:'studio-session',seq:0,timestamp_ms:-1,health:{model:'unavailable'}
  }});
  const overviewRequest=socket.sent.find(request=>request.action==='overview');
  assert.ok(overviewRequest);
  socket.sendEvent({type:'studio.response',request_id:overviewRequest.request_id,
    ok:true,data:{workspace:'/tmp/handd-user-workspace',
      sample_count:1,review_counts:{unreviewed:0,accepted:1,rejected:0},
      samples:[{sample_id:'SAMPLE0',gesture:'Fist',review_status:'unreviewed',
                lifecycle_status:'active'}],snapshot_ready:true,eligible_count:1,
      snapshot_blockers:[],snapshot_warnings:[]}});
  assert.equal(elements['review-sample'].children.length,1);
  elements['review-sample'].value='SAMPLE0';
  elements['review-accept'].emit('click');
  assert.equal(socket.sent.at(-1).action,'accept');
  assert.equal(socket.sent.at(-1).sample_id,'SAMPLE0');
  elements['build-snapshot'].emit('click');
  assert.equal(socket.sent.at(-1).action,'snapshot');
});

test('Studio Collect starts only on explicit action and exposes Pause Resume Finish',async()=>{
  // Existing Studio WebSocket connected by previous test.
  const socket=connections.at(-1);
  elements['collect-participant'].value='P001';
  elements['collect-gesture'].value='Fist';
  elements['collect-hand'].value='Right';
  elements['collect-target'].value='2';
  elements['collect-interval'].value='100';
  const before=socket.sent?.length || 0;
  elements['collect-start'].emit('click');
  assert.equal(socket.sent.length,before+1);
  assert.equal(socket.sent.at(-1).action,'collect_start');
  assert.equal(socket.sent.at(-1).participant,'P001');
  assert.equal(socket.sent.at(-1).target,2);
  const request_id=socket.sent.at(-1).request_id;
  socket.sendEvent({type:'studio.response',request_id,ok:true,data:{
    state:'capturing',count:0,target:2,participant:'P001',gesture:'Fist'
  }});
  assert.match(elements['collect-progress'].textContent,/0\s*\/\s*2/);
  elements['collect-pause'].emit('click');
  assert.equal(socket.sent.at(-1).action,'collect_pause');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{state:'paused',count:1,target:2}});
  elements['collect-resume'].emit('click');
  assert.equal(socket.sent.at(-1).action,'collect_resume');
  elements['collect-finish'].emit('click');
  assert.equal(socket.sent.at(-1).action,'collect_finish');
});

test('eraser UI subtracts only existing ink, not camera or controls', async()=>{
  const svg=elements.drawing;
  elements['tool-pen'].emit('click');
  svg.emit('pointerdown',{button:0,pointerId:1,clientX:180,clientY:160});
  svg.emit('pointermove',{clientX:340,clientY:300});
  svg.emit('pointerup',{});
  await Promise.resolve();
  elements['tool-eraser'].emit('click');
  svg.emit('pointerdown',{button:0,pointerId:2,clientX:260,clientY:220});
  svg.emit('pointermove',{clientX:290,clientY:255});
  svg.emit('pointerup',{});
  await Promise.resolve();
  const defs=svg.children.find(child=>child.id==='defs');
  assert.ok(defs,'eraser must only contribute a mask in SVG defs');
  const mask=defs.children.at(-1);
  const erasedPath=mask.children.at(-1);
  assert.equal(mask.id,'mask');
  assert.equal(erasedPath.attributes.stroke,'black');
  assert.notEqual(erasedPath.attributes.stroke,'#fffef9',
    'white painting would cover the camera');
  assert.equal(erasedPath.parentElement,mask);
  const oldInk=svg.children.find(child=>child.id==='g');
  assert.equal(oldInk.attributes.mask?.startsWith('url(#handd-erase-'),true);
  elements.undo.emit('click');
  assert.equal(svg.children.some(child=>child.id==='defs'),false,
    'undo eraser restores ink and removes its mask');
  elements.redo.emit('click');
  assert.equal(svg.children.some(child=>child.id==='defs'),true);
});
