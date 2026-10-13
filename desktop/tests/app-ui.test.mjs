import test from 'node:test';
import assert from 'node:assert/strict';

const names = ['drawing','undo','redo','clear','tool-pen','tool-eraser',
  'nav-whiteboard','nav-studio','whiteboard','studio','canvas-shell',
  'modal-canvas-slot','full-overlay','close-expand','expand','camera-fallback',
  'modal-undo','modal-redo','modal-camera','modal-clean','modal-save',
  'app-header','app-main','app-footer',
  'preview','footer-session','status','lamp','gesture','tracking','diagnostics',
  'cursor','restart','model-status','camera-scene','hands-overlay',
  'view-camera','view-clean','toggle-hands','tool-wiggly',
  'workspace-path','select-workspace','workspace-message','dataset-summary',
  'review-sample','refresh-dataset','review-accept','review-reject',
  'review-drop','review-restore','build-snapshot','snapshot-result',
  'collect-participant','collect-gesture','collect-hand','collect-target',
  'collect-interval','collect-start','collect-pause','collect-resume',
  'collect-finish','collect-progress','models-list','models-refresh',
  'models-activate','models-status','browse-workspace','create-workspace',
  'save-drawing','open-drawing','export-drawing','document-status'];
names.push('collect-preview','collect-hands','collect-visual-status',
  'review-hands','review-visual-status','review-prev','review-next','review-page',
  'review-world','review-angle','collect-visual');
names.push('browse-gesture','browse-participant','browse-review-status','browse-apply',
  'review-capture','review-qc-sample','review-qc-open','review-batch-accept',
  'review-queue-status');
names.push('browse-inspect','review-suggested-sample','review-suggested-open',
  'review-sample-prev','review-sample-next','review-sample-position',
  'review-world-mode','review-pitch','review-reset-view');
names.push('review-sample-scrub');
names.push('workspace-active-label');
names.push('collect-progress-bar','collect-go-review');
names.push('collect-quality-status');
names.push('color-black','color-red','color-blue','color-green','color-purple','color-orange',
  'brush-size','brush-size-value','brush-size-preview',
  'eraser-size','eraser-size-value','eraser-size-preview','modifier-feedback');
for(const section of ['collect','browse','review','snapshots','models','runtime'])
  names.push('studio-pane-'+section);
for(const section of ['collect','browse','review','snapshots','models','runtime'])
  names.push('studio-tab-'+section);
class Element {
  constructor(id) {
    this.id = id; this.hidden = false; this.listeners = {}; this.children = [];
    this.value = ''; this.disabled = false;
    this.style = {}; this.attributes = {}; this.classList = {toggle() {}};
    this.inert = false;
    this.parentElement = null;
  }
  addEventListener(key, cb) { this.listeners[key] = cb; }
  setAttribute(key, val) { this.attributes[key] = val; }
  removeAttribute(key) { delete this.attributes[key]; }
  getAttribute(key) { return this.attributes[key] ?? null; }
  append(node) { node.remove(); this.children.push(node); node.parentElement = this; }
  prepend(node) { node.remove(); this.children.unshift(node); node.parentElement = this; }
  replaceChildren() { this.children = []; }
  remove() {
    if (!this.parentElement) return;
    const list=this.parentElement.children;
    list.splice(list.indexOf(this),1);
    this.parentElement=null;
  }
  setPointerCapture() {}
  getBoundingClientRect() { return {left:0,top:0,width:1000,height:600}; }
  focus() { document.activeElement=this; }
  contains(node) {
    for(let current=node;current;current=current.parentElement)
      if(current===this)return true;
    return false;
  }
  emit(type, event = {}) { return this.listeners[type]?.(event); }
}
const elements = Object.fromEntries(names.map(name => [name, new Element(name)]));
elements['full-overlay'].hidden=true; // matches real HTML's initial hidden attribute
const documentListeners={};
const dialogButtons=['modal-camera','modal-clean','modal-undo','modal-redo',
  'modal-save','close-expand','tool-pen',
  'tool-eraser','tool-wiggly','expand'];
elements['full-overlay'].querySelectorAll=()=>dialogButtons.map(id=>elements[id])
  .filter(button=>!button.disabled);
elements['review-world-mode'].value='canonical';
elements['review-angle'].value='25';
elements['review-pitch'].value='0';
elements.drawing.parentElement = elements['canvas-shell'];
elements['canvas-shell'].parentElement = new Element('workspace');
elements['modal-canvas-slot'].parentElement=elements['full-overlay'];
for(const id of ['close-expand','modal-undo','modal-redo',
                 'modal-camera','modal-clean','modal-save'])
  elements[id].parentElement=elements['full-overlay'];
for(const id of ['tool-pen','tool-eraser','tool-wiggly','expand'])
  elements[id].parentElement=elements['canvas-shell'];
globalThis.document = {
  getElementById(id) { if (!elements[id]) throw Error(id); return elements[id]; },
  createElementNS(_namespace, tag) { return new Element(tag); },
  createElement(tag) { return new Element(tag); },
  body:{style:{overflow:'auto'}},
  activeElement:null,
  addEventListener(type,callback) {
    (documentListeners[type]??=[]).push(callback);
  },
};
function sendKey(key, options={}) {
  const event={key,shiftKey:false,defaultPrevented:false,
    preventDefault(){this.defaultPrevented=true;},...options};
  for(const listener of documentListeners.keydown||[])listener(event);
  return event;
}
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

test('expanded Whiteboard is a keyboard-contained dialog and Escape restores focus',()=>{
  elements.expand.focus();
  elements.expand.emit('click');
  assert.equal(elements['full-overlay'].hidden,false);
  assert.equal(document.activeElement,elements['close-expand']);
  for(const id of ['app-header','app-main','app-footer'])
    assert.equal(elements[id].inert,true,id+' must be inaccessible behind modal');
  assert.equal(document.body.style.overflow,'hidden');
  elements['modal-undo'].focus();
  assert.equal(sendKey('Tab').defaultPrevented,false,
    'Tab between controls is handled by native focus order');
  elements.expand.focus();
  assert.equal(sendKey('Tab').defaultPrevented,true);
  assert.equal(document.activeElement,elements['modal-camera']);
  assert.equal(sendKey('Tab',{shiftKey:true}).defaultPrevented,true);
  assert.equal(document.activeElement,elements.expand);
  const escape=sendKey('Escape');
  assert.equal(escape.defaultPrevented,true);
  assert.equal(elements['full-overlay'].hidden,true);
  assert.equal(document.activeElement,elements.expand);
  for(const id of ['app-header','app-main','app-footer'])
    assert.equal(elements[id].inert,false);
  assert.equal(document.body.style.overflow,'auto');
  assert.equal(sendKey('Escape').defaultPrevented,false,
    'Escape outside modal must remain free for other UI actions');
});

test('expanded view uses one live canvas; modal Undo/Redo and section switch preserve strokes',()=>{
  const originalCanvas=elements['canvas-shell'];
  const originalDrawing=elements.drawing;
  const originalPreview=elements.preview;
  const connectionCount=connections.length;
  const startCount=elements.drawing.children.length;
  elements.expand.emit('click');
  assert.equal(elements.expand.attributes['aria-expanded'],'true');
  assert.equal(elements['modal-canvas-slot'].children.filter(x=>x===originalCanvas).length,1);
  assert.equal(elements.drawing,originalDrawing);
  assert.equal(elements.preview,originalPreview);
  assert.equal(connections.length,connectionCount,'expand must not start a second runtime');
  elements.drawing.emit('pointerdown',{button:0,pointerId:8,clientX:200,clientY:280});
  elements.drawing.emit('pointermove',{clientX:320,clientY:380});
  elements.drawing.emit('pointerup',{});
  assert.equal(elements.drawing.children.length,startCount+1);
  elements['modal-undo'].emit('click');
  assert.equal(elements.drawing.children.length,startCount);
  elements['modal-redo'].emit('click');
  assert.equal(elements.drawing.children.length,startCount+1);
  elements['nav-studio'].emit('click');
  assert.equal(elements['full-overlay'].hidden,true,
    'a navigation transition cannot strand the canvas inside a hidden dialog');
  assert.equal(elements.expand.attributes['aria-expanded'],'false');
  assert.equal(elements['canvas-shell'].parentElement.id,'workspace');
  assert.equal(elements.whiteboard.hidden,true);
  assert.equal(document.activeElement,elements['nav-studio'],
    'navigation must not leave keyboard focus on a control in hidden Whiteboard');
  elements['nav-whiteboard'].emit('click');
  assert.equal(elements.drawing.children.length,startCount+1);
  elements.undo.emit('click');
  assert.equal(elements.drawing.children.length,startCount);
  elements.redo.emit('click');
  assert.equal(elements.drawing.children.length,startCount+1);
});

test('expanded Whiteboard exposes Camera/Clean and editable Save without duplicating the canvas',async()=>{
  const drawing=elements.drawing, preview=elements.preview;
  const saved=[];
  window.__TAURI__={core:{invoke:async(command,args)=>{
    saved.push({command,args});
    return '/tmp/expanded-drawing.handd.json';
  }}};
  elements.expand.emit('click');
  elements['modal-clean'].emit('click');
  assert.equal(elements['view-clean'].attributes['aria-pressed'],'true');
  assert.equal(elements['modal-clean'].attributes['aria-pressed'],'true');
  elements['modal-camera'].emit('click');
  assert.equal(elements['view-camera'].attributes['aria-pressed'],'true');
  assert.equal(elements['modal-camera'].attributes['aria-pressed'],'true');
  await elements['modal-save'].emit('click');
  assert.equal(saved.length,1);
  assert.equal(saved[0].command,'save_drawing');
  assert.doesNotMatch(JSON.stringify(saved[0].args),/camera|preview|jpeg|video/i);
  assert.equal(elements.drawing,drawing);
  assert.equal(elements.preview,preview);
  elements['close-expand'].emit('click');
});

test('sidecar startup failure remains visible until an explicit retry', async () => {
  window.__TAURI__ = {core: {invoke: async command => {
    if (command === 'sidecar_status') return {error:'RUNTIME FAILED - USE RESTART CAMERA'};
    if (command === 'restart_sidecar') return null;
    throw Error('Unexpected command: ' + command);
  }}};
  await timers[0]();
  assert.equal(elements.status.textContent, 'SIDECAR / RUNTIME FAILED - USE RESTART CAMERA');
  elements.restart.emit('click');
  await Promise.resolve();
  assert.equal(elements.status.textContent, 'RESTARTING SIDECAR…');
});

test('Whiteboard and Studio navigation does not erase canvas', () => {
  const drawingCount=elements.drawing.children.length;
  elements['nav-studio'].emit('click');
  assert.equal(elements.whiteboard.hidden, true);
  elements['nav-whiteboard'].emit('click');
  assert.equal(elements.whiteboard.hidden, false);
  assert.equal(elements.drawing.children.length, drawingCount);
});

test('Studio uses separate task tabs instead of forcing all panels into one screen',()=>{
  elements['nav-studio'].emit('click');
  elements['studio-tab-review'].emit('click');
  assert.equal(elements['studio-pane-review'].hidden,false);
  assert.equal(elements['studio-pane-collect'].hidden,true);
  assert.equal(elements['studio-tab-review'].attributes['aria-selected'],'true');
  elements['studio-tab-browse'].emit('click');
  assert.equal(elements['studio-pane-browse'].hidden,false);
  assert.equal(elements['studio-pane-review'].hidden,true);
  elements['studio-tab-runtime'].emit('click');
  assert.equal(elements['studio-pane-runtime'].hidden,false,
    'runtime health must remain accessible after splitting Studio views');
  elements['studio-tab-collect'].emit('click');
  elements['nav-whiteboard'].emit('click');
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
  assert.equal(elements.drawing.children.at(-1).children[0].attributes.stroke,'#5263e6');
  elements.undo.emit('click');
  elements.redo.emit('click');
  await Promise.resolve();
  assert.equal(elements.drawing.children.at(-1).children[0].attributes.stroke,'#5263e6');
});

test('connection READY distinguishes legacy model from unavailable model', async () => {
  const fixtureAuth='synthetic-test-token';
  const meta = {port: 39001, token: fixtureAuth, runtime_session_id: 'session-a'};
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
  const group=elements.drawing.children.at(-1);
  assert.equal(group.children.length,4);
  const path=group.children[0];
  const stable=path.attributes.d;
  const variantPaths=group.children.slice(1).map(node=>node.attributes.d);
  timers[1](); // Wiggly presentation timer; transport poll is timers[0].
  assert.notEqual(elements.drawing.attributes['data-boil-frame'],undefined);
  assert.equal(path.attributes.d,stable);
  assert.equal(new Set(variantPaths).size,3);
  reducedMotion=true;
  timers[1]();
  assert.equal(elements.drawing.attributes['data-boil-frame'],undefined);
  assert.equal(path.attributes.d,stable);
  reducedMotion=false;
  elements.drawing.emit('pointerup',{});
});

test('Studio uses explicit workspace and manual review/immutable snapshot commands',async()=>{
  elements['workspace-path'].value='/tmp/handd-user-workspace';
  const fixtureAuth='synthetic-token';
  let selected=null;
  window.__TAURI__ = {core:{invoke:async (command,args)=>{
    if(command==='select_workspace'){selected=args.path;return args.path;}
    if(command==='sidecar_status') return {
      port:39009,token:fixtureAuth,runtime_session_id:'studio-session',
      workspace:'/tmp/handd-user-workspace',
    };
    throw Error(command);
  }}};
  elements['browse-workspace'].emit('click');
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
      review_offset:0,review_page_size:40,
      samples:[{sample_id:'SAMPLE0',gesture:'Fist',review_status:'unreviewed',
                lifecycle_status:'active'}],snapshot_ready:true,eligible_count:1,
      snapshot_blockers:[],snapshot_warnings:[]}});
  assert.equal(elements['review-sample'].children.length,1);
  assert.equal(elements['workspace-active-label'].textContent,'handd-user-workspace');
  assert.equal(elements['review-next'].disabled,true);
  assert.match(elements['review-page'].textContent,/1 of 1/);
  assert.equal(socket.sent.at(-1).action,'sample_detail');
  assert.equal(socket.sent.at(-1).sample_id,'SAMPLE0');
  assert.equal(elements['review-accept'].disabled,true,
    'accept must wait for a valid stored landmark preview');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,ok:true,
    data:{sample_id:'SAMPLE0',gesture:'Fist',hand:'Right',raw_mp_handedness:'Left',
      review_status:'unreviewed',lifecycle_status:'active',
      image_landmarks:Array.from({length:21},(_,i)=>[.2+i*.02,.35,0]),
      world_landmarks:Array.from({length:21},(_,i)=>[i*.01,(i%5)*.02,(i%7)*.01]),
      canonical_landmarks:Array.from({length:21},(_,i)=>[i*.01,(i%5)*.02,(i%7)*.01])}});
  assert.equal(elements['review-hands'].children[0].style.display,'');
  assert.equal(elements['review-hands'].children[0].children.length,41);
  assert.match(elements['review-visual-status'].textContent,/Stored 21 joints/);
  assert.equal(elements['review-accept'].disabled,false);
  assert.equal(elements['review-world'].children[0].style.display,'');
  const beforeRotation=elements['review-world'].children[0].children[0].attributes.x1;
  elements['review-angle'].value='-60';
  elements['review-angle'].emit('input');
  assert.notEqual(elements['review-world'].children[0].children[0].attributes.x1,beforeRotation);
  elements['review-sample'].value='SAMPLE0';
  elements['review-accept'].emit('click');
  assert.equal(socket.sent.at(-1).action,'accept');
  assert.equal(socket.sent.at(-1).sample_id,'SAMPLE0');
  elements['build-snapshot'].emit('click');
  assert.equal(socket.sent.at(-1).action,'snapshot');
});

test('Studio Collect reuses the same authenticated camera preview and live landmarks',()=>{
  elements['nav-studio'].emit('click');
  const socket=connections.at(-1);
  elements['collect-preview'].naturalWidth=1280;
  elements['collect-preview'].naturalHeight=720;
  elements['collect-preview'].onload();
  assert.equal(elements['collect-visual'].style.aspectRatio,'1280/720');
  assert.equal(elements['collect-preview'].src,
    'http://127.0.0.1:39009/mjpeg?token=synthetic-token');
  assert.equal(elements.preview.getAttribute('src'),null,
    'the hidden Whiteboard should not consume a second MJPEG stream');
  const pts=Array.from({length:21},(_,i)=>({x:.2+i*.02,y:.35}));
  socket.sendEvent({type:'runtime.update',runtime_session_id:'studio-session',
    seq:1,timestamp_ms:100,payload:{drawing:{physical_hand:'Right',landmarks:pts,
      pointer:{x:.3,y:.35},raw_gesture:'Index_Finger'},modifier:{physical_hand:'Left'}}});
  assert.equal(elements['collect-hands'].children[0].style.display,'');
  assert.match(elements['collect-visual-status'].textContent,/Index_Finger/);
});

test('Studio camera tracking never draws accidentally on the hidden Whiteboard',()=>{
  const socket=connections.at(-1);
  const previous=elements.drawing.children.length;
  socket.sendEvent({type:'runtime.update',runtime_session_id:'studio-session',
    seq:2,timestamp_ms:180,payload:{
      drawing:{physical_hand:'Right',pointer:{x:.4,y:.5},action:'draw'},
      modifier:{physical_hand:'Left'},
    }});
  assert.equal(elements.drawing.children.length,previous);
  assert.equal(elements.cursor.hidden,true);
});

test('Review navigates beyond the first 40 samples instead of silently truncating',()=>{
  const socket=connections.at(-1);
  elements['refresh-dataset'].emit('click');
  const current=socket.sent.at(-1);
  assert.equal(current.action,'overview');
  socket.sendEvent({type:'studio.response',request_id:current.request_id,ok:true,
    data:{workspace:'/tmp/handd-user-workspace',sample_count:85,
      review_offset:0,review_page_size:40,review_counts:{unreviewed:85},
      samples:Array.from({length:40},(_,i)=>({sample_id:'S'+i,gesture:'Fist',
        review_status:'unreviewed',lifecycle_status:'active'}))}});
  assert.equal(elements['review-next'].disabled,false);
  elements['review-next'].emit('click');
  assert.equal(socket.sent.at(-1).action,'overview');
  assert.equal(socket.sent.at(-1).offset,40);
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,ok:true,
    data:{workspace:'/tmp/handd-user-workspace',sample_count:85,
      review_offset:40,review_page_size:40,review_counts:{unreviewed:85},
      samples:Array.from({length:40},(_,i)=>({sample_id:'S'+(40+i),
        gesture:'Fist',review_status:'unreviewed',lifecycle_status:'active'}))}});
  assert.equal(elements['review-sample'].children[0].value,'S40');
  assert.match(elements['review-page'].textContent,/41–80 of 85/);
});

test('Browse filters and deliberate QC gate batch accept; no invented model suggestions',()=>{
  const socket=connections.at(-1);
  elements['browse-gesture'].value='Fist';
  elements['browse-participant'].value='P001';
  elements['browse-review-status'].value='';
  elements['browse-apply'].emit('click');
  const filter=socket.sent.at(-1);
  assert.equal(filter.action,'overview');
  assert.equal(filter.gesture,'Fist');
  assert.equal(filter.participant,'P001');
  const overview=(statuses)=>({
    workspace:'/tmp/handd-user-workspace',sample_count:2,browse_total:2,
    review_offset:0,review_page_size:40,review_counts:{unreviewed:1,accepted:1},
    captures:[{capture_id:'C001',gesture:'Fist',participant_id:'P001',sample_count:2}],
    samples:statuses.map((review_status,i)=>({sample_id:'SAMPLE'+i,capture_id:'C001',
      gesture:'Fist',review_status,lifecycle_status:'active'})),
  });
  socket.sendEvent({type:'studio.response',request_id:filter.request_id,
    ok:true,data:overview(['unreviewed','unreviewed'])});
  const planRequest=socket.sent.filter(r=>r.action==='review_plan').at(-1);
  assert.equal(planRequest.capture_id,'C001');
  socket.sendEvent({type:'studio.response',request_id:planRequest.request_id,ok:true,
    data:{capture_id:'C001',pending:2,token:'v1',assessment_state:'no_model_assessment',
      suggested:[],qc_sample_ids:['SAMPLE0'],qc_unreviewed:['SAMPLE0'],qc_rejected:[],
      can_batch_accept:false}});
  assert.equal(elements['review-batch-accept'].disabled,true);
  assert.match(elements['review-queue-status'].textContent,/No verified model evidence/);
  elements['review-qc-open'].emit('click');
  assert.equal(socket.sent.at(-1).action,'sample_detail');
  assert.equal(socket.sent.at(-1).sample_id,'SAMPLE0');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,ok:true,
    data:{sample_id:'SAMPLE0',gesture:'Fist',hand:'Right',raw_mp_handedness:'Left',
      review_status:'unreviewed',lifecycle_status:'active',
      image_landmarks:Array.from({length:21},(_,i)=>[.1+i*.025,.3,0]),
      world_landmarks:Array.from({length:21},(_,i)=>[i*.01,(i%5)*.02,(i%7)*.01]),
      canonical_landmarks:Array.from({length:21},(_,i)=>[i*.01,(i%5)*.02,(i%7)*.01])}});
  elements['review-accept'].emit('click');
  const accept=socket.sent.at(-1);
  assert.equal(accept.action,'accept');
  socket.sendEvent({type:'studio.response',request_id:accept.request_id,
    ok:true,data:{sample_id:'SAMPLE0',review_status:'accepted'}});
  const refresh=socket.sent.at(-1);
  socket.sendEvent({type:'studio.response',request_id:refresh.request_id,ok:true,
    data:overview(['accepted','unreviewed'])});
  const newRequest=socket.sent.filter(r=>r.action==='review_plan').at(-1);
  const rejectedRevision='v1-rejected';
  socket.sendEvent({type:'studio.response',request_id:newRequest.request_id,ok:true,
    data:{capture_id:'C001',pending:1,token:rejectedRevision,assessment_state:'no_model_assessment',
      suggested:[],qc_sample_ids:['SAMPLE0'],qc_unreviewed:[],qc_rejected:['SAMPLE0'],
      can_batch_accept:false}});
  assert.equal(elements['review-batch-accept'].disabled,true);
  assert.match(elements['review-queue-status'].textContent,/batch acceptance blocked/);
  // After resolving a bad QC observation with a new documented human decision,
  // server supplies a new revision token before batch acceptance.
  elements['refresh-dataset'].emit('click');
  const qcRefresh=socket.sent.at(-1);
  socket.sendEvent({type:'studio.response',request_id:qcRefresh.request_id,ok:true,
    data:overview(['accepted','unreviewed'])});
  const clearRequest=socket.sent.filter(r=>r.action==='review_plan').at(-1);
  socket.sendEvent({type:'studio.response',request_id:clearRequest.request_id,ok:true,
    data:{capture_id:'C001',pending:1,token:'v2',assessment_state:'no_model_assessment',
      suggested:[],qc_sample_ids:['SAMPLE0'],qc_unreviewed:[],qc_rejected:[],can_batch_accept:true}});
  assert.equal(elements['review-batch-accept'].disabled,false);
  elements['review-batch-accept'].emit('click');
  assert.equal(socket.sent.at(-1).action,'batch_accept');
  assert.equal(socket.sent.at(-1).token,'v2');
});

test('Review keeps flagged predictions and sample inspection on the same screen',()=>{
  const socket=connections.at(-1);
  elements['review-capture'].emit('change');
  const next=socket.sent.at(-1);
  assert.equal(next.action,'review_plan');
  socket.sendEvent({type:'studio.response',request_id:next.request_id,ok:true,
    data:{capture_id:'C001',pending:2,token:'v3',assessment_state:'model_assessed',
      assessment_model_id:'candidate-test',assessed_count:2,
      suggested_pending:1,can_batch_accept:false,
      suggested:[{sample_id:'SAMPLE1',reason:'model_disagreement',
        predicted_label:'Idle',review_status:'unreviewed'}],
      qc_sample_ids:['SAMPLE0'],qc_unreviewed:[],qc_rejected:[]}});
  assert.match(elements['review-queue-status'].textContent,/1 flagged/);
  assert.equal(elements['review-batch-accept'].disabled,true);
  elements['review-suggested-open'].emit('click');
  assert.equal(elements['studio-pane-review'].hidden,false);
  assert.equal(elements['studio-pane-browse'].hidden,true);
  assert.equal(socket.sent.at(-1).action,'sample_detail');
  assert.equal(socket.sent.at(-1).sample_id,'SAMPLE1');
  const xyz=Array.from({length:21},(_,i)=>[i*.02,(i%5)*.02,(i%7)*.01]);
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{sample_id:'SAMPLE1',gesture:'Fist',hand:'Right',
      raw_mp_handedness:'Left',review_status:'unreviewed',lifecycle_status:'active',
      image_landmarks:xyz,world_landmarks:xyz,canonical_landmarks:xyz}});
  assert.equal(elements['review-accept'].disabled,false);
  const original=elements['review-world'].children[0].children[0].attributes.x1;
  elements['review-pitch'].value='50';
  elements['review-pitch'].emit('input');
  assert.notEqual(elements['review-world'].children[0].children[0].attributes.x1,original);
  elements['review-world'].emit('pointerdown',{clientX:50,clientY:50,pointerId:1});
  elements['review-world'].emit('pointermove',{clientX:90,clientY:70});
  elements['review-world'].emit('pointerup',{});
  assert.notEqual(Number(elements['review-angle'].value),25);
  elements['review-reset-view'].emit('click');
  assert.equal(elements['review-pitch'].value,'0');
  assert.equal(elements['review-angle'].value,'25');
});

test('historical accepted OOF flags remain inspectable and Accept confirms an audited acknowledgement',()=>{
  const fixturePageId='review-token';
  const socket=connections.at(-1);
  elements['review-capture'].value='C001';
  elements['review-capture'].emit('change');
  const request=socket.sent.at(-1);
  socket.sendEvent({type:'studio.response',request_id:request.request_id,ok:true,
    data:{capture_id:'C001',token:fixturePageId,assessment_state:'model_assessed',
      assessment_revision:'oof-revision',assessment_model_id:'candidate-test',
      assessed_count:1,historical_oof_count:1,suggested_pending:1,pending:0,
      can_batch_accept:false,qc_sample_ids:[],qc_unreviewed:[],qc_rejected:[],
      suggested:[{sample_id:'SAMPLE0',reason:'model_disagreement',
        predicted_label:'Idle',review_status:'accepted',
        assessment_origin:'heldout_session_oof'}]}});
  assert.equal(elements['review-suggested-sample'].value,'SAMPLE0');
  assert.match(elements['review-suggested-sample'].children[0].textContent,/already accepted/);
  elements['review-suggested-open'].emit('click');
  const detailRequest=socket.sent.at(-1);
  const xyz=Array.from({length:21},(_,i)=>[i*.02,(i%5)*.02,(i%7)*.01]);
  socket.sendEvent({type:'studio.response',request_id:detailRequest.request_id,ok:true,
    data:{sample_id:'SAMPLE0',gesture:'Fist',hand:'Right',
      raw_mp_handedness:'Left',review_status:'accepted',lifecycle_status:'active',
      image_landmarks:xyz,world_landmarks:xyz,canonical_landmarks:xyz}});
  assert.match(elements['review-accept'].textContent,/Confirm accepted \(OOF review\)/);
  elements['review-accept'].emit('click');
  const acknowledgement=socket.sent.at(-1);
  assert.equal(acknowledgement.action,'acknowledge');
  assert.equal(acknowledgement.sample_id,'SAMPLE0');
  assert.equal(acknowledgement.assessment_revision,'oof-revision');
  assert.equal(acknowledgement.capture_id,'C001');
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
    state:'capturing',count:0,target:2,participant:'P001',gesture:'Fist',
    capture_id:'C-NEW'
  }});
  assert.match(elements['collect-progress'].textContent,/0\s*\/\s*2/);
  socket.sendEvent({type:'studio.collection',data:{
    state:'capturing',count:0,target:2,capture_id:'C-NEW',
    quality_skips:{outside_frame:3,no_hand:2},last_skip_reason:'outside_frame',
  }});
  assert.match(elements['collect-quality-status'].textContent,/5 incomplete tracking/);
  assert.match(elements['collect-quality-status'].textContent,/Hand clipped/);
  elements['collect-pause'].emit('click');
  assert.equal(elements['collect-go-review'].disabled,true);
  assert.equal(elements['collect-progress-bar'].max,2);
  assert.equal(socket.sent.at(-1).action,'collect_pause');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{state:'paused',count:1,target:2}});
  elements['collect-resume'].emit('click');
  assert.equal(socket.sent.at(-1).action,'collect_resume');
  elements['collect-finish'].emit('click');
  assert.equal(socket.sent.at(-1).action,'collect_finish');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{state:'finished',capture_id:'C-NEW',count:2,target:2}});
  assert.equal(elements['collect-progress-bar'].value,2);
  assert.equal(elements['collect-go-review'].disabled,false);
  elements['collect-go-review'].emit('click');
  assert.equal(elements['studio-pane-review'].hidden,false);
  assert.equal(socket.sent.at(-1).action,'review_plan');
  assert.equal(socket.sent.at(-1).capture_id,'C-NEW');
});

test('Studio activates only selected compatible candidate, shows Active Model on Whiteboard',()=>{
  const socket=connections.at(-1);
  const before=socket.sent?.length||0;
  elements['models-refresh'].emit('click');
  assert.equal(socket.sent.length,before+1);
  assert.equal(socket.sent.at(-1).action,'models');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{active_model_id:null,selected_candidate_id:null,health:'legacy_unverified',
      candidates:[
        {artifact_id:'candidate-validated',compatible:true,labels:5,active:false},
        {artifact_id:'candidate-broken',compatible:false,reason:'bad checksum',active:false},
      ]}});
  assert.equal(elements['models-list'].children.length,2);
  elements['models-list'].value='candidate-broken';
  elements['models-activate'].emit('click');
  assert.notEqual(socket.sent.at(-1).action,'model_activate',
    'invalid Candidate never sent for activation');
  elements['models-list'].value='candidate-validated';
  elements['models-activate'].emit('click');
  assert.equal(socket.sent.at(-1).action,'model_activate');
  assert.equal(socket.sent.at(-1).artifact_id,'candidate-validated');
  socket.sendEvent({type:'studio.response',request_id:socket.sent.at(-1).request_id,
    ok:true,data:{artifact_id:'candidate-validated',label_order:['Fist','Index_Finger']}});
  socket.sendEvent({type:'studio.model',snapshot:{
    active_model_id:'candidate-validated',health:{model:'ready',camera:'tracking'}
  }});
  assert.match(elements['model-status'].textContent,/candidate-validated/);
  assert.match(elements['models-status'].textContent,/candidate-validated/);
});

test('Runtime visibly distinguishes experimental MPS and CPU fallback',()=>{
  const socket=connections.at(-1);
  socket.sendEvent({type:'studio.model',snapshot:{active_model_id:'candidate-validated',
    health:{model:'ready',camera:'tracking',inference_backend:'mps'}}});
  assert.match(elements['model-status'].textContent,/MPS \(experimental\)/);
  socket.sendEvent({type:'studio.model',snapshot:{active_model_id:'candidate-validated',
    health:{model:'error',camera:'tracking',inference_backend:'torch-cpu-mps-fallback'}}});
  assert.match(elements['model-status'].textContent,/CPU fallback from MPS/);
});

test('Save/Open preserves editable strokes; Export emits camera-free SVG; cancel preserves drawing',async()=>{
  const invoked=[];
  const saved={document:null};
  window.__TAURI__={core:{invoke:async(command,args)=>{
    invoked.push(command);
    if(command==='save_drawing'){saved.document=args.document;return '/tmp/ink.handd.json';}
    if(command==='open_drawing')return {path:'/tmp/ink.handd.json',document:saved.document};
    if(command==='export_svg'){
      assert.match(args.svg,/viewBox="0 0 1000 600"/);
      assert.doesNotMatch(args.svg,/<(image|video|script)/);
      return '/tmp/ink.svg';
    }
    throw Error(command);
  }}};
  const originalCount=elements.drawing.children.length;
  elements['save-drawing'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(saved.document);
  assert.equal(JSON.parse(saved.document).format,'handd-whiteboard');
  assert.equal(elements.drawing.children.length,originalCount);
  elements.clear.emit('click');
  assert.equal(elements.drawing.children.length,0);
  elements['open-drawing'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements.drawing.children.length,originalCount);
  elements['export-drawing'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(invoked.includes('export_svg'));
  assert.match(elements['document-status'].textContent,/ink.svg/);
});

test('Open/Create Workspace use native picker; typed path is explicit alternative',async()=>{
  const commands=[];
  window.__TAURI__={core:{invoke:async command=>{
    commands.push(command);
    return '/tmp/native-user-project';
  }}};
  elements['select-workspace'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(commands,['pick_workspace','select_workspace']);
  assert.equal(elements['workspace-path'].value,'/tmp/native-user-project');
  elements['create-workspace'].emit('click');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(commands.at(-1),'create_workspace');
});

test('native Open shows selected folder and validation failure instead of appearing to do nothing',async()=>{
  const old=window.__TAURI__;
  const previousPath=elements['workspace-path'].value;
  const calls=[];
  window.__TAURI__={core:{invoke:async (command,args)=>{
    calls.push([command,args]);
    if(command==='pick_workspace')return '/Users/test/Documents/New Project';
    if(command==='select_workspace')throw Error('Select a Hand-D workspace containing handd.sqlite');
    throw Error(command);
  }}};
  try {
    await elements['select-workspace'].emit('click');
    assert.equal(elements['workspace-path'].value,'/Users/test/Documents/New Project');
    assert.deepEqual(calls.map(row=>row[0]),['pick_workspace','select_workspace']);
    assert.match(elements['workspace-message'].textContent,/handd.sqlite/);
    assert.match(elements['workspace-message'].textContent,/Create workspace/i);
  } finally {window.__TAURI__=old;elements['workspace-path'].value=previousPath;}
});

test('native Open uses canonical Workspace path from Rust for restart matching',async()=>{
  const previous=window.__TAURI__;
  const previousPath=elements['workspace-path'].value;
  const fakeAuth='synthetic-canonical-session';
  window.__TAURI__={core:{invoke:async command=>{
    if(command==='pick_workspace')return '/var/folders/mac-alias';
    if(command==='select_workspace')return '/private/var/folders/mac-alias';
    if(command==='sidecar_status')return {port:39901,token:fakeAuth,
      runtime_session_id:'canonical-mac-test',workspace:'/private/var/folders/mac-alias'};
    return null;
  }}};
  try {
    await elements['select-workspace'].emit('click');
    assert.equal(elements['workspace-path'].value,'/private/var/folders/mac-alias');
    assert.match(elements['workspace-message'].textContent,/private\/var\/folders/);
    await timers[0]();
    assert.equal(typeof connections.at(-1).onmessage,'function');
  } finally {window.__TAURI__=previous;elements['workspace-path'].value=previousPath;}
});

test('missing native bridge reports an error instead of silently treating Open/Create as canceled',async()=>{
  const previous=window.__TAURI__;
  delete window.__TAURI__;
  try {
    elements['workspace-message'].textContent='Before opening';
    await elements['select-workspace'].emit('click');
    assert.match(elements['workspace-message'].textContent,/Tauri.*unavailable/i);
    elements['workspace-message'].textContent='Before creating';
    elements['create-workspace'].emit('click');
    await new Promise(resolve=>setImmediate(resolve));
    assert.match(elements['workspace-message'].textContent,/Tauri.*unavailable/i);
  } finally {
    window.__TAURI__=previous;
  }
});

test('workspace picker stays responsive and allows only one native dialog at a time',async()=>{
  const previous=window.__TAURI__;
  const previousPath=elements['workspace-path'].value;
  let resolvePicker;
  const commands=[];
  window.__TAURI__={core:{invoke:command=>{
    commands.push(command);
    return new Promise(resolve=>{resolvePicker=resolve;});
  }}};
  try {
    elements['select-workspace'].emit('click');
    assert.match(elements['workspace-message'].textContent,/opening.*picker/i);
    assert.equal(elements['select-workspace'].disabled,true);
    assert.equal(elements['create-workspace'].disabled,true);
    elements['select-workspace'].emit('click');
    elements['create-workspace'].emit('click');
    assert.deepEqual(commands,['pick_workspace'],
      'rapid clicks cannot spawn multiple macOS native pickers');
    resolvePicker(null);
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(elements['select-workspace'].disabled,false);
    assert.equal(elements['create-workspace'].disabled,false);
    assert.equal(elements['workspace-path'].value,previousPath,
      'canceling must preserve the selected workspace');
  } finally {
    window.__TAURI__=previous;
  }
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

test('large eraser does not enqueue overlapping 5px point updates every move',async()=>{
  const svg=elements.drawing;
  elements['tool-eraser'].emit('click');
  elements['eraser-size'].value='80';
  elements['eraser-size'].emit('input');
  svg.emit('pointerdown',{button:0,pointerId:14,clientX:100,clientY:250});
  svg.emit('pointermove',{clientX:105,clientY:250});
  svg.emit('pointermove',{clientX:115,clientY:250});
  svg.emit('pointerup',{});
  await Promise.resolve();
  const defs=svg.children.find(node=>node.id==='defs');
  const eraser=defs.children.at(-1).children.at(-1);
  assert.match(eraser.attributes.d,/L115\.00 250\.00/);
  assert.doesNotMatch(eraser.attributes.d,/L105\.00/,
    'large erasers do not need overlapping mask-paint updates at 5px intervals');
});

test('Pen and Wiggly use selectable colors and live brush thickness indicators',async()=>{
  elements['nav-whiteboard'].emit('click');
  elements['color-red'].emit('click');
  elements['brush-size'].value='12';
  elements['brush-size'].emit('input');
  elements['tool-pen'].emit('click');
  elements.drawing.emit('pointerdown',{button:0,pointerId:31,clientX:120,clientY:160});
  elements.drawing.emit('pointermove',{clientX:360,clientY:380});
  elements.drawing.emit('pointerup',{});
  const pen=elements.drawing.children.at(-1);
  assert.equal(pen.attributes.stroke,'#E24742');
  assert.equal(pen.attributes['stroke-width'],'12');
  elements['tool-wiggly'].emit('click');
  elements['color-green'].emit('click');
  elements.drawing.emit('pointerdown',{button:0,pointerId:32,clientX:150,clientY:140});
  elements.drawing.emit('pointermove',{clientX:380,clientY:310});
  elements.drawing.emit('pointerup',{});
  const wiggly=elements.drawing.children.at(-1);
  assert.equal(wiggly.children[0].attributes.stroke,'#188A75');
  assert.ok(wiggly.children.slice(1).every(node=>node.attributes.stroke==='#188A75'));
  assert.match(elements['brush-size-value'].textContent,/12/);
  assert.equal(elements['color-green'].attributes['aria-pressed'],'true');
});

test('brush and eraser cursors preview their actual SVG footprint, including live slider changes',()=>{
  elements['nav-whiteboard'].emit('click');
  elements['tool-pen'].emit('click');
  elements['brush-size'].value='10';
  elements['brush-size'].emit('input');
  const before=elements.drawing.children.length;
  elements.drawing.emit('pointermove',{clientX:250,clientY:300});
  assert.equal(elements.cursor.hidden,false);
  assert.equal(elements.cursor.style.left,'25%');
  assert.equal(elements.cursor.style.top,'50%');
  assert.equal(elements.cursor.style.width,'1%');
  assert.ok(Math.abs(parseFloat(elements.cursor.style.height)-(10/6))<0.001);
  assert.equal(elements.drawing.children.length,before,'hover preview never paints');
  elements['brush-size'].value='20';
  elements['brush-size'].emit('input');
  assert.equal(elements.cursor.style.width,'2%','slider immediately resizes hovered brush');
  assert.equal(elements['brush-size-preview'].style.width,'20px',
    'brush preview changes even while the cursor is off the canvas');
  elements['tool-wiggly'].emit('click');
  assert.equal(elements.cursor.style.width,'2%','Wiggly shares brush footprint');
  elements['color-green'].emit('click');
  assert.equal(elements.cursor.style.borderColor,'#188A75',
    'the hovered cursor immediately adopts the selected ink color');
  elements['tool-eraser'].emit('click');
  elements['eraser-size'].value='40';
  elements['eraser-size'].emit('input');
  assert.equal(elements.cursor.style.width,'8%','eraser radius 40 means diameter 80');
  assert.equal(elements['eraser-size-preview'].style.width,'10px',
    'compact eraser thumbnail scales with the real radius');
  assert.ok(Math.abs(parseFloat(elements.cursor.style.height)-(80/6))<0.001);
  elements.drawing.emit('pointerleave');
  assert.equal(elements.cursor.hidden,true);
});

test('two-hand ruler previews one editable straight segment and thumb modifier adjusts size',async()=>{
  elements['nav-whiteboard'].emit('click');
  const fixtureAuth='synthetic';
  window.__TAURI__={core:{invoke:async command=>{
    if(command==='sidecar_status')return {port:39107,token:fixtureAuth,runtime_session_id:'dual-hand-ui',
      workspace:elements['workspace-path'].value};
    throw Error(command);
  }}};
  await timers[0]();
  const socket=connections.at(-1);
  assert.equal(typeof socket.onmessage,'function','test must reach active live sidecar transport');
  socket.sendEvent({type:'runtime.ready',snapshot:{runtime_session_id:'dual-hand-ui',seq:0,timestamp_ms:0,health:{camera:'tracking',model:'legacy_unverified'}}});
  let seq=0;
  const send=(drawing,modifier)=>socket.sendEvent({type:'runtime.update',
    runtime_session_id:'dual-hand-ui',seq:++seq,timestamp_ms:seq*60,
    payload:{drawing:{physical_hand:'Right',...drawing},
      modifier:{physical_hand:'Left',...modifier},health:{camera:'tracking',model:'legacy_unverified'}}});
  elements['tool-pen'].emit('click');
  const initial=elements.drawing.children.length;
  send({action:'draw',pointer:{x:.2,y:.2}},
    {action:'ruler',pointer:{x:.5,y:.4}});
  assert.match(elements['modifier-feedback'].textContent,/ruler.*straight/i,
    'runtime gesture must reach the Whiteboard reducer');
  send({action:'draw',pointer:{x:.8,y:.7}},
    {action:'ruler',pointer:{x:.5,y:.4}});
  assert.equal(elements.drawing.children.length,initial+1);
  assert.match(elements['modifier-feedback'].textContent,/ruler.*straight/i);
  const ruler=elements.drawing.children.at(-1);
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(ruler.attributes.d,/M200\.00 120\.00 L800\.00 420\.00/);
  send({action:null,pointer:{x:.8,y:.7}}, {action:null,pointer:null});
  send({action:'draw',pointer:{x:.4,y:.5}},
    {action:'tool_adjust',pointer:{x:1,y:.2}});
  assert.match(elements['modifier-feedback'].textContent,/brush.*20/i);
  assert.equal(elements['brush-size'].value,'20');
  assert.equal(elements.cursor.style.width,'2%',
    'live gesture adjustment changes the projected pen width');
  send({action:'erase',pointer:{x:.5,y:.5}},
    {action:'tool_adjust',pointer:{x:0,y:.2}});
  assert.match(elements['modifier-feedback'].textContent,/eraser.*10/i);
  assert.equal(elements['eraser-size'].value,'10');
  const eraser=elements.drawing.children.at(-1);
  assert.equal(eraser.children.at(-1)?.attributes?.['stroke-width']??eraser.attributes['stroke-width'],'20');
  send({action:'erase',pointer:{x:.6,y:.5}},
    {action:'tool_adjust',pointer:{x:1,y:.2}});
  assert.equal(elements['eraser-size'].value,'80');
  assert.equal(elements.cursor.style.width,'16%',
    'live eraser footprint follows its radius, not the old 16px cursor');
});
