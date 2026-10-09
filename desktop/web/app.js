import {RuntimeGate, Strokes} from './runtime-state.mjs';
import {StrokeRenderer} from './stroke-renderer.mjs';
import {HandOverlay,fitScene} from './hand-overlay.mjs';

const $ = id => document.getElementById(id);
const strokes = new Strokes();
const strokeRenderer = new StrokeRenderer($('drawing'));
const handOverlay = new HandOverlay($('hands-overlay'));
const gate = new RuntimeGate();
let tool = 'draw';
let cameraMode = true;
let handsVisible = true;
let pointerStroke = null;
let gestureStroke = null;
let ws = null;
let sidecarKey = null;
let previewUrl = null;
let pendingWorkspace = null;
let studioRequestNumber = 0;
const studioRequests = new Map();
let collectState = 'idle';

function modelLabel(health) {
  switch (health?.model) {
    case 'ready': return 'V2 MODEL: READY';
    case 'legacy_unverified':
      return 'LEGACY MODEL ACTIVE · LABELS NOT INDEPENDENTLY VERIFIED';
    case 'error': return 'MODEL: INFERENCE ERROR';
    case 'unavailable': return 'NO MODEL LOADED · TRACKING ONLY';
    default: return 'MODEL: WAITING FOR SIDECAR';
  }
}
function displayHealth(health) {
  $('model-status').textContent = modelLabel(health);
  $('diagnostics').textContent = JSON.stringify(health ?? {}, null, 2);
}

function redraw() {
  strokeRenderer.sync(strokes.paths);
}
function beginStroke(point, source, action) {
  const path = {points: [point], source, tool: action};
  strokes.paths.push(path);
  strokes.redoStack = [];
  redraw();
  return path;
}
function extendStroke(path, point) {
  if (!path) return;
  const last = path.points.at(-1);
  if (last && Math.hypot(last.x - point.x, last.y - point.y) < 0.002) return;
  path.points.push(point);
  strokeRenderer.pointAdded(path);
}
function setTool(action) {
  if (!['draw','erase','wiggly'].includes(action)) return;
  tool = action;
  $('tool-pen').classList.toggle('selected', action === 'draw');
  $('tool-eraser').classList.toggle('selected', action === 'erase');
  $('tool-wiggly').classList.toggle('selected', action === 'wiggly');
  $('tool-wiggly').setAttribute('aria-pressed',String(action === 'wiggly'));
}
function releaseGesture(clearHands = true) {
  gestureStroke = null;
  $('cursor').hidden = true;
  if (clearHands) handOverlay.clear();
}
function onRuntimeEvent(event) {
  if (!gate.accept(event)) return;
  handOverlay.update(event.payload);
  const draw = event.payload?.drawing ?? {};
  const modifier = event.payload?.modifier ?? {};
  $('gesture').textContent = draw.stable_gesture || draw.raw_gesture || (
    event.payload?.health?.camera === 'tracking'
      ? 'HAND DETECTED · NO CLASSIFICATION'
      : 'NO HAND DETECTED'
  );
  $('tracking').textContent = 'Drawing: ' + (draw.physical_hand || 'Right')
    + ' / ' + (draw.action || 'no action') + ' · Modifier: '
    + (modifier.action || 'inactive');
  displayHealth(event.payload?.health);
  const point = draw.pointer;
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)
      || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) {
    releaseGesture(false); // keep Modifier landmarks if Drawing Hand disappears
    return;
  }
  $('cursor').hidden = false;
  $('cursor').style.left = (point.x * 100) + '%';
  $('cursor').style.top = (point.y * 100) + '%';
  const action = draw.action;
  if (action !== 'draw' && action !== 'erase') {
    gestureStroke = null;
    return;
  }
  const effectiveTool = action === 'draw' && tool === 'wiggly' ? 'wiggly' : action;
  if (!gestureStroke || gestureStroke.tool !== effectiveTool)
    gestureStroke = beginStroke(point, 'gesture', effectiveTool);
  else
    extendStroke(gestureStroke, point);
}
function mousePoint(event) {
  const rect = $('drawing').getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}
const canvas = $('drawing');
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId);
  pointerStroke = beginStroke(mousePoint(event), 'pointer', tool);
});
canvas.addEventListener('pointermove', event => {
  if (pointerStroke) extendStroke(pointerStroke, mousePoint(event));
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture'])
  canvas.addEventListener(event, () => { pointerStroke = null; });
$('undo').addEventListener('click', () => { strokes.undo(); redraw(); });
$('redo').addEventListener('click', () => { strokes.redo(); redraw(); });
$('clear').addEventListener('click', () => { strokes.clear(); redraw(); });
$('tool-pen').addEventListener('click', () => setTool('draw'));
$('tool-eraser').addEventListener('click', () => setTool('erase'));
$('tool-wiggly').addEventListener('click', () => setTool('wiggly'));

// A single undistorted camera-plane coordinate system for the image, points,
// cursor and strokes; letterboxes stay outside this scene.
let cameraWidth = 640, cameraHeight = 480;
function layoutScene() {
  const canvasBox = $('canvas-shell').getBoundingClientRect();
  const fit = fitScene(canvasBox.width,canvasBox.height,cameraWidth,cameraHeight);
  const scene = $('camera-scene');
  scene.style.left = fit.left + 'px';
  scene.style.top = fit.top + 'px';
  scene.style.width = fit.width + 'px';
  scene.style.height = fit.height + 'px';
}
if (typeof ResizeObserver !== 'undefined')
  new ResizeObserver(layoutScene).observe($('canvas-shell'));
layoutScene();

function updatePreview() {
  const preview = $('preview');
  if (!cameraMode || !previewUrl) {
    preview.removeAttribute('src');
    return; // MJPEG disconnect allows Python to suspend its JPEG encoder
  }
  if (preview.getAttribute?.('src') !== previewUrl) preview.src = previewUrl;
}
function setCameraMode(enabled) {
  cameraMode = Boolean(enabled);
  $('canvas-shell').classList.toggle('clean',!cameraMode);
  $('view-camera').classList.toggle('selected',cameraMode);
  $('view-clean').classList.toggle('selected',!cameraMode);
  $('view-camera').setAttribute('aria-pressed',String(cameraMode));
  $('view-clean').setAttribute('aria-pressed',String(!cameraMode));
  updatePreview();
}
function setHandsVisible(enabled) {
  handsVisible = Boolean(enabled);
  handOverlay.setVisible(handsVisible);
  $('toggle-hands').classList.toggle('selected',handsVisible);
  $('toggle-hands').setAttribute('aria-pressed',String(handsVisible));
  $('toggle-hands').textContent = handsVisible ? '✋ Hide hands' : '✋ Show hands';
}
$('view-camera').addEventListener('click',()=>setCameraMode(true));
$('view-clean').addEventListener('click',()=>setCameraMode(false));
$('toggle-hands').addEventListener('click',()=>setHandsVisible(!handsVisible));

// Discrete ~12fps boil: three cached ink variants, one SVG root frame selector.
// No geometry work on the clock and no interference with MediaPipe input.
const reducedMotion = typeof matchMedia === 'function'
  ? matchMedia('(prefers-reduced-motion: reduce)') : {matches:false};
let animationStarted = typeof performance !== 'undefined' ? performance.now() : 0;

function navigate(view) {
  const whiteboard = view === 'whiteboard';
  $('whiteboard').hidden = !whiteboard;
  $('studio').hidden = whiteboard;
  $('nav-whiteboard').classList.toggle('selected', whiteboard);
  $('nav-studio').classList.toggle('selected', !whiteboard);
  $('nav-whiteboard').setAttribute('aria-selected', String(whiteboard));
  $('nav-studio').setAttribute('aria-selected', String(!whiteboard));
}
$('nav-whiteboard').addEventListener('click', () => navigate('whiteboard'));
$('nav-studio').addEventListener('click', () => navigate('studio'));

function studioRequest(action, fields = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN && ws.readyState !== 1) {
    $('workspace-message').textContent = 'The sidecar is not connected.';
    return;
  }
  const request_id = 'studio-' + (++studioRequestNumber);
  studioRequests.set(request_id, action);
  ws.send(JSON.stringify({type:'studio.request',request_id,action,...fields}));
}
function renderDataset(data) {
  const counts = data.review_counts || {};
  $('dataset-summary').textContent =
    (data.sample_count || 0) + ' samples · '
    + (counts.unreviewed || 0) + ' pending · '
    + (counts.accepted || 0) + ' accepted · '
    + (counts.rejected || 0) + ' rejected · '
    + (data.eligible_count || 0) + ' eligible for Development';
  const sampleSelect = $('review-sample');
  const oldValue = sampleSelect.value;
  sampleSelect.replaceChildren();
  const items = data.samples || [];
  for (const sample of items) {
    const option = document.createElement('option');
    option.value = sample.sample_id;
    option.textContent = sample.sample_id + ' · ' + sample.gesture
      + ' · ' + sample.review_status + ' / ' + sample.lifecycle_status;
    sampleSelect.append(option);
  }
  sampleSelect.value = items.some(sample=>sample.sample_id===oldValue)
    ? oldValue : (items[0]?.sample_id || '');
  $('build-snapshot').disabled = !data.snapshot_ready || !data.eligible_count;
  if (data.snapshot_blockers?.length) {
    $('snapshot-result').textContent = 'Blocked: ' + data.snapshot_blockers.join(', ');
  } else if (data.snapshot_warnings?.length) {
    $('snapshot-result').textContent = 'Warnings: ' + data.snapshot_warnings.join(', ');
  } else {
    $('snapshot-result').textContent = 'Ready for an explicit Development Snapshot.';
  }
}
function renderCollection(data) {
  if (!data || typeof data.state !== 'string') return;
  collectState = data.state;
  $('collect-progress').textContent =
    data.state.toUpperCase() + ' · ' + (data.count ?? 0)
    + ' / ' + (data.target ?? 0) + ' samples'
    + (data.gesture ? ' · ' + data.gesture : '')
    + (data.error ? ' · ' + data.error : '');
  $('collect-start').disabled = ['capturing','paused'].includes(data.state);
  $('collect-pause').disabled = data.state !== 'capturing';
  $('collect-resume').disabled = data.state !== 'paused';
  $('collect-finish').disabled = !['capturing','paused'].includes(data.state);
}
function handleStudioResponse(message) {
  const action = studioRequests.get(message.request_id);
  if (!action) return;
  studioRequests.delete(message.request_id);
  if (!message.ok) {
    const target = action === 'snapshot' ? $('snapshot-result')
      : action.startsWith('collect_') ? $('collect-progress') : $('workspace-message');
    target.textContent = message.error || 'Studio operation failed.';
    return;
  }
  if (action === 'overview') {
    $('workspace-message').textContent = 'Connected workspace: ' + message.data.workspace;
    renderDataset(message.data);
  } else if (action === 'snapshot') {
    $('snapshot-result').textContent = 'Snapshot created: '
      + message.data.snapshot_id + ' (' + message.data.workspace_relative_path + ')';
    studioRequest('overview');
  } else if (action.startsWith('collect_')) {
    renderCollection(message.data);
    if (['complete','finished'].includes(message.data.state)) studioRequest('overview');
  } else {
    $('workspace-message').textContent = 'Saved manual review for ' + message.data.sample_id;
    studioRequest('overview');
  }
}
$('select-workspace').addEventListener('click', async () => {
  const path = $('workspace-path').value.trim();
  if (!path) {
    $('workspace-message').textContent = 'Enter an existing workspace directory.';
    return;
  }
  try {
    const selected = await window.__TAURI__?.core?.invoke('select_workspace', {path});
    if (!selected) throw Error('Tauri is unavailable');
    pendingWorkspace = selected;
    $('workspace-message').textContent = 'Connecting workspace: ' + selected;
    attachSidecar(null);
  } catch (error) {
    $('workspace-message').textContent = String(error);
  }
});
$('refresh-dataset').addEventListener('click', () => studioRequest('overview'));
for (const action of ['accept','reject','drop','restore']) {
  $('review-' + action).addEventListener('click', () => {
    const sample_id = $('review-sample').value;
    if (sample_id) studioRequest(action, {sample_id});
  });
}
$('build-snapshot').addEventListener('click', () => {
  if ($('build-snapshot').disabled) return;
  if (typeof window.confirm === 'function'
      && !window.confirm('Create a new immutable Development Snapshot now?')) return;
  studioRequest('snapshot', {note:'Created explicitly from Hand-D Studio'});
});
// Start/Pause/Resume/Finish are the only gates for durable Sample writes.
// The server validates all parameters; frontend never creates Samples.
$('collect-start').addEventListener('click', () => {
  studioRequest('collect_start', {
    participant:$('collect-participant').value,
    gesture:$('collect-gesture').value.trim(),
    hand:$('collect-hand').value,
    target:Number($('collect-target').value),
    interval_ms:Number($('collect-interval').value),
  });
});
for (const action of ['pause','resume','finish']) {
  $('collect-' + action).addEventListener('click', () => {
    studioRequest('collect_' + action);
  });
}

const normalParent = $('canvas-shell').parentElement;
$('expand').addEventListener('click', () => {
  $('modal-canvas-slot').append($('canvas-shell'));
  $('full-overlay').hidden = false;
  $('close-expand').focus();
  layoutScene();
});
function collapse() {
  if ($('full-overlay').hidden) return;
  normalParent.prepend($('canvas-shell'));
  $('full-overlay').hidden = true;
  $('expand').focus();
  layoutScene();
}
$('close-expand').addEventListener('click', collapse);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') collapse();
});

function connectionState(ready, label) {
  $('lamp').classList.toggle('ready', ready);
  $('status').textContent = label;
  if (!ready) releaseGesture();
}
function attachSidecar(status) {
  const key = status ? [status.port, status.token, status.runtime_session_id].join(':') : null;
  if (key === sidecarKey && ws && ws.readyState !== WebSocket.CLOSED) return;
  if (ws) {
    ws.onclose = null;
    ws.onerror = null;
    ws.onmessage = null;
    ws.close();
    ws = null;
  }
  sidecarKey = key;
  releaseGesture();
  if (!status) {
    studioRequests.clear();
    renderCollection({state:'idle',count:0,target:0});
    previewUrl = null;
    $('preview').removeAttribute('src');
    $('camera-fallback').hidden = false;
    $('footer-session').textContent = 'No sidecar session';
    displayHealth(null);
    connectionState(false, 'STARTING / RECONNECTING SIDECAR');
    return;
  }
  const host = '127.0.0.1:' + status.port;
  if (status.workspace) $('workspace-path').value = status.workspace;
  const token = encodeURIComponent(status.token);
  const preview = $('preview');
  previewUrl = 'http://' + host + '/mjpeg?token=' + token;
  preview.onload = () => {
    $('camera-fallback').hidden = true;
    if (preview.naturalWidth && preview.naturalHeight
        && (cameraWidth !== preview.naturalWidth
            || cameraHeight !== preview.naturalHeight)) {
      cameraWidth = preview.naturalWidth;
      cameraHeight = preview.naturalHeight;
      layoutScene();
    }
  };
  preview.onerror = () => { $('camera-fallback').hidden = false; };
  updatePreview();
  $('footer-session').textContent = 'SESSION ' + status.runtime_session_id.slice(0, 9);
  connectionState(false, 'CONNECTING WEBSOCKET…');
  const socket = new WebSocket('ws://' + host + '/ws?token=' + token);
  ws = socket;
  socket.onmessage = message => {
    if (ws !== socket) return;
    try {
      const event = JSON.parse(message.data);
      if (event.type === 'runtime.ready') {
        gate.install(event.snapshot);
        displayHealth(event.snapshot?.health);
        connectionState(true, 'SIDECAR / CONNECTED');
        if (status.workspace) studioRequest('overview');
        if (status.workspace) studioRequest('collect_status');
        else $('workspace-message').textContent =
          'No workspace selected. Open an existing project to review samples.';
      } else if (event.type === 'runtime.update') onRuntimeEvent(event);
      else if (event.type === 'studio.response') handleStudioResponse(event);
      else if (event.type === 'studio.collection') {
        renderCollection(event.data);
        if (event.data?.state === 'complete') studioRequest('overview');
      }
    } catch (error) {
      console.error('Invalid runtime message', error);
    }
  };
  socket.onclose = () => {
    if (ws === socket) connectionState(false, 'SIDECAR RECONNECTING');
  };
  socket.onerror = () => {
    if (ws === socket) connectionState(false, 'SIDECAR / CONNECTION ERROR');
  };
}
async function poll() {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) {
    connectionState(false, 'TAURI IS REQUIRED FOR CAMERA ACCESS');
    return;
  }
  try {
    const status = await invoke('sidecar_status');
    if (pendingWorkspace && status?.workspace !== pendingWorkspace) {
      attachSidecar(null);
      return;
    }
    if (pendingWorkspace && status?.workspace === pendingWorkspace) pendingWorkspace = null;
    attachSidecar(status);
  }
  catch (error) { console.error(error); attachSidecar(null); }
}
$('restart').addEventListener('click', async () => {
  try {
    connectionState(false, 'RESTARTING SIDECAR…');
    await window.__TAURI__?.core?.invoke('restart_sidecar');
  } catch (error) { console.error(error); }
});
setInterval(poll, 1200);
poll();
setInterval(()=>{
  const animate = !reducedMotion.matches
    && !$('whiteboard').hidden
    && (typeof document.hidden === 'undefined' || !document.hidden);
  const now = typeof performance !== 'undefined' ? performance.now() : 0;
  strokeRenderer.animateWiggly(now-animationStarted,{enabled:animate});
},83);
