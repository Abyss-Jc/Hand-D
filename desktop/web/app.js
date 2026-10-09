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

function modelLabel(health) {
  switch (health?.model) {
    case 'ready': return 'MODELO V2: LISTO';
    case 'legacy_unverified':
      return 'MODELO ANTIGUO ACTIVO · ETIQUETAS SIN VALIDACIÓN INDEPENDIENTE';
    case 'error': return 'MODELO: ERROR DE INFERENCIA';
    case 'unavailable': return 'MODELO NO CARGADO · SOLO TRACKING';
    default: return 'MODELO: ESPERANDO SIDECAR';
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
      ? 'MANO DETECTADA · SIN CLASIFICACIÓN'
      : 'SIN MANO DETECTADA'
  );
  $('tracking').textContent = 'Dibujar: ' + (draw.physical_hand || 'Right')
    + ' / ' + (draw.action || 'sin acción') + ' · Modificar: '
    + (modifier.action || 'inactivo');
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
  $('toggle-hands').textContent = handsVisible ? '✋ Ocultar manos' : '✋ Mostrar manos';
}
$('view-camera').addEventListener('click',()=>setCameraMode(true));
$('view-clean').addEventListener('click',()=>setCameraMode(false));
$('toggle-hands').addEventListener('click',()=>setHandsVisible(!handsVisible));

// No animated geometry is stored. 12 Hz deliberately decouples visual wiggle
// from both MediaPipe and the browser's normal drawing render cadence.
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
    previewUrl = null;
    $('preview').removeAttribute('src');
    $('camera-fallback').hidden = false;
    $('footer-session').textContent = 'Sin sesión conectada';
    displayHealth(null);
    connectionState(false, 'SIDECAR INICIANDO / RECONECTANDO');
    return;
  }
  const host = '127.0.0.1:' + status.port;
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
  $('footer-session').textContent = 'SESIÓN ' + status.runtime_session_id.slice(0, 9);
  connectionState(false, 'CONECTANDO WEBSOCKET…');
  const socket = new WebSocket('ws://' + host + '/ws?token=' + token);
  ws = socket;
  socket.onmessage = message => {
    if (ws !== socket) return;
    try {
      const event = JSON.parse(message.data);
      if (event.type === 'runtime.ready') {
        gate.install(event.snapshot);
        displayHealth(event.snapshot?.health);
        connectionState(true, 'SIDECAR / CONECTADO');
      } else if (event.type === 'runtime.update') onRuntimeEvent(event);
    } catch (error) {
      console.error('Invalid runtime message', error);
    }
  };
  socket.onclose = () => {
    if (ws === socket) connectionState(false, 'SIDECAR RECONECTANDO');
  };
  socket.onerror = () => {
    if (ws === socket) connectionState(false, 'SIDECAR / ERROR DE TRANSPORTE');
  };
}
async function poll() {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) {
    connectionState(false, 'SE REQUIERE TAURI PARA USAR LA CÁMARA');
    return;
  }
  try { attachSidecar(await invoke('sidecar_status')); }
  catch (error) { console.error(error); attachSidecar(null); }
}
$('restart').addEventListener('click', async () => {
  try {
    connectionState(false, 'REINICIANDO SIDECAR…');
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
  strokeRenderer.animateWiggly((now-animationStarted)/340,{enabled:animate});
},85);
