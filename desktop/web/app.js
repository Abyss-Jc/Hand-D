import {RuntimeGate, Strokes} from './runtime-state.mjs';
import {StrokeRenderer} from './stroke-renderer.mjs';

const $ = id => document.getElementById(id);
const strokes = new Strokes();
const strokeRenderer = new StrokeRenderer($('drawing'));
const gate = new RuntimeGate();
let tool = 'draw';
let pointerStroke = null;
let gestureStroke = null;
let ws = null;
let sidecarKey = null;

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
  if (action !== 'draw' && action !== 'erase') return;
  tool = action;
  $('tool-pen').classList.toggle('selected', action === 'draw');
  $('tool-eraser').classList.toggle('selected', action === 'erase');
}
function releaseGesture() {
  gestureStroke = null;
  $('cursor').hidden = true;
}
function onRuntimeEvent(event) {
  if (!gate.accept(event)) return;
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
    releaseGesture();
    return;
  }
  const bounds = $('drawing').getBoundingClientRect();
  $('cursor').hidden = false;
  $('cursor').style.left = (point.x * bounds.width) + 'px';
  $('cursor').style.top = (point.y * bounds.height) + 'px';
  const action = draw.action;
  if (action !== 'draw' && action !== 'erase') {
    gestureStroke = null;
    return;
  }
  if (!gestureStroke || gestureStroke.tool !== action)
    gestureStroke = beginStroke(point, 'gesture', action);
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
});
function collapse() {
  if ($('full-overlay').hidden) return;
  normalParent.prepend($('canvas-shell'));
  $('full-overlay').hidden = true;
  $('expand').focus();
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
  preview.src = 'http://' + host + '/mjpeg?token=' + token;
  preview.onload = () => { $('camera-fallback').hidden = true; };
  preview.onerror = () => { $('camera-fallback').hidden = false; };
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
