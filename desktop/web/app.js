import {RuntimeGate, Strokes} from './runtime-state.mjs';
import {StrokeRenderer} from './stroke-renderer.mjs';
import {HandOverlay,fitScene} from './hand-overlay.mjs';
import {projectWorldLandmarks} from './world-landmarks.mjs';
import {encodeWhiteboard,decodeWhiteboard,exportCleanSvg} from './drawing-document.mjs';

const $ = id => document.getElementById(id);
const strokes = new Strokes();
const strokeRenderer = new StrokeRenderer($('drawing'));
const handOverlay = new HandOverlay($('hands-overlay'));
const collectOverlay = new HandOverlay($('collect-hands'));
const reviewOverlay = new HandOverlay($('review-hands'));
const worldOverlay = new HandOverlay($('review-world'));
let selectedReviewWorld = null;
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
let reviewOffset = 0;
let qualityPlan = null;
let selectedReviewCanonical = null;
let reviewSamples = [];
let reviewNextPage = false;
let pendingReviewPage = null;
let collectState = 'idle';
let latestCaptureId = null;
let activeModelId = null;
let modelChoices = new Map();
let drawingDirty = false;

function modelLabel(health) {
  switch (health?.model) {
    case 'ready': return 'V2 MODEL: READY'
      + (activeModelId ? ' · ' + activeModelId : '');
    case 'legacy_unverified':
      return 'LEGACY MODEL ACTIVE · LABELS NOT INDEPENDENTLY VERIFIED';
    case 'error': return 'MODEL: LOAD / INFERENCE ERROR';
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
function markDrawingEdited() {
  drawingDirty = true;
  $('document-status').textContent = 'Unsaved drawing changes · video is never saved.';
}
function beginStroke(point, source, action) {
  const path = {points: [point], source, tool: action};
  strokes.paths.push(path);
  strokes.redoStack = [];
  redraw();
  markDrawingEdited();
  return path;
}
function extendStroke(path, point) {
  if (!path) return;
  const last = path.points.at(-1);
  if (last && Math.hypot(last.x - point.x, last.y - point.y) < 0.002) return;
  path.points.push(point);
  strokeRenderer.pointAdded(path);
  markDrawingEdited();
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
  collectOverlay.update(event.payload);
  const tracked = ['drawing','modifier'].filter(role=>event.payload?.[role]?.landmarks?.length===21);
  $('collect-visual-status').textContent = tracked.length
    ? tracked.map(role=>role+' hand: '+(event.payload[role].raw_gesture || 'tracking')).join(' · ')
    : 'No tracked hand';
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
  if ($('whiteboard').hidden) {
    // Studio observes gestures but must never edit the hidden Whiteboard.
    releaseGesture(false);
    return;
  }
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
$('undo').addEventListener('click', () => { strokes.undo(); redraw(); markDrawingEdited(); });
$('redo').addEventListener('click', () => { strokes.redo(); redraw(); markDrawingEdited(); });
$('clear').addEventListener('click', () => { strokes.clear(); redraw(); markDrawingEdited(); });
$('tool-pen').addEventListener('click', () => setTool('draw'));
$('tool-eraser').addEventListener('click', () => setTool('erase'));
$('tool-wiggly').addEventListener('click', () => setTool('wiggly'));
// Native dialogs and atomic persistence are implemented on Rust's local OS
// boundary. The camera frame never enters a file payload.
$('save-drawing').addEventListener('click',async()=>{
  try {
    const document=encodeWhiteboard(strokes.paths);
    const path=await window.__TAURI__?.core?.invoke('save_drawing',{document});
    if (path) {
      drawingDirty=false;
      $('document-status').textContent='Saved editable drawing: '+path;
    }
  } catch(error) {
    $('document-status').textContent='Could not save drawing: '+String(error);
  }
});
$('open-drawing').addEventListener('click',async()=>{
  if (strokes.paths.length && typeof window.confirm==='function'
      && !window.confirm('Replace your current drawing? Save any changes first.'))return;
  try {
    const result=await window.__TAURI__?.core?.invoke('open_drawing');
    if (!result) return;
    const paths=decodeWhiteboard(result.document);
    releaseGesture();
    pointerStroke=null;
    strokes.paths=paths;
    strokes.redoStack=[];
    redraw();
    drawingDirty=false;
    $('document-status').textContent='Opened editable drawing: '+result.path;
  } catch(error) {
    $('document-status').textContent='Could not open drawing: '+String(error);
  }
});
$('export-drawing').addEventListener('click',async()=>{
  try{
    const svg=exportCleanSvg(strokes.paths);
    const path=await window.__TAURI__?.core?.invoke('export_svg',{svg});
    if(path)$('document-status').textContent='Exported clean SVG (no camera): '+path;
  }catch(error){
    $('document-status').textContent='Could not export clean drawing: '+String(error);
  }
});

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
  if (!cameraMode || !previewUrl || $('whiteboard').hidden) {
    preview.removeAttribute('src');
  } else if (preview.getAttribute?.('src') !== previewUrl) {
    preview.src = previewUrl;
  }
  const collect=$('collect-preview');
  if (!previewUrl || $('studio').hidden) {
    collect.removeAttribute('src');
  } else if (collect.getAttribute?.('src') !== previewUrl) {
    collect.src=previewUrl;
  }
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
  updatePreview(); // At most one MJPEG consumer; camera inference keeps running.
}
$('nav-whiteboard').addEventListener('click', () => navigate('whiteboard'));
$('nav-studio').addEventListener('click', () => navigate('studio'));
function selectStudioTab(tab) {
  for (const name of ['collect','browse','review','snapshots','models','runtime']) {
    $('studio-pane-'+name).hidden=name!==tab;
    $('studio-tab-'+name).setAttribute('aria-selected',String(name===tab));
  }
}
for (const tab of ['collect','browse','review','snapshots','models','runtime'])
  $('studio-tab-'+tab).addEventListener('click',()=>selectStudioTab(tab));

function studioRequest(action, fields = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN && ws.readyState !== 1) {
    $('workspace-message').textContent = 'The sidecar is not connected.';
    return;
  }
  const request_id = 'studio-' + (++studioRequestNumber);
  studioRequests.set(request_id, action);
  ws.send(JSON.stringify({type:'studio.request',request_id,action,...fields}));
}
function browseFilters() {
  return {
    gesture:$('browse-gesture').value.trim()||null,
    participant:$('browse-participant').value||null,
    review_status:$('browse-review-status').value||null,
  };
}
function requestBrowse(offset=0) {
  studioRequest('overview',{offset,...browseFilters()});
}
function renderDataset(data) {
  $('workspace-active-label').textContent =
    data.workspace?.split(/[\\/]/).filter(Boolean).at(-1)||'Workspace open';
  reviewOffset=data.review_offset??0;
  const pageSize=data.review_page_size??40;
  const total=data.browse_total??data.sample_count??0;
  const shown=(data.samples??[]).length;
  reviewNextPage=reviewOffset+shown<total;
  $('review-prev').disabled=reviewOffset===0;
  $('review-next').disabled=reviewOffset+shown>=total;
  $('review-page').textContent=shown
    ? (reviewOffset+1)+'–'+(reviewOffset+shown)+' of '+total
    : 'No samples on this page';
  const counts = data.review_counts || {};
  $('dataset-summary').textContent =
    (data.sample_count || 0) + ' samples · '
    + (counts.unreviewed || 0) + ' pending · '
    + (counts.accepted || 0) + ' accepted · '
    + (counts.rejected || 0) + ' rejected · '
    + (data.eligible_count || 0) + ' eligible for Development · '
    + total + ' match filters';
  const captureSelect=$('review-capture');
  const oldCapture=captureSelect.value;
  captureSelect.replaceChildren();
  for(const capture of data.captures||[]) {
    const option=document.createElement('option');
    option.value=capture.capture_id;
    option.textContent=capture.participant_id+' / '+capture.gesture
      +' / '+capture.sample_count+' samples · '+capture.capture_id.slice(0,8);
    captureSelect.append(option);
  }
  captureSelect.value=(data.captures||[]).some(c=>c.capture_id===oldCapture)
    ? oldCapture:(data.captures?.[0]?.capture_id||'');
  requestQualityPlan();
  const sampleSelect = $('review-sample');
  const oldValue = sampleSelect.value;
  sampleSelect.replaceChildren();
  const items = data.samples || [];
  reviewSamples = items.map(item=>item.sample_id);
  const scrub=$('review-sample-scrub');
  scrub.max=String(Math.max(0,reviewSamples.length-1));
  scrub.disabled=reviewSamples.length<2;
  for (const sample of items) {
    const option = document.createElement('option');
    option.value = sample.sample_id;
    option.textContent = sample.sample_id + ' · ' + sample.gesture
      + ' · ' + sample.review_status + ' / ' + sample.lifecycle_status;
    sampleSelect.append(option);
  }
  sampleSelect.value = pendingReviewPage
    ? (pendingReviewPage==='last'?items.at(-1)?.sample_id:items[0]?.sample_id)||''
    : items.some(sample=>sample.sample_id===oldValue)
      ? oldValue : (items[0]?.sample_id || '');
  pendingReviewPage=null;
  requestReviewSample();
  $('build-snapshot').disabled = !data.snapshot_ready || !data.eligible_count;
  if (data.snapshot_blockers?.length) {
    $('snapshot-result').textContent = 'Blocked: ' + data.snapshot_blockers.join(', ');
  } else if (data.snapshot_warnings?.length) {
    $('snapshot-result').textContent = 'Warnings: ' + data.snapshot_warnings.join(', ');
  } else {
    $('snapshot-result').textContent = 'Ready for an explicit Development Snapshot.';
  }
}
$('review-prev').addEventListener('click',()=>{
  if (reviewOffset>0) requestBrowse(Math.max(0,reviewOffset-40));
});
$('review-next').addEventListener('click',()=>{
  if (!$('review-next').disabled) requestBrowse(reviewOffset+40);
});
$('browse-apply').addEventListener('click',()=>requestBrowse(0));
$('browse-inspect').addEventListener('click',()=>{
  if (!$('review-sample').value) return;
  selectStudioTab('review');
  requestReviewSample();
});
function requestQualityPlan() {
  qualityPlan=null;
  $('review-batch-accept').disabled=true;
  $('review-qc-sample').replaceChildren();
  $('review-suggested-sample').replaceChildren();
  const capture_id=$('review-capture').value;
  if(!capture_id) {
    $('review-queue-status').textContent='No Development Capture selected.';
    return;
  }
  $('review-queue-status').textContent='Loading Quality Check…';
  studioRequest('review_plan',{capture_id});
}
$('review-capture').addEventListener('change',requestQualityPlan);
function renderQualityPlan(data) {
  if(!data||data.capture_id!==$('review-capture').value)return;
  qualityPlan=data;
  const suggested=$('review-suggested-sample');
  suggested.replaceChildren();
  for (const row of (data.suggested||[]).filter(r=>r.review_status!=='accepted'
                                                 && r.review_status!=='rejected')) {
    const option=document.createElement('option');
    option.value=row.sample_id;
    option.textContent=row.sample_id+' · '+row.reason.replaceAll('_',' ')
      +' · '+row.predicted_label;
    suggested.append(option);
  }
  suggested.value=suggested.children[0]?.value||'';
  if (!suggested.children.length) {
    const option=document.createElement('option');
    option.value='';
    option.textContent='No model-flagged observations';
    suggested.append(option);
  }
  const select=$('review-qc-sample');
  select.replaceChildren();
  for(const id of data.qc_sample_ids||[]) {
    const option=document.createElement('option');
    option.value=id;
    option.textContent=id+(data.qc_unreviewed?.includes(id)?' · needs review':' · reviewed');
    select.append(option);
  }
  select.value=data.qc_sample_ids?.[0]||'';
  $('review-batch-accept').disabled=!data.can_batch_accept;
  const qcRejected=data.qc_rejected?.length||0;
  const evidence=data.assessment_state==='model_assessed'
    ? (data.assessment_model_id+' · '+(data.assessed_count||0)
      +' out-of-training assessed · '+(data.suggested_pending||0)+' flagged to inspect')
    : data.assessment_state==='assessment_unavailable'
      ? 'Model assessment unavailable: '+(data.assessment_error||'invalid evidence')
      : data.assessment_state==='unsupported_gesture'
        ? 'Gesture is not in the Active Model — QC-only bootstrap'
      : 'No verified model evidence yet — QC-only bootstrap';
  $('review-queue-status').textContent=(data.pending||0)+' pending · '
    +(data.qc_unreviewed?.length||0)+' QC observations need review · '+evidence+'. '
    +(data.unassessable_pending ? data.unassessable_pending+' invalid geometry observations need individual review. ' : '')
    +(qcRejected ? qcRejected+' QC rejected: inspect this Capture individually; batch acceptance blocked.' : '');
}
function inspectSample(sample_id) {
  if(!sample_id)return;
  const select=$('review-sample');
  if(![...select.children].some(option=>option.value===sample_id)) {
    const option=document.createElement('option');
    option.value=sample_id;
    option.textContent=sample_id+' · Quality Check';
    select.append(option);
  }
  select.value=sample_id;
  selectStudioTab('review');
  requestReviewSample();
}
$('review-qc-open').addEventListener('click',()=>inspectSample($('review-qc-sample').value));
$('review-suggested-open').addEventListener('click',()=>inspectSample($('review-suggested-sample').value));
$('review-batch-accept').addEventListener('click',()=>{
  const plan=qualityPlan;
  if(!plan?.can_batch_accept||plan.capture_id!==$('review-capture').value)return;
  if(typeof window.confirm==='function'&&!window.confirm(
    'You reviewed the QC subset and confirmed the intended gesture while collecting. '
    +'Accept the remaining unreviewed, active observations in this Capture? '
    +'Every acceptance is recorded in the audit trail.'))return;
  $('review-batch-accept').disabled=true;
  studioRequest('batch_accept',{capture_id:plan.capture_id,token:plan.token});
});
function requestReviewSample() {
  const sample_id=$('review-sample').value;
  reviewOverlay.clear();
  worldOverlay.clear();
  selectedReviewWorld=null;
  selectedReviewCanonical=null;
  $('review-accept').disabled=true;
  const position=reviewSamples.indexOf(sample_id);
  $('review-sample-scrub').value=String(Math.max(0,position));
  $('review-sample-position').textContent=sample_id
    ? sample_id+(position>=0?' · '+(reviewOffset+position+1)+' / filtered results':' · flagged/QC')
    : 'No sample selected';
  $('review-visual-status').textContent=sample_id
    ? 'Loading stored landmarks for '+sample_id : 'No sample selected';
  if (sample_id) studioRequest('sample_detail',{sample_id});
}
$('review-sample').addEventListener('change',requestReviewSample);
function stepReview(direction) {
  const current=reviewSamples.indexOf($('review-sample').value);
  const next=current+direction;
  if (next<0&&reviewOffset>0) {
    pendingReviewPage='last';
    requestBrowse(Math.max(0,reviewOffset-40));
    return;
  }
  if (next>=reviewSamples.length&&reviewNextPage) {
    pendingReviewPage='first';
    requestBrowse(reviewOffset+40);
    return;
  }
  if (next<0||next>=reviewSamples.length)return;
  inspectSample(reviewSamples[next]);
}
$('review-sample-prev').addEventListener('click',()=>stepReview(-1));
$('review-sample-next').addEventListener('click',()=>stepReview(1));
$('review-sample-scrub').addEventListener('change',()=>{
  const sample_id=reviewSamples[Number($('review-sample-scrub').value)];
  if(sample_id)inspectSample(sample_id);
});
function renderReviewWorld() {
  worldOverlay.clear();
  const selected=$('review-world-mode').value==='raw'
    ? selectedReviewWorld : selectedReviewCanonical;
  const points=projectWorldLandmarks(selected,Number($('review-angle').value),
                                    Number($('review-pitch').value),
                                    $('review-world-mode').value==='canonical');
  if (points) worldOverlay.update({drawing:{landmarks:points}});
}
$('review-angle').addEventListener('input',renderReviewWorld);
$('review-pitch').addEventListener('input',renderReviewWorld);
$('review-world-mode').addEventListener('change',renderReviewWorld);
$('review-reset-view').addEventListener('click',()=>{
  $('review-angle').value='25';
  $('review-pitch').value='0';
  renderReviewWorld();
});
let dragReview=null;
$('review-world').addEventListener('pointerdown',event=>{
  if (!Number.isFinite(event.clientX)||!Number.isFinite(event.clientY))return;
  dragReview={x:event.clientX,y:event.clientY};
  $('review-world').setPointerCapture?.(event.pointerId);
});
$('review-world').addEventListener('pointermove',event=>{
  if (!dragReview)return;
  const dx=event.clientX-dragReview.x,dy=event.clientY-dragReview.y;
  dragReview={x:event.clientX,y:event.clientY};
  $('review-angle').value=String(Math.max(-180,Math.min(180,
    Number($('review-angle').value)+dx*.5)));
  $('review-pitch').value=String(Math.max(-90,Math.min(90,
    Number($('review-pitch').value)+dy*.5)));
  renderReviewWorld();
});
for (const type of ['pointerup','pointercancel'])
  $('review-world').addEventListener(type,()=>{dragReview=null;});
function renderSampleDetail(data) {
  if (!data || data.sample_id !== $('review-sample').value) return;
  const points=data.image_landmarks;
  const valid=Array.isArray(points)&&points.length===21
    && points.every(p=>Array.isArray(p)&&p.length===3
      && Number.isFinite(p[0])&&Number.isFinite(p[1])
      && p[0]>=0&&p[0]<=1&&p[1]>=0&&p[1]<=1);
  reviewOverlay.clear();
  if (!valid) {
    $('review-visual-status').textContent='Invalid stored image landmarks; do not accept blindly.';
    return;
  }
  const canonical=data.canonical_landmarks;
  const canonicalValid=Array.isArray(canonical)&&canonical.length===21&&
    canonical.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite));
  $('review-accept').disabled=!canonicalValid;
  const role=data.raw_mp_handedness==='Left'?'drawing':'modifier';
  reviewOverlay.update({[role]:{landmarks:points.map(p=>({x:p[0],y:p[1]}))}});
  selectedReviewWorld=data.world_landmarks;
  selectedReviewCanonical=canonicalValid?canonical:null;
  renderReviewWorld();
  $('review-visual-status').textContent='Stored 21 joints · '+data.gesture
    +' · '+data.hand+' hand · '+data.review_status+' / '+data.lifecycle_status
    +(canonicalValid ? ' · canonical view available'
      : ' · invalid canonical geometry: reject or drop, not Accept')
    +' · not a camera photo';
}
function renderCollection(data) {
  if (!data || typeof data.state !== 'string') return;
  collectState = data.state;
  latestCaptureId = data.capture_id || latestCaptureId;
  const count=Math.max(0,Number(data.count)||0),target=Math.max(1,Number(data.target)||1);
  $('collect-progress-bar').max=target;
  $('collect-progress-bar').value=Math.min(count,target);
  $('collect-go-review').disabled=!(latestCaptureId&&
    ['complete','finished'].includes(data.state)&&count>0);
  $('collect-progress').textContent =
    data.state.toUpperCase() + ' · ' + (data.count ?? 0)
    + ' / ' + (data.target ?? 0) + ' samples'
    + (data.gesture ? ' · ' + data.gesture : '')
    + (data.error ? ' · ' + data.error : '');
  const reasons = data.quality_skips || {};
  const rejected=(reasons.no_hand||0)+(reasons.bad_tracking||0)
    +(reasons.outside_frame||0)+(reasons.wrong_hand||0);
  const hints={
    no_hand:'No hand detected. Move into the camera view.',
    outside_frame:'Hand clipped by camera edge. Center the entire hand.',
    bad_tracking:'Tracking is incomplete. Hold the hand steady briefly.',
    wrong_hand:'Use the selected physical hand.',
    sampling_interval:'Waiting for the sampling interval (normal).',
    stale:'Outdated observation ignored (normal).',
  };
  $('collect-quality-status').textContent = rejected+' incomplete tracking results skipped, not saved. '
    +(hints[data.last_skip_reason]||'Valid observations count toward the target.');
  $('collect-start').disabled = ['capturing','paused'].includes(data.state);
  $('collect-pause').disabled = data.state !== 'capturing';
  $('collect-resume').disabled = data.state !== 'paused';
  $('collect-finish').disabled = !['capturing','paused'].includes(data.state);
}
$('collect-go-review').addEventListener('click',()=>{
  if ($('collect-go-review').disabled||!latestCaptureId)return;
  const select=$('review-capture');
  if(![...select.children].some(row=>row.value===latestCaptureId)){
    const option=document.createElement('option');
    option.value=latestCaptureId;
    option.textContent='Just captured · '+latestCaptureId.slice(0,8);
    select.append(option);
  }
  select.value=latestCaptureId;
  selectStudioTab('review');
  requestQualityPlan();
});
function renderModels(data) {
  activeModelId = data.active_model_id || null;
  const selected = $('models-list');
  const previous = selected.value;
  selected.replaceChildren();
  modelChoices = new Map();
  for (const candidate of data.candidates || []) {
    const option = document.createElement('option');
    option.value = candidate.artifact_id;
    option.disabled = !candidate.compatible;
    option.textContent = candidate.artifact_id
      + (candidate.compatible ? ' · compatible · ' + candidate.labels + ' labels'
        : ' · incompatible (' + (candidate.reason || 'invalid') + ')')
      + (candidate.active ? ' · ACTIVE' : '');
    selected.append(option);
    modelChoices.set(candidate.artifact_id, candidate);
  }
  selected.value = modelChoices.has(previous)
    ? previous : (data.selected_candidate_id || data.candidates?.find(c=>c.compatible)?.artifact_id || '');
  $('models-activate').disabled = !modelChoices.get(selected.value)?.compatible;
  $('models-status').textContent = 'Active: ' + (activeModelId || 'no verified v2 model')
    + ' · Runtime: ' + (data.health || 'unknown');
  displayHealth({model:data.health});
}
$('models-list').addEventListener('change', () => {
  $('models-activate').disabled = !modelChoices.get($('models-list').value)?.compatible;
});
$('models-refresh').addEventListener('click',()=>studioRequest('models'));
$('models-activate').addEventListener('click',()=>{
  const artifact_id = $('models-list').value;
  if (!modelChoices.get(artifact_id)?.compatible) return;
  if (typeof window.confirm === 'function'
      && !window.confirm('Make ' + artifact_id + ' the Active Model?')) return;
  studioRequest('model_activate',{artifact_id});
});
function handleStudioResponse(message) {
  const action = studioRequests.get(message.request_id);
  if (!action) return;
  studioRequests.delete(message.request_id);
  if (!message.ok) {
    const target = ['review_plan','batch_accept'].includes(action)
      ? $('review-queue-status')
      : action === 'sample_detail' ? $('review-visual-status')
      : ['models','model_activate'].includes(action) ? $('models-status')
      : action === 'snapshot' ? $('snapshot-result')
      : action.startsWith('collect_') ? $('collect-progress') : $('workspace-message');
    target.textContent = message.error || 'Studio operation failed.';
    return;
  }
  if (action === 'models') {
    renderModels(message.data);
  } else if (action === 'review_plan') {
    renderQualityPlan(message.data);
  } else if (action === 'batch_accept') {
    $('review-queue-status').textContent='Accepted '+message.data.accepted_count
      +' remaining observations. Review events saved.';
    requestBrowse(reviewOffset);
  } else if (action === 'model_activate') {
    activeModelId = message.data.artifact_id;
    $('models-status').textContent = 'Active: ' + activeModelId + ' · verified and loaded';
    studioRequest('models');
  } else if (action === 'sample_detail') {
    renderSampleDetail(message.data);
  } else if (action === 'overview') {
    $('workspace-message').textContent = 'Connected workspace: ' + message.data.workspace;
    renderDataset(message.data);
  } else if (action === 'snapshot') {
    $('snapshot-result').textContent = 'Snapshot created: '
      + message.data.snapshot_id + ' (' + message.data.workspace_relative_path + ')';
    requestBrowse(reviewOffset);
  } else if (action.startsWith('collect_')) {
    renderCollection(message.data);
    if (['complete','finished'].includes(message.data.state)) requestBrowse(0);
  } else {
    $('workspace-message').textContent = 'Saved manual review for ' + message.data.sample_id;
    requestBrowse(reviewOffset);
  }
}
$('browse-workspace').addEventListener('click', async () => {
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
async function nativeWorkspaceDialog(command) {
  try {
    const selected=await window.__TAURI__?.core?.invoke(command);
    if(!selected)return; // native Cancel must never change the workspace
    pendingWorkspace=selected;
    $('workspace-path').value=selected;
    $('workspace-message').textContent='Connecting workspace: '+selected;
    attachSidecar(null);
  }catch(error){
    $('workspace-message').textContent=String(error);
  }
}
$('select-workspace').addEventListener('click',()=>nativeWorkspaceDialog('pick_workspace'));
$('create-workspace').addEventListener('click',()=>{
  if(typeof window.confirm==='function'&&!window.confirm(
    'Initialize a new Hand-D Project Workspace in an empty folder?'))return;
  nativeWorkspaceDialog('create_workspace');
});
$('refresh-dataset').addEventListener('click', () => requestBrowse(reviewOffset));
for (const action of ['accept','reject','drop','restore']) {
  $('review-' + action).addEventListener('click', () => {
    if (action==='accept' && $('review-accept').disabled) return;
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
    activeModelId = null;
    renderCollection({state:'idle',count:0,target:0});
    previewUrl = null;
    $('preview').removeAttribute('src');
    $('collect-preview').removeAttribute('src');
    collectOverlay.clear();
    $('collect-visual-status').textContent='Waiting for camera connection';
    reviewOverlay.clear();
    worldOverlay.clear();
    selectedReviewWorld=null;
    $('review-accept').disabled=true;
    $('review-sample').replaceChildren();
    $('review-visual-status').textContent='Select a workspace and sample to inspect.';
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
  $('collect-preview').onload = () => {
    const image = $('collect-preview');
    if (image.naturalWidth > 0 && image.naturalHeight > 0)
      $('collect-visual').style.aspectRatio = image.naturalWidth + '/' + image.naturalHeight;
  };
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
        activeModelId = event.snapshot?.active_model_id || null;
        displayHealth(event.snapshot?.health);
        connectionState(true, 'SIDECAR / CONNECTED');
        if (status.workspace) studioRequest('overview');
        if (status.workspace) studioRequest('collect_status');
        if (status.workspace) studioRequest('models');
        else $('workspace-message').textContent =
          'No workspace selected. Open an existing project to review samples.';
      } else if (event.type === 'runtime.update') onRuntimeEvent(event);
      else if (event.type === 'studio.response') handleStudioResponse(event);
      else if (event.type === 'studio.model') {
        activeModelId = event.snapshot?.active_model_id || null;
        displayHealth(event.snapshot?.health);
        $('models-status').textContent = 'Active: ' + (activeModelId || 'none')
          + ' · runtime updated';
      }
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
    if (status?.error) {
      attachSidecar(null);
      connectionState(false, 'SIDECAR / ' + status.error);
      return;
    }
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
