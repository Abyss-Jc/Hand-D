import test from 'node:test';
import assert from 'node:assert/strict';
import {StrokeRenderer,MAX_ANIMATED_WIGGLY} from '../web/stroke-renderer.mjs';

function fixture(options={}) {
  const stats={created:0,dUpdates:0,removes:0};
  const callbacks=[];
  class Element {
    constructor(tag){this.tag=tag;this.attributes={};this.children=[];this.parentElement=null;}
    setAttribute(name, value){this.attributes[name]=value;if(name==='d')stats.dUpdates++;}
    removeAttribute(name){delete this.attributes[name];}
    append(child){child.remove();this.children.push(child);child.parentElement=this;}
    remove(){if(!this.parentElement)return;const list=this.parentElement.children;list.splice(list.indexOf(this),1);stats.removes++;this.parentElement=null;}
  }
  const svg=new Element('svg');
  const renderer=new StrokeRenderer(svg,{
    scheduleFrame:callback=>{callbacks.push(callback);},
    createPath:()=>{stats.created++;return new Element('path');},
    createSvg:tag=>{stats.created++;return new Element(tag);},
    minEraseIntervalMs:0,
    ...options,
  });
  function frame(){for(const cb of callbacks.splice(0))cb();}
  const makeStroke=(tool='draw', points=[])=>({tool, points});
  return {renderer,svg,stats,callbacks,frame,makeStroke};
}

test('RED: updates only the active path once per animation frame regardless of past strokes',()=>{
  const {renderer,svg,stats,frame,makeStroke}=fixture();
  const previous=[];
  for(let i=0;i<50;i++){
    const stroke=makeStroke('draw',[{x:i/60,y:.1}]);
    previous.push(stroke);
    renderer.sync(previous);
  }
  frame();
  const earlier=svg.children[0];
  const existingUpdates=stats.dUpdates;
  const active=makeStroke();
  renderer.sync([...previous,active]);
  for(let i=0;i<600;i++){
    active.points.push({x:i/700,y:0.4});
    renderer.pointAdded(active);
  }
  assert.equal(svg.children.length,51,'one SVG path per stroke');
  assert.equal(svg.children[0],earlier,'all old paths retain their DOM identity');
  assert.equal(stats.dUpdates,existingUpdates,'no full path redraw before animation frame');
  frame();
  assert.equal(svg.children.length,51);
  assert.equal(stats.dUpdates-existingUpdates,1,'one active-path update even with 600 new points');
  assert.match(svg.children.at(-1).attributes.d,/L/);
});

test('eraser masks earlier ink instead of painting white on the camera',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const first=makeStroke('draw',[{x:.1,y:.2},{x:.3,y:.4}]);
  const second=makeStroke('erase',[{x:.3,y:.4}]);
  const after=makeStroke('draw',[{x:.3,y:.4}]);
  renderer.sync([first,second,after]);
  frame();
  const firstElement=renderer.elements.get(first).path;
  const eraseElement=renderer.elements.get(second).path;
  const afterElement=renderer.elements.get(after).path;
  assert.notEqual(eraseElement.attributes.stroke,'#fffef9',
    'eraser must NEVER paint white over the camera or controls');
  assert.equal(eraseElement.attributes.stroke,'black');
  assert.equal(eraseElement.attributes['stroke-width'],'32');
  assert.equal(eraseElement.parentElement.tag,'mask',
    'erasure belongs inside an SVG mask, not the visible drawing surface');
  assert.equal(firstElement.parentElement.tag,'g',
    'earlier ink is wrapped in an erase mask');
  assert.equal(afterElement.parentElement,svg,
    'later ink must not be removed by the preceding eraser');
  assert.equal(svg.children.some(node=>node===eraseElement),false);
  assert.equal(svg.children.some(node=>node===afterElement),true);
  renderer.sync([first]);
  assert.equal(firstElement.parentElement,svg);
  assert.equal(svg.children.length,1,'undo eraser removes mask structure');
  renderer.sync([first,second]);
  frame();
  assert.equal(renderer.elements.get(first).path,firstElement,
    'undo/redo preserves geometry and element identity');
  renderer.sync([]);
  assert.equal(svg.children.length,0);
});

test('multiple erasers remain chronological, never erase future ink',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const a=makeStroke('wiggly',[{x:.3,y:.4},{x:.4,y:.4}]);
  const erase1=makeStroke('erase',[{x:.3,y:.4}]);
  const b=makeStroke('draw',[{x:.3,y:.4}]);
  const erase2=makeStroke('erase',[{x:.4,y:.4}]);
  const c=makeStroke('draw',[{x:.4,y:.4}]);
  renderer.sync([a,erase1,b,erase2,c]);frame();
  assert.equal(renderer.elements.get(c).path.parentElement,svg,
    'stroke after second eraser must be unaffected');
  assert.equal(renderer.elements.get(erase1).path.parentElement.tag,'mask');
  assert.equal(renderer.elements.get(erase2).path.parentElement.tag,'mask');
  renderer.sync([a,erase1,b]);frame();
  assert.equal(renderer.elements.get(b).path.parentElement,svg);
  renderer.sync([a]);frame();
  assert.equal(renderer.elements.get(a).node.parentElement,svg);
});

test('RED: a queued animation frame cannot resurrect removed strokes',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const stroke=makeStroke('draw',[{x:.5,y:.5}]);
  renderer.sync([stroke]);
  renderer.sync([]);
  frame();
  assert.equal(svg.children.length,0);
});

test('Pen and each Wiggly boil frame honor the selected per-stroke color and size',()=>{
  const {renderer,frame,makeStroke}=fixture();
  const pen={...makeStroke('draw',[{x:.1,y:.1}]),color:'#E24742',width:12};
  const wiggly={...makeStroke('wiggly',[{x:.1,y:.1},{x:.2,y:.2}]),color:'#188A75',width:8};
  renderer.sync([pen,wiggly]);frame();
  assert.equal(renderer.elements.get(pen).path.attributes.stroke,'#E24742');
  assert.equal(renderer.elements.get(pen).path.attributes['stroke-width'],'12');
  const item=renderer.elements.get(wiggly);
  assert.equal(item.path.attributes.stroke,'#188A75');
  assert.equal(item.path.attributes['stroke-width'],'8');
  assert.ok(item.variants.every(v=>v.attributes.stroke==='#188A75'));
  assert.ok(item.variants.every(v=>Number(v.attributes['stroke-width'])>7));
});

test('a long Wiggly session bounds animated SVG paths, retaining older ink and undo recovery',()=>{
  const {renderer,frame,makeStroke}=fixture();
  const strokes=[];
  for(let i=0;i<MAX_ANIMATED_WIGGLY+90;i++){
    const stroke=makeStroke('wiggly',[{x:.05,y:.05},{x:.5,y:.5}]);
    strokes.push(stroke);
    renderer.append(stroke);
  }
  frame();
  const first=renderer.elements.get(strokes[0]);
  assert.equal(first.node.children.length,1,'older Wiggly stays inked but not animated');
  assert.ok(first.path.attributes.d.includes('L'),'static geometry must survive demotion');
  assert.equal(renderer.elements.get(strokes.at(-1)).node.children.length,4,
    'recent Wiggly keeps all three animated variants');
  const variantCount=strokes.reduce((sum,stroke)=>sum+renderer.elements.get(stroke).variants.length,0);
  assert.equal(variantCount,MAX_ANIMATED_WIGGLY*3,'SVG animation work stays bounded');
  renderer.animateWiggly(90);
  assert.equal(renderer.svg.attributes['data-boil-frame'],'1');
  renderer.sync(strokes.slice(0,2));
  frame();
  assert.equal(renderer.elements.get(strokes[0]).node.children.length,4,
    'undo to old paths reactivates their animation');
  assert.equal(renderer.elements.get(strokes[0]).path,first.path,
    'original static SVG node and shape are preserved');
});

test('incremental append preserves eraser chronology without needing document resync',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const ink=makeStroke('wiggly',[{x:.2,y:.2},{x:.6,y:.6}]);
  const erase=makeStroke('erase',[{x:.5,y:.5}]);
  const later=makeStroke('draw',[{x:.3,y:.3}]);
  renderer.append(ink);
  renderer.append(erase);
  renderer.append(later);
  frame();
  assert.equal(renderer.elements.get(ink).node.parentElement.tag,'g');
  assert.equal(renderer.elements.get(erase).node.parentElement.tag,'mask');
  assert.equal(renderer.elements.get(later).node.parentElement,svg);
  assert.equal(renderer.order.length,3);
});

test('eraser layers stay flat after many ink/erase alternations, with chronology and undo',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const sequence=[];
  for(let index=0;index<40;index++){
    const ink=makeStroke('draw',[{x:index/100,y:.2},{x:index/100+.2,y:.3}]);
    const eraser=makeStroke('erase',[{x:index/100,y:.25}]);
    sequence.push(ink,eraser);
    renderer.append(ink);
    renderer.append(eraser);
  }
  const later=makeStroke('draw',[{x:.7,y:.4}]);
  renderer.append(later);
  frame();
  const levels=[];
  function walk(node,masked=0){
    const level=masked+(node.attributes.mask?1:0);
    levels.push(level);
    for(const child of node.children)walk(child,level);
  }
  walk(svg);
  assert.equal(Math.max(...levels),1,'no stack of 40 nested SVG masks to re-rasterize');
  assert.equal(renderer.elements.get(later).path.parentElement,svg,
    'new ink after erasing is not erased retroactively');
  const firstLayer=renderer.elements.get(sequence[0]).path.parentElement;
  assert.equal(firstLayer.tag,'g');
  const firstMaskId=firstLayer.attributes.mask.match(/#([^)]*)/)[1];
  const defs=svg.children.find(child=>child.tag==='defs');
  const firstMask=defs.children.find(node=>node.attributes.id===firstMaskId);
  const refs=firstMask.children.filter(child=>child.tag==='use').length;
  assert.ok(refs>=19&&refs<39,
    'early ink receives intersecting erasers, without referencing distant ones');
  renderer.sync(sequence.slice(0,2));frame();
  assert.equal(renderer.elements.get(sequence[0]).path.parentElement.tag,'g');
  renderer.sync([...sequence,later]);frame();
  assert.equal(renderer.elements.get(later).path.parentElement,svg);
});

test('consecutive erasers on the same ink epoch reuse a single mask',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const a=makeStroke('draw',[{x:.1,y:.3}]);
  const erasers=Array.from({length:35},(_,i)=>
    makeStroke('erase',[{x:.2+i/100,y:.3}]));
  renderer.append(a);
  for(const erase of erasers)renderer.append(erase);
  frame();
  const defs=svg.children.find(child=>child.tag==='defs');
  assert.equal(defs.children.filter(child=>child.tag==='mask').length,1);
  for(const erase of erasers)
    assert.equal(renderer.elements.get(erase).path.parentElement.tag,'mask');
});

test('active eraser paint is paced, coalesces moves, and eventually shows the final point',()=>{
  let time=0;
  const timers=[];
  const {renderer,frame,makeStroke,stats}=fixture({
    minEraseIntervalMs:32,
    now:()=>time,
    delayFrame:(callback,delay)=>timers.push({callback,delay}),
  });
  renderer.append(makeStroke('draw',[{x:.1,y:.2}]));
  const erase=makeStroke('erase',[{x:.1,y:.2}]);
  renderer.append(erase);
  frame();
  const initial=stats.dUpdates;
  for(let i=0;i<40;i++){
    erase.points.push({x:.1+i/100,y:.2});
    renderer.pointAdded(erase);
  }
  assert.equal(timers.length,1,'one delayed repaint for many eraser movements');
  assert.equal(timers[0].delay,32);
  frame();
  assert.equal(stats.dUpdates,initial,'no paint before the pacing timer');
  time=32;
  timers.shift().callback();
  frame();
  assert.equal(stats.dUpdates,initial+1);
  assert.match(renderer.elements.get(erase).path.attributes.d,/L490\.00 120\.00/);
});

test('only ink epochs intersected by an eraser gain mask references, even if it later moves',()=>{
  const {renderer,svg,frame,makeStroke}=fixture();
  const left=makeStroke('draw',[{x:.1,y:.3}]);
  const leftErase=makeStroke('erase',[{x:.1,y:.3}]);
  const right=makeStroke('draw',[{x:.9,y:.3}]);
  const rightErase=makeStroke('erase',[{x:.9,y:.3}]);
  for(const stroke of [left,leftErase,right,rightErase])renderer.append(stroke);
  frame();
  const defs=svg.children.find(n=>n.tag==='defs');
  const oldMask=defs.children[0];
  assert.equal(oldMask.children.filter(n=>n.tag==='use').length,0,
    'a distant erase cannot affect old ink and needs no reference');
  rightErase.points.push({x:.02,y:.3});
  renderer.pointAdded(rightErase);
  frame();
  assert.equal(oldMask.children.filter(n=>n.tag==='use').length,1,
    'a segment crossing old ink must link its mask even if neither endpoint is inside');
  rightErase.points.push({x:.15,y:.3});
  renderer.pointAdded(rightErase);
  frame();
  assert.equal(oldMask.children.filter(n=>n.tag==='use').length,1,
    'later points must not duplicate an existing reference');
  const later=makeStroke('draw',[{x:.1,y:.3}]);
  renderer.append(later);
  frame();
  assert.equal(renderer.elements.get(later).node.parentElement,svg,
    'ink created after this erase remains unmasked');
});

test('a wide animated Wiggly stroke is not skipped at its displayed edge',()=>{
  const {renderer,svg,frame,makeStroke}=fixture();
  renderer.append({...makeStroke('wiggly',[{x:.5,y:.3}]),width:160});
  renderer.append({...makeStroke('erase',[{x:.5,y:.3}]),width:20});
  renderer.append(makeStroke('draw',[{x:.603,y:.3}]));
  renderer.append({...makeStroke('erase',[{x:.603,y:.3}]),width:20});
  frame();
  const oldMask=svg.children.find(n=>n.tag==='defs').children[0];
  assert.equal(oldMask.children.filter(n=>n.tag==='use').length,1,
    'bounds must include the wide Wiggly display variants, not just its centerline');
});

test('new pen input bypasses a pending eraser paint timer',()=>{
  let time=0;
  const timers=[];
  const {renderer,frame,makeStroke}=fixture({
    minEraseIntervalMs:32,now:()=>time,
    delayFrame:(callback,delay)=>timers.push(callback),
  });
  renderer.append(makeStroke('draw',[{x:.1,y:.2}]));
  const erase=makeStroke('erase',[{x:.1,y:.2}]);
  renderer.append(erase);
  frame();
  erase.points.push({x:.2,y:.2});
  renderer.pointAdded(erase);
  const pen=makeStroke('draw',[{x:.8,y:.8}]);
  renderer.append(pen);
  frame();
  assert.match(renderer.elements.get(pen).path.attributes.d,/M800\.00 480\.00/);
  assert.match(renderer.elements.get(erase).path.attributes.d,/L200\.00 120\.00/);
  time=32;
  timers[0]();frame();
  assert.equal(renderer.elements.get(pen).path.attributes.d,'M800.00 480.00 l0.1 0.1');
});
