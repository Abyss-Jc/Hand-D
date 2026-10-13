import test from 'node:test';
import assert from 'node:assert/strict';
import {StrokeRenderer,MAX_ANIMATED_WIGGLY} from '../web/stroke-renderer.mjs';

function fixture() {
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
