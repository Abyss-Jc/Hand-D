import test from 'node:test';
import assert from 'node:assert/strict';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';

function fixture() {
  const stats={created:0,dUpdates:0,removes:0};
  const callbacks=[];
  class Element {
    constructor(tag){this.tag=tag;this.attributes={};this.children=[];this.parentElement=null;}
    setAttribute(name, value){this.attributes[name]=value;if(name==='d')stats.dUpdates++;}
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
