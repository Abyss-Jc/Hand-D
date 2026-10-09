import test from 'node:test';
import assert from 'node:assert/strict';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';

function fixture() {
  const stats={created:0,dUpdates:0,removes:0};
  const callbacks=[];
  class Element {
    constructor(tag){this.tag=tag;this.attributes={};this.children=[];this.parentElement=null;}
    setAttribute(name, value){this.attributes[name]=value;if(name==='d')stats.dUpdates++;}
    append(child){this.children.push(child);child.parentElement=this;}
    remove(){if(!this.parentElement)return;const list=this.parentElement.children;list.splice(list.indexOf(this),1);stats.removes++;this.parentElement=null;}
  }
  const svg=new Element('svg');
  const renderer=new StrokeRenderer(svg,{
    scheduleFrame:callback=>{callbacks.push(callback);},
    createPath:()=>{stats.created++;return new Element('path');},
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

test('RED: undo redo clear never re-create unaffected SVG paths and preserve eraser style',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const first=makeStroke('draw',[{x:.1,y:.2},{x:.3,y:.4}]);
  const second=makeStroke('erase',[{x:.5,y:.6}]);
  renderer.sync([first,second]);
  frame();
  const firstElement=svg.children[0];
  assert.equal(svg.children[1].attributes['stroke-width'],'32');
  renderer.sync([first]);
  assert.equal(svg.children.length,1);
  assert.equal(svg.children[0],firstElement);
  renderer.sync([first,second]);
  frame();
  assert.equal(svg.children.length,2);
  assert.equal(svg.children[0],firstElement);
  renderer.sync([]);
  assert.equal(svg.children.length,0);
});

test('RED: a queued animation frame cannot resurrect removed strokes',()=>{
  const {renderer,svg,makeStroke,frame}=fixture();
  const stroke=makeStroke('draw',[{x:.5,y:.5}]);
  renderer.sync([stroke]);
  renderer.sync([]);
  frame();
  assert.equal(svg.children.length,0);
});
