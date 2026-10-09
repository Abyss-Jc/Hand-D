import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';
import {LINE_BOIL_FRAME_MS} from '../web/line-boil.mjs';

class Node {
  constructor(tag='svg') {
    this.tag=tag;this.children=[];this.attributes={};this.style={};
    this.parentElement=null;this.updates=0;
  }
  append(e){e.remove();e.parentElement=this;this.children.push(e);}
  setAttribute(k,v){this.attributes[k]=String(v);if(k==='d')this.updates++;}
  removeAttribute(k){delete this.attributes[k];}
  remove(){
    if(!this.parentElement)return;
    const a=this.parentElement.children;a.splice(a.indexOf(this),1);
    this.parentElement=null;
  }
}
function fixture() {
  const svg=new Node();
  const renderer=new StrokeRenderer(svg,{
    createPath:()=>new Node('path'),createSvg:tag=>new Node(tag),
    scheduleFrame:cb=>cb(),
  });
  return {svg,renderer};
}
const squiggle=()=>({
  tool:'wiggly',
  points:Array.from({length:25},(_,i)=>({
    x:.12+i*.027,y:.42+Math.sin(i*.3)*.1,
  })),
});

test('Wiggly is three precomputed ink frames plus editable canonical path',()=>{
  const {svg,renderer}=fixture();
  const wiggle=squiggle();
  const normal={tool:'draw',points:[{x:.1,y:.2},{x:.2,y:.3}]};
  const canonical=JSON.stringify(wiggle.points);
  renderer.sync([wiggle,normal]);
  assert.equal(svg.children.length,2,'one presentation group per Wiggly stroke');
  const group=svg.children[0];
  assert.equal(group.tag,'g');
  assert.equal(group.children.length,4,
    'one canonical SVG path and three authored line-boil variants');
  const [original,...variants]=group.children;
  assert.equal(original.attributes['data-lineboil-static'],'');
  assert.deepEqual(variants.map(p=>p.attributes['data-lineboil-frame']),
                   ['0','1','2']);
  assert.equal(new Set(variants.map(p=>p.attributes.d)).size,3);
  assert.ok(variants.every(p=>p.attributes.d !== original.attributes.d));
  assert.equal(renderer.elements.get(wiggle).path,original);
  assert.equal(JSON.stringify(wiggle.points),canonical);
  assert.equal(svg.children[1].attributes.stroke,'#1a1d1c');
});

test('12fps discrete clock only toggles ONE root attribute, no path rewrites',()=>{
  const {svg,renderer}=fixture();
  const wiggle=squiggle();
  renderer.sync([wiggle]);
  const canonical=renderer.elements.get(wiggle).path;
  const allPaths=svg.children[0].children;
  const beforeUpdates=allPaths.reduce((sum,p)=>sum+p.updates,0);
  const beforeD=allPaths.map(p=>p.attributes.d);
  renderer.animateWiggly(0,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'0');
  renderer.animateWiggly(LINE_BOIL_FRAME_MS/2,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'0');
  renderer.animateWiggly(LINE_BOIL_FRAME_MS+1,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'1');
  renderer.animateWiggly(2*LINE_BOIL_FRAME_MS+1,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'2');
  renderer.animateWiggly(3*LINE_BOIL_FRAME_MS+1,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'0');
  assert.deepEqual(allPaths.map(p=>p.attributes.d),beforeD);
  assert.equal(allPaths.reduce((sum,p)=>sum+p.updates,0),beforeUpdates);
  renderer.animateWiggly(999,{enabled:false});
  assert.equal(svg.attributes['data-boil-frame'],undefined,
    'reduced motion restores static canonical presentation');
  assert.equal(canonical.attributes.d,beforeD[0]);
});

test('only active Wiggly stroke recomputes its frames, undo/redo restores geometry',()=>{
  const {svg,renderer}=fixture();
  const first=squiggle();
  const second={tool:'wiggly',points:[{x:.5,y:.7},{x:.6,y:.8}]};
  renderer.sync([first,second]);
  const firstGroup=svg.children[0], secondGroup=svg.children[1];
  const firstPaths=firstGroup.children.map(p=>p.attributes.d);
  const secondPaths=secondGroup.children.map(p=>p.attributes.d);
  second.points.push({x:.7,y:.75});
  renderer.pointAdded(second);
  assert.deepEqual(firstGroup.children.map(p=>p.attributes.d),firstPaths);
  assert.notDeepEqual(secondGroup.children.map(p=>p.attributes.d),secondPaths);
  renderer.sync([first]);
  assert.equal(svg.children.length,1);
  assert.equal(svg.children[0],firstGroup);
  renderer.sync([first,second]);
  assert.deepEqual(svg.children[1].children.map(p=>p.attributes.d),
                   secondGroup.children.map(p=>p.attributes.d),
    'redo restores the same deterministic line-boil geometry');
  renderer.sync([]);
  assert.equal(svg.children.length,0);
  renderer.animateWiggly(100,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],undefined,
    'no active Wiggly paths means no animation work');
});

test('multiple erasers clip Wiggly frame groups but never the camera or future ink',()=>{
  const {svg,renderer}=fixture();
  const ink=squiggle();
  const erased={tool:'erase',points:[{x:.3,y:.5}]};
  const newest=squiggle();
  renderer.sync([ink,erased,newest]);
  const mask=svg.children.find(n=>n.tag==='defs');
  assert.ok(mask);
  assert.equal(mask.children[0].tag,'mask');
  assert.equal(mask.children[0].children.at(-1).attributes.stroke,'black');
  assert.equal(renderer.elements.get(ink).node.parentElement.tag,'g');
  assert.match(renderer.elements.get(ink).node.parentElement.attributes.mask,/url\(#handd-erase/);
  assert.equal(renderer.elements.get(newest).node.parentElement,svg);
  renderer.animateWiggly(2*LINE_BOIL_FRAME_MS,{enabled:true});
  assert.equal(svg.attributes['data-boil-frame'],'2');
  renderer.sync([ink]);
  assert.equal(renderer.elements.get(ink).node,svg.children[0]);
  assert.equal(svg.children.length,1);
});

test('CSS selects exactly one variant per tick, reduced motion overrides clock',()=>{
  const css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
  assert.match(css,/#drawing \[data-lineboil-frame\]\s*\{\s*display:\s*none/);
  for(let i=0;i<3;i++){
    const selector='#drawing[data-boil-frame="'+i+'"] [data-lineboil-frame="'+i+'"]';
    assert.ok(css.includes(selector),'missing active frame '+i+' CSS selector');
  }
  assert.match(css,/prefers-reduced-motion:\s*reduce/);
  assert.match(css,/#drawing \[data-lineboil-static\]\s*\{\s*display:\s*inline!important/);
});
