import test from 'node:test';
import assert from 'node:assert/strict';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';

class Node {
  constructor(){this.children=[];this.attributes={};this.parentElement=null;}
  append(e){e.parentElement=this;this.children.push(e);}
  setAttribute(k,v){this.attributes[k]=v;}
  remove(){if(!this.parentElement)return;const a=this.parentElement.children;a.splice(a.indexOf(this),1);this.parentElement=null;}
}

test('wiggly style animates only its SVG geometry while keeping canonical points untouched',()=>{
  const svg=new Node();
  const renderer=new StrokeRenderer(svg,{createPath:()=>new Node(),scheduleFrame:cb=>cb()});
  const wiggle={tool:'wiggly',points:Array.from({length:12},(_,i)=>({x:i/20,y:.4}))};
  const normal={tool:'draw',points:[{x:.1,y:.2},{x:.2,y:.3}]};
  const unchanged=JSON.stringify(wiggle.points);
  renderer.sync([wiggle,normal]);
  const stable=svg.children[0].attributes.d;
  const normalPath=svg.children[1].attributes.d;
  renderer.animateWiggly(0,{enabled:true});
  const frame1=svg.children[0].attributes.d;
  renderer.animateWiggly(.7,{enabled:true});
  assert.notEqual(svg.children[0].attributes.d,frame1);
  assert.notEqual(svg.children[0].attributes.d,stable);
  assert.equal(svg.children[1].attributes.d,normalPath);
  assert.equal(JSON.stringify(wiggle.points),unchanged);
  renderer.animateWiggly(1.4,{enabled:false});
  assert.equal(svg.children[0].attributes.d,stable);
  renderer.sync([normal]);
  renderer.animateWiggly(2,{enabled:true});
  assert.equal(svg.children.length,1);
});

test('Wiggly displacement is clearly visible but bounded and reversible', () => {
  const svg=new Node();
  const renderer=new StrokeRenderer(svg,{createPath:()=>new Node(),scheduleFrame:cb=>cb()});
  const points=Array.from({length:22},(_,i)=>({x:.12+i*.028,y:.5}));
  const stroke={tool:'wiggly',points};
  renderer.sync([stroke]);
  const baseline=svg.children[0].attributes.d;
  renderer.animateWiggly(1.7,{enabled:true});
  const rendered=svg.children[0].attributes.d;
  const nums=[...rendered.matchAll(/-?\d+(?:\.\d+)?/g)].map(x=>Number(x[0]));
  const maxDisplacement=Math.max(...points.map((p,i)=>
    Math.hypot(nums[2*i]-p.x*1000,nums[2*i+1]-p.y*600)));
  assert.ok(maxDisplacement>=6,
    'Wiggly should move at least 6 SVG pixels, rather than barely tremble');
  assert.ok(maxDisplacement<=14,'Wiggly must remain near its editable path');
  assert.notEqual(rendered,baseline);
  renderer.animateWiggly(2.4,{enabled:true});
  assert.notEqual(svg.children[0].attributes.d,rendered);
  renderer.animateWiggly(3,{enabled:false});
  assert.equal(svg.children[0].attributes.d,baseline);
  assert.deepEqual(stroke.points,points);
});
