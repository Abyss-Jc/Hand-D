import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLineBoilFrames, LINE_BOIL_FRAMES, LINE_BOIL_FRAME_MS,
} from '../web/line-boil.mjs';

const nums = path => [...path.matchAll(/-?\d+(?:\.\d+)?/g)].map(match=>+match[0]);
const coords = path => {
  const values = nums(path);
  return Array.from({length:values.length/2},(_,i)=>[values[i*2],values[i*2+1]]);
};

test('three deterministic bounded drawing variants with static endpoints', () => {
  const points = [{x:.10,y:.3},{x:.4,y:.3},{x:.5,y:.5},{x:.7,y:.5}];
  const before = JSON.stringify(points);
  const frames = buildLineBoilFrames(points, {seed:1234});
  assert.equal(LINE_BOIL_FRAMES,3);
  assert.ok(LINE_BOIL_FRAME_MS>=80&&LINE_BOIL_FRAME_MS<=85);
  assert.equal(frames.length,3);
  assert.equal(new Set(frames.map(f=>f.d)).size,3);
  assert.deepEqual(frames,buildLineBoilFrames(points,{seed:1234}));
  assert.notDeepEqual(frames,buildLineBoilFrames(points,{seed:4321}));
  for(const {d,width} of frames){
    assert.ok(width>=4.5&&width<=6);
    const rendered=coords(d);
    assert.deepEqual(rendered[0],[100,180]);
    assert.deepEqual(rendered.at(-1),[700,300]);
    // Reference polyline: horizontal, diagonal, horizontal.
    for(const [x,y] of rendered){
      let expected;
      if(x<=400) expected=180;
      else if(x<=500) expected=180+(x-400)*1.2;
      else expected=300;
      assert.ok(Math.abs(y-expected)<=6.5,
        'local perpendicular boil must not fling ink far from the base drawing');
    }
  }
  assert.equal(JSON.stringify(points),before);
});

test('equivalent motion at different point sampling rates has the same three frames',()=>{
  const sparse=[{x:.1,y:.2},{x:.4,y:.2},{x:.4,y:.65},{x:.8,y:.65}];
  const dense=[];
  for(let i=0;i<sparse.length-1;i++){
    for(let t=0;t<9;t++){
      dense.push({x:sparse[i].x+(sparse[i+1].x-sparse[i].x)*t/9,
                  y:sparse[i].y+(sparse[i+1].y-sparse[i].y)*t/9});
    }
  }
  dense.push(sparse.at(-1));
  assert.deepEqual(buildLineBoilFrames(sparse,{seed:8}),
                   buildLineBoilFrames(dense,{seed:8}),
    'drawing depends on distance and corners, NOT MediaPipe sample count');
});

test('sharp corners and loops remain, long strokes remain memory-bounded',()=>{
  const corner=[{x:0.1,y:0.1},{x:0.6,y:0.1},{x:0.6,y:0.7}];
  const frames=buildLineBoilFrames(corner,{seed:41});
  for(const frame of frames){
    const vertices=coords(frame.d);
    assert.ok(vertices.some(([x,y])=>Math.hypot(x-600,y-60)<6),
      'preserve the corner at 600,60');
  }
  const long=Array.from({length:6000},(_,i)=>({
    x:.5+0.43*Math.cos(i*.23), y:.5+0.42*Math.sin(i*.23),
  }));
  for(const frame of buildLineBoilFrames(long,{seed:42})){
    assert.ok(coords(frame.d).length<=1400,'bounded presenter geometry');
    assert.ok(frame.d.length<28000,'bounded SVG string size');
  }
});
