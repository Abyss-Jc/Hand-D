import test from 'node:test';
import assert from 'node:assert/strict';
import {HandOverlay, HAND_EDGES, fitScene} from '../web/hand-overlay.mjs';

class Element {
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.attributes={};this.parentElement=null;}
  setAttribute(k,v){this.attributes[k]=v;}
  append(node){node.parentElement=this;this.children.push(node);}
}

test('all 21 joints and 20 anatomical connections per hand, keyed by role',()=>{
  const root=new Element('svg');
  const overlay=new HandOverlay(root,{createSvg:name=>new Element(name)});
  const points=Array.from({length:21},(_,i)=>({x:i/40,y:i/50}));
  overlay.update({drawing:{landmarks:points},modifier:{landmarks:points}});
  assert.equal(HAND_EDGES.length,20);
  assert.equal(root.children.length,2);
  for(const hand of root.children){
    assert.equal(hand.children.filter(node=>node.tag==='circle').length,21);
    assert.equal(hand.children.filter(node=>node.tag==='line').length,20);
  }
  assert.notEqual(root.children[0].attributes.stroke,root.children[1].attributes.stroke);
  assert.equal(root.children[0].children[20].attributes.cx,0);
  overlay.update({drawing:{landmarks:null},modifier:{landmarks:points}});
  assert.equal(root.children[0].style.display,'none');
  assert.equal(root.children[1].style.display,'');
  overlay.setVisible(false);
  assert.equal(root.style.display,'none');
  overlay.setVisible(true);
  assert.equal(root.style.display,'');
});

test('invalid hand or stale releases skeleton, no NaN in SVG',()=>{
  const root=new Element('svg');
  const overlay=new HandOverlay(root,{createSvg:name=>new Element(name)});
  overlay.update({drawing:{landmarks:[{x:NaN,y:0}]},modifier:{}});
  assert.equal(root.children[0].style.display,'none');
  const valid=Array.from({length:21},()=>({x:.5,y:.5}));
  overlay.update({drawing:{landmarks:valid},modifier:{}});
  assert.equal(root.children[0].style.display,'');
  overlay.clear();
  assert.equal(root.children[0].style.display,'none');
  assert.equal(root.children[1].style.display,'none');
});

test('fit contain keeps the 4:3 camera undistorted in wide and tall viewports',()=>{
  assert.deepEqual(fitScene(1200,600,640,480),{left:200,top:0,width:800,height:600});
  assert.deepEqual(fitScene(600,900,640,480),{left:0,top:225,width:600,height:450});
  assert.deepEqual(fitScene(640,480,640,480),{left:0,top:0,width:640,height:480});
});
