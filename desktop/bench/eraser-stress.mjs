/** Synthetic SVG structure and JS update benchmark, NOT a WebKit raster/FPS test.
 * Run: node desktop/bench/eraser-stress.mjs
 */
import {performance} from 'node:perf_hooks';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';

class SvgNode {
  constructor(tag) {this.tag=tag;this.attributes={};this.children=[];this.parentElement=null;}
  setAttribute(name,value){this.attributes[name]=value;}
  removeAttribute(name){delete this.attributes[name];}
  remove(){if(!this.parentElement)return;const siblings=this.parentElement.children;
    siblings.splice(siblings.indexOf(this),1);this.parentElement=null;}
  append(child){child.remove();this.children.push(child);child.parentElement=this;}
}

for(const pattern of ['overlap','localized'])for(const erasers of [20,60,120]) {
  const svg=new SvgNode('svg');
  const renderer=new StrokeRenderer(svg,{
    scheduleFrame:fn=>fn(),
    minEraseIntervalMs:0, // count structure, not the real-time paint scheduler
    createPath:()=>new SvgNode('path'),createSvg:tag=>new SvgNode(tag),
  });
  const start=performance.now();
  for(let index=0;index<erasers;index++){
    const col=index%10,row=Math.floor(index/10)%6;
    const x=pattern==='localized'?.05+col*.1:.1;
    const y=pattern==='localized'?.07+row*.15:.2;
    renderer.append({tool:'draw',points:pattern==='localized'
      ? [{x,y},{x:x+.02,y:y+.02}]
      : [{x:.1,y:.2},{x:.7,y:.8}]});
    const points=Array.from({length:40},(_,i)=>pattern==='localized'
      ? {x:x+i/2000,y:y+.01}
      : {x:(i+1)/65,y:.2+index/300});
    renderer.append({tool:'erase',points});
  }
  const renderMs=Number((performance.now()-start).toFixed(2));
  let maxMaskDepth=0,maskReferences=0,maskGroups=0;
  function visit(node,depth){
    const newDepth=depth+(node.attributes.mask?1:0);
    maxMaskDepth=Math.max(maxMaskDepth,newDepth);
    if(node.tag==='use')maskReferences++;
    if(node.attributes.mask)maskGroups++;
    for(const child of node.children)visit(child,newDepth);
  }
  visit(svg,0);
  console.log(JSON.stringify({pattern,erasers,renderMs,maxMaskDepth,maskGroups,maskReferences}));
}
