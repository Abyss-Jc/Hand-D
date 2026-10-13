/** Synthetic SVG/JS stress check; not a browser paint/FPS measurement.
 * Run: node desktop/bench/wiggly-stress.mjs
 */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {StrokeRenderer,MAX_ANIMATED_WIGGLY} from '../web/stroke-renderer.mjs';

class SvgNode {
  constructor(tag) {
    this.tag=tag;
    this.attributes={};
    this.children=[];
    this.parentElement=null;
  }
  setAttribute(key,value) {this.attributes[key]=value;}
  removeAttribute(key) {delete this.attributes[key];}
  remove() {
    if(!this.parentElement)return;
    const children=this.parentElement.children;
    children.splice(children.indexOf(this),1);
    this.parentElement=null;
  }
  append(node) {
    node.remove();
    this.children.push(node);
    node.parentElement=this;
  }
}

const points=Array.from({length:12},(_,index)=>({
  x:index/100,y:(index%2)*0.01,
}));
for(const count of [600,1200,2400]){
  for(const method of ['sync','append']){
    const svg=new SvgNode('svg');
    const renderer=new StrokeRenderer(svg,{
      scheduleFrame:callback=>callback(),
      createPath:()=>new SvgNode('path'),
      createSvg:tag=>new SvgNode(tag),
    });
    const start=performance.now();
    for(let index=0;index<count;index++){
      const stroke={tool:'wiggly',points};
      if(method==='sync')renderer.sync([...renderer.order,stroke]);
      else renderer.append(stroke);
    }
    const setupMs=+(performance.now()-start).toFixed(1);
    const groups=[...renderer.elements.values()];
    const svgPaths=groups.reduce((sum,entry)=>sum+entry.node.children.length,0);
    const variants=groups.reduce((sum,entry)=>sum+entry.variants.length,0);
    assert.equal(svgPaths,count+MAX_ANIMATED_WIGGLY*3);
    assert.equal(variants,MAX_ANIMATED_WIGGLY*3);
    console.log(JSON.stringify({strokes:count,method,setupMs,svgPaths,variants}));
  }
}
