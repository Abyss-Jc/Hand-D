/** Synthetic SVG presentation budget probe, not a WebKit FPS measurement.
 * Run: node desktop/tests/benchmark-line-boil.mjs
 * The 12fps clock must not rewrite any stroke geometry, even for many lines.
 */
import {performance} from 'node:perf_hooks';
import {StrokeRenderer} from '../web/stroke-renderer.mjs';
import {LINE_BOIL_FRAME_MS} from '../web/line-boil.mjs';

const stats = {created:0, dUpdates:0, rootSwitches:0};
class Node {
  constructor(tag) {this.tag=tag;this.children=[];this.parentElement=null;this.attributes={};}
  append(child){child.remove();this.children.push(child);child.parentElement=this;}
  remove(){if(!this.parentElement)return;const nodes=this.parentElement.children;
    nodes.splice(nodes.indexOf(this),1);this.parentElement=null;}
  setAttribute(k,v){this.attributes[k]=String(v);if(k==='d')stats.dUpdates++;
    if(this.tag==='svg'&&k==='data-boil-frame')stats.rootSwitches++;}
  removeAttribute(k){delete this.attributes[k];}
}
const svg=new Node('svg');
const renderer=new StrokeRenderer(svg,{
  createPath:()=>{stats.created++;return new Node('path');},
  createSvg:tag=>{stats.created++;return new Node(tag);},
  scheduleFrame:callback=>callback(),
});
const strokes=Array.from({length:120},(_,i)=>({
  tool:i%17===0?'draw':'wiggly',
  points:Array.from({length:30},(_,j)=>({
    x:.03+j*.029,
    y:.09+i*.006+Math.sin(j*.4+i)*.006,
  })),
}));
const started=performance.now();
renderer.sync(strokes);
const initialMs=performance.now()-started;
const geometryUpdates=stats.dUpdates;
const tickStarted=performance.now();
for(let i=0;i<360;i++)
  renderer.animateWiggly(i*LINE_BOIL_FRAME_MS+.1,{enabled:true});
const tickMs=performance.now()-tickStarted;
const afterTickUpdates=stats.dUpdates-geometryUpdates;
renderer.animateWiggly(30001,{enabled:false});
console.log(JSON.stringify({
  probe:'synthetic_svg_not_webkit',
  strokes:strokes.length,wobblyStrokes:strokes.filter(s=>s.tool==='wiggly').length,
  cachedFramesPerStroke:3,
  initialMs:Math.round(initialMs*100)/100,
  clock360FramesMs:Math.round(tickMs*100)/100,
  pathUpdatesPerClockTick:afterTickUpdates/360,
  svgPathsCreated:stats.created,rootSwitches:stats.rootSwitches,
},null,2));
if(afterTickUpdates!==0 || stats.rootSwitches!==360 || svg.children.length!==120)
  process.exitCode=3;
