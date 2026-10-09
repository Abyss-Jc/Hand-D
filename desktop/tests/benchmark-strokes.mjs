/** Reproducible synthetic SVG-cost probe; not a WebKit FPS benchmark.
 * Run: node desktop/tests/benchmark-strokes.mjs
 * Simulates 120 finished strokes + one active stroke in the same app module.
 */
import {performance} from 'node:perf_hooks';

const ids = ['drawing','undo','redo','clear','tool-pen','tool-eraser',
  'nav-whiteboard','nav-studio','whiteboard','studio','canvas-shell',
  'modal-canvas-slot','full-overlay','close-expand','expand','camera-fallback',
  'preview','footer-session','status','lamp','gesture','tracking','diagnostics',
  'cursor','restart','model-status','camera-scene','hands-overlay',
  'view-camera','view-clean','toggle-hands','tool-wiggly'];
const counters = {nodesCreated:0, replaceChildren:0, pathUpdates:0, svgAppend:0};
const frameTasks = [];
globalThis.requestAnimationFrame = cb => { frameTasks.push(cb); return frameTasks.length; };
globalThis.cancelAnimationFrame = () => {};
globalThis.setInterval = () => {};
class Element {
  constructor(id) {
    this.id = id;this.hidden=false;this.listeners={};this.children=[];
    this.style={};this.parentElement=null;this.classList={toggle(){}};
  }
  addEventListener(name,cb){this.listeners[name]=cb;}
  setAttribute(name,value){ if(name==='d')counters.pathUpdates++;this[name]=value;}
  removeAttribute(name){delete this[name];}
  append(node){this.children.push(node);node.parentElement=this;counters.svgAppend++;}
  prepend(node){this.children.unshift(node);node.parentElement=this;}
  replaceChildren(){this.children=[];counters.replaceChildren++;}
  remove(){this.parentElement?.children.splice(this.parentElement.children.indexOf(this),1);}
  setPointerCapture(){}
  getBoundingClientRect(){return {left:0,top:0,width:1000,height:600};}
  focus(){}
  emit(name, event={}){this.listeners[name]?.(event);}
}
const objects=Object.fromEntries(ids.map(name=>[name,new Element(name)]));
objects.drawing.parentElement=objects['canvas-shell'];
objects['canvas-shell'].parentElement=new Element('workspace');
globalThis.document = {
  getElementById(name){return objects[name]||(()=>{throw new Error(name)})();},
  createElementNS(_ns,type){counters.nodesCreated++;return new Element(type);},
  addEventListener(){},
};
globalThis.window={};
await import('../web/app.js');
function flushRaf(){
  for(let i=0;i<100&&frameTasks.length;i++){
    const tasks=frameTasks.splice(0);
    for(const cb of tasks)cb(performance.now());
  }
}
const svg=objects.drawing;
function stroke(n, shift){
 svg.emit('pointerdown',{button:0,pointerId:1,clientX:100+shift,clientY:100});
 for(let i=1;i<n;i++){
   svg.emit('pointermove',{clientX:100+shift+(i%650),clientY:100+(i%370)});
   if(i%10===0)flushRaf();
 }
 svg.emit('pointerup',{});
 flushRaf();
}
const started=performance.now();
for(let i=0;i<120;i++)stroke(8,i);
stroke(450,10);
const wallMs=performance.now()-started;
console.log(JSON.stringify({probe:"synthetic_js_dom_not_webkit",finishedStrokes:121,totalPoints:120*8+450,wallMs:Math.round(wallMs*100)/100,...counters,renderedNodes:svg.children.length},null,2));
if(svg.children.length!==121)process.exit(5);
