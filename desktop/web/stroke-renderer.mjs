/** Incremental SVG drawing with immutable 3-frame line-boil presentations.
 * Only the active stroke updates path data during input; ticking 12fps changes
 * one SVG root attribute, not thousands of paths or canonical points.
 */
import {buildLineBoilFrames, LINE_BOIL_FRAME_MS} from './line-boil.mjs';
import {strokeStyle} from './drawing-document.mjs';

const SVG_NS = 'http://www.w3.org/2000/svg';
let nextEraserMaskId = 0;
// Older completed ink stays visible as a static stroke. Only recent Wiggly
// strokes spend SVG style/paint time on the 12fps animation clock.
export const MAX_ANIMATED_WIGGLY = 64;

function strokeBounds(stroke,from=0) {
  if (from>=stroke.points.length) return null;
  // Include the pen radius, the eraser footprint, and Wiggly's bounded
  // display-only displacement/width. Conservative boxes never drop an erase.
  const width=strokeStyle(stroke).width;
  const padding=stroke.tool==='wiggly'?width*.565+4:width/2+1;
  const box={left:Infinity,top:Infinity,right:-Infinity,bottom:-Infinity};
  for(let i=from;i<stroke.points.length;i++){
    const point=stroke.points[i];
    box.left=Math.min(box.left,point.x*1000-padding);
    box.top=Math.min(box.top,point.y*600-padding);
    box.right=Math.max(box.right,point.x*1000+padding);
    box.bottom=Math.max(box.bottom,point.y*600+padding);
  }
  return box;
}

function unionBounds(a,b) {
  if(!a||!b)return null; // Unknown ink geometry must not be pruned.
  return {left:Math.min(a.left,b.left),top:Math.min(a.top,b.top),
    right:Math.max(a.right,b.right),bottom:Math.max(a.bottom,b.bottom)};
}

function intersects(a,b) {
  return !a||!b||(a.left<=b.right&&b.left<=a.right
    &&a.top<=b.bottom&&b.top<=a.bottom);
}

export class StrokeRenderer {
  constructor(svg, {
    scheduleFrame = callback => typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame(callback) : queueMicrotask(callback),
    delayFrame = (callback,ms) => setTimeout(callback,ms),
    now = () => performance.now(),
    minEraseIntervalMs = 1000/30,
    createPath = () => document.createElementNS(SVG_NS, 'path'),
    createSvg = tag => document.createElementNS(SVG_NS, tag),
  } = {}) {
    this.svg = svg;
    this.scheduleFrame = scheduleFrame;
    this.delayFrame = delayFrame;
    this.now = now;
    this.minEraseIntervalMs = minEraseIntervalMs;
    this.createPath = createPath;
    this.createSvg = createSvg;
    this.elements = new Map();
    this.pending = new Set();
    this.scheduled = false;
    this.order = [];
    this.defs = null;
    this.eraserLayers = [];
    this.unmaskedInk = [];
    this.eraserLinks = new Map();
    this.lastErasePaintMs = null;
    this.lastErasedStroke = null;
    this.delayed = false;
    this.scheduleVersion = 0;
    this.maskLayerNodes = new Set();
    this.wigglyCount = 0;
    this.animatedWiggly = [];
    this.boilFrame = null;
  }

  /** O(1) entry point for live drawing: no scanning earlier document strokes. */
  append(stroke) {
    if (this.elements.has(stroke)) throw Error('Stroke already added');
    this._createStroke(stroke,stroke.tool === 'wiggly');
    this.order.push(stroke);
    if (stroke.tool === 'wiggly') {
      this.animatedWiggly.push(stroke);
      if (this.animatedWiggly.length > MAX_ANIMATED_WIGGLY)
        this._demote(this.elements.get(this.animatedWiggly.shift()));
    }
    this.wigglyCount = this.animatedWiggly.length;
  }

  sync(paths) {
    // Normal drawing only appends one new path; NEVER rebuild previous ink on
    // point updates. Undo/redo that removes an eraser rebuilds mask structure
    // while retaining the existing ink <path> nodes.
    const appendOnly = this.order.length <= paths.length
      && this.order.every((stroke, i) => stroke === paths[i]);
    const keep = new Set(paths);
    for (const [stroke, entry] of this.elements) {
      if (!keep.has(stroke)) {
        entry.node.remove();
        this.elements.delete(stroke);
        this.pending.delete(stroke);
        this.eraserLinks.delete(stroke);
      }
    }
    if (!appendOnly) {
      for (const entry of this.elements.values()) entry.node.remove();
      for (const child of [...this.svg.children]) child.remove();
      this.defs = null;
      this.eraserLayers = [];
      this.unmaskedInk = [];
      this.eraserLinks.clear();
      this.maskLayerNodes.clear();
      this.lastErasePaintMs = null;
      this.lastErasedStroke = null;
    }
    const active = paths.filter(stroke=>stroke.tool === 'wiggly').slice(-MAX_ANIMATED_WIGGLY);
    const activeSet = new Set(active);
    const newPaths = appendOnly ? paths.slice(this.order.length) : paths;
    for (const stroke of newPaths) {
      if (this.elements.has(stroke)) {
        this._attach(stroke, this.elements.get(stroke).node);
        continue;
      }
      this._createStroke(stroke,activeSet.has(stroke));
    }
    for (const [stroke,entry] of this.elements) {
      if (stroke.tool !== 'wiggly') continue;
      if (activeSet.has(stroke)) this._promote(stroke,entry);
      else this._demote(entry);
    }
    this.order = [...paths];
    this.animatedWiggly = active;
    this.wigglyCount = active.length;
    if (!this.wigglyCount) this._showBoilFrame(null);
  }

  _createStroke(stroke,animate) {
    const path=this.createPath();
    const style=strokeStyle(stroke);
    path.setAttribute('fill','none');
    path.setAttribute('stroke',style.color);
    path.setAttribute('stroke-width',String(style.width));
    path.setAttribute('stroke-linejoin','round');
    path.setAttribute('stroke-linecap','round');
    path.setAttribute('pointer-events','none');
    const node=stroke.tool==='wiggly'?this.createSvg('g'):path;
    if(stroke.tool==='wiggly'){
      node.setAttribute('data-lineboil-stroke','');
      path.setAttribute('data-lineboil-static','');
      node.append(path);
    }
    this._attach(stroke,node);
    const first=stroke.points[0]||{x:0,y:0};
    const seed=Number.isInteger(stroke.boilSeed)?stroke.boilSeed
      :(Math.imul(Math.round(first.x*1e6),2654435761)
        ^ Math.imul(Math.round(first.y*1e6),1597334677))>>>0;
    const entry={path,node,variants:[],seed,count:0,geometry:''};
    this.elements.set(stroke,entry);
    if(animate)this._promote(stroke,entry);
    this.pointAdded(stroke);
  }

  _promote(stroke,entry) {
    if(entry.variants.length)return;
    const style=strokeStyle(stroke);
    entry.node.setAttribute('data-lineboil-active','');
    for(let frame=0;frame<3;frame++){
      const variant=this.createPath();
      variant.setAttribute('fill','none');
      variant.setAttribute('stroke',style.color);
      variant.setAttribute('stroke-linejoin','round');
      variant.setAttribute('stroke-linecap','round');
      variant.setAttribute('pointer-events','none');
      variant.setAttribute('data-lineboil-frame',String(frame));
      entry.node.append(variant);
      entry.variants.push(variant);
    }
    // Rebuild frame geometry only when an undo promotes an existing stroke.
    if(entry.count)this._updateVariants(stroke,entry);
  }

  _demote(entry) {
    if(!entry.variants.length)return;
    for(const variant of entry.variants)variant.remove();
    entry.variants=[];
    entry.node.removeAttribute('data-lineboil-active');
  }

  _updateVariants(stroke,entry) {
    const frames=buildLineBoilFrames(stroke.points,{seed:entry.seed});
    const width=strokeStyle(stroke).width/5;
    frames.forEach((frame,i)=>{
      entry.variants[i].setAttribute('d',frame.d);
      entry.variants[i].setAttribute('stroke-width',String(frame.width*width));
    });
  }

  _attach(stroke, node) {
    if (stroke.tool !== 'erase') {
      // Ink drawn AFTER an eraser is on top of the previous masked layers.
      this.svg.append(node);
      this.unmaskedInk.push(stroke);
      return;
    }
    if (!this.defs) {
      this.defs = this.createSvg('defs');
      this.svg.append(this.defs);
    }
    const newInk = [...this.svg.children].filter(layer=>
      layer !== this.defs && !this.maskLayerNodes.has(layer));
    // Reuse the last mask for consecutive erases with no new ink.
    const reused = !newInk.length && this.eraserLayers.length;
    const maskId = 'handd-erase-' + (++nextEraserMaskId);
    let mask;
    if (reused) {
      mask = this.eraserLayers.at(-1).mask;
    } else {
      mask = this.createSvg('mask');
      mask.setAttribute('id',maskId);
      mask.setAttribute('maskUnits','userSpaceOnUse');
      mask.setAttribute('maskContentUnits','userSpaceOnUse');
      mask.setAttribute('mask-type','luminance');
      mask.setAttribute('x','0');mask.setAttribute('y','0');
      mask.setAttribute('width','1000');mask.setAttribute('height','600');
      const opaque=this.createSvg('rect');
      opaque.setAttribute('x','0');opaque.setAttribute('y','0');
      opaque.setAttribute('width','1000');opaque.setAttribute('height','600');
      opaque.setAttribute('fill','white');
      mask.append(opaque);
      this.defs.append(mask);
    }
    // Each ink epoch has one flat mask, rather than nested scene masks.
    const pathId='handd-erase-path-'+nextEraserMaskId;
    node.setAttribute('id',pathId);
    mask.append(node); // black subtracts ink alpha, never the camera
    if(newInk.length){
      const bounds=this.unmaskedInk.slice(1).reduce(
        (box,ink)=>unionBounds(box,strokeBounds(ink)),strokeBounds(this.unmaskedInk[0]));
      const inkLayer=this.createSvg('g');
      inkLayer.setAttribute('mask','url(#'+maskId+')');
      for(const layer of newInk)inkLayer.append(layer);
      this.svg.append(inkLayer);
      this.eraserLayers.push({mask,layer:inkLayer,bounds});
      this.maskLayerNodes.add(inkLayer);
    }
    this.unmaskedInk=[];
    this.eraserLinks.set(stroke,{pathId,count:0,bounds:null,
      pendingLayers:this.eraserLayers.filter(layer=>layer.mask!==mask)});
    this._linkEraser(stroke);
  }

  _linkEraser(stroke) {
    const link=this.eraserLinks.get(stroke);
    if(!link||link.count===stroke.points.length)return;
    // Expand using only new points; the bounds of the connected path include
    // every segment. Earlier ink epochs are frozen once their mask is created.
    const from=link.count>stroke.points.length?0:link.count;
    const bounds=from===0 ? strokeBounds(stroke)
      : unionBounds(link.bounds,strokeBounds(stroke,from));
    link.count=stroke.points.length;
    link.bounds=bounds;
    // An epoch already linked to this eraser can never need a second <use>.
    // Remove it from the candidate set so a long stroke does not revisit it
    // on every paint (especially on a dense/full-canvas drawing).
    link.pendingLayers=link.pendingLayers.filter(layer=>{
      if(!intersects(layer.bounds,bounds))return true;
      const reference=this.createSvg('use');
      reference.setAttribute('href','#'+link.pathId);
      layer.mask.append(reference);
      return false;
    });
  }

  pointAdded(stroke) {
    if (!this.elements.has(stroke)) return;
    this.pending.add(stroke);
    if (this.scheduled) {
      // New pen input should not wait for an old eraser's paint timer.
      if (stroke.tool !== 'erase' && this.delayed) {
        this.delayed=false;
        const version=++this.scheduleVersion;
        this.scheduleFrame(()=>{if(version===this.scheduleVersion)this.flush();});
      }
      return;
    }
    this.scheduled = true;
    const remaining=stroke.tool==='erase'&&this.lastErasedStroke===stroke
      ? this.lastErasePaintMs+this.minEraseIntervalMs-this.now():0;
    const version=++this.scheduleVersion;
    if(remaining>0){
      this.delayed=true;
      this.delayFrame(()=>{
        if(version!==this.scheduleVersion)return;
        this.scheduleFrame(()=>{if(version===this.scheduleVersion)this.flush();});
      },remaining);
    }else this.scheduleFrame(()=>{if(version===this.scheduleVersion)this.flush();});
  }

  flush() {
    this.scheduled = false;
    this.delayed = false;
    const pending = [...this.pending];
    this.pending.clear();
    for (const stroke of pending) {
      const entry = this.elements.get(stroke);
      if (!entry) continue; // stroke was undone or document cleared
      if(stroke.tool==='erase')this._linkEraser(stroke);
      const count = stroke.points.length;
      if (count < entry.count) {
        entry.geometry = '';
        entry.count = 0;
      }
      if (count === entry.count) continue;
      for (let i = entry.count; i < count; i++) {
        const p = stroke.points[i];
        entry.geometry += (i === 0 ? 'M' : ' L')
          + (p.x * 1000).toFixed(2) + ' ' + (p.y * 600).toFixed(2);
      }
      entry.count = count;
      entry.path.setAttribute(
        'd', entry.geometry + (count === 1 ? ' l0.1 0.1' : '')
      );
      if(stroke.tool==='erase'){
        this.lastErasePaintMs=this.now();
        this.lastErasedStroke=stroke;
      }
      if (entry.variants.length) {
        this._updateVariants(stroke,entry);
      }
    }
  }

  replacePoints(stroke) {
    // Ruler preview updates its second endpoint without accumulating points.
    const entry=this.elements.get(stroke);
    if (!entry) return;
    entry.count=0;
    entry.geometry='';
    this.pointAdded(stroke);
  }

  _showBoilFrame(frame) {
    if (frame === this.boilFrame) return;
    if (frame === null) this.svg.removeAttribute('data-boil-frame');
    else this.svg.setAttribute('data-boil-frame',String(frame));
    this.boilFrame = frame;
  }

  /** Low-FPS line boil: swap display layers with one root attribute only. */
  animateWiggly(elapsedMs,{enabled=true}={}) {
    if (!enabled || !this.wigglyCount) {
      this._showBoilFrame(null);
      return;
    }
    this._showBoilFrame(
      Math.floor(Math.max(0,elapsedMs)/LINE_BOIL_FRAME_MS)%3);
  }
}
