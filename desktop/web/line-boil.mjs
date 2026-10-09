/**
 * Hand-D original line-boil geometry (not copied from WigglyPaint).
 *
 * Three deterministic, gently different pen outlines are computed when a
 * stroke changes. The display clock later swaps visible frames only.
 * Sampling is by arc length so mouse speed / MediaPipe callback rate do not
 * change the texture. No input point or gesture model state is mutated.
 */
export const LINE_BOIL_FRAMES = 3;
export const LINE_BOIL_FRAME_MS = 1000 / 12;
const X = 1000;
const Y = 600;
const SAMPLE_GAP = 9;
const MAX_VERTICES = 1200;
const BOIL_AMPLITUDE = 3.3;
const CORNER_COS = Math.cos(Math.PI / 4); // preserve turns sharper than 45°

const xy = point => ({x:point.x * X,y:point.y * Y});
const finite = point => point && Number.isFinite(point.x) && Number.isFinite(point.y);
const fmt = n => n.toFixed(2);

// Stateless seeded field; the frame's values never depend on sample count.
function hashNoise(seed, frame, knot) {
  let value = (seed ^ Math.imul(frame + 1, 0x9e3779b9)
    ^ Math.imul(knot + 101, 0x85ebca6b)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return ((value ^ (value >>> 16)) >>> 0) / 0xffffffff * 2 - 1;
}

function field(seed, frame, distance) {
  const coordinate = distance / 24;
  const index = Math.floor(coordinate);
  const t = coordinate - index;
  const smooth = t*t*(3-2*t);
  return hashNoise(seed, frame, index) * (1-smooth)
    + hashNoise(seed, frame, index+1) * smooth;
}

function buildSamples(points) {
  const vertices = [];
  for(const p of points) {
    if (!finite(p)) continue;
    const v = xy(p);
    const last=vertices.at(-1);
    if (!last || Math.hypot(v.x-last.x,v.y-last.y) > 0.00001)
      vertices.push(v);
  }
  if (vertices.length === 0) return [];
  if (vertices.length === 1) return [{...vertices[0],distance:0}];

  const distances = [0];
  for(let i=1;i<vertices.length;i++){
    const a=vertices[i-1], b=vertices[i];
    distances.push(distances[i-1] + Math.hypot(b.x-a.x,b.y-a.y));
  }
  const total=distances.at(-1);
  if (total < .00001) return [{...vertices[0],distance:0}];
  const step=Math.max(SAMPLE_GAP,total / (MAX_VERTICES-1));
  const marks=[0,total];
  for(let i=1;i<MAX_VERTICES-1;i++){
    const d=i*step;
    if(d>=total) break;
    marks.push(d);
  }

  // Preserve meaningful user-authored corners independently of the clock.
  // When pathological paths contain thousands of corners, choose a bounded
  // subset with the largest turns; never grow DOM geometry without limit.
  const corners = [];
  for(let i=1;i<vertices.length-1;i++){
    const a=vertices[i-1],b=vertices[i],c=vertices[i+1];
    const len1=Math.hypot(b.x-a.x,b.y-a.y);
    const len2=Math.hypot(c.x-b.x,c.y-b.y);
    const dot=((b.x-a.x)*(c.x-b.x)+(b.y-a.y)*(c.y-b.y))/(len1*len2);
    if (dot < CORNER_COS) corners.push({distance:distances[i],score:1-dot});
  }
  corners.sort((a,b)=>b.score-a.score);
  const room=Math.max(0,MAX_VERTICES-marks.length);
  for(const c of corners.slice(0,room)) marks.push(c.distance);
  marks.sort((a,b)=>a-b);
  const unique=[];
  for(const d of marks)
    if (!unique.length || d-unique.at(-1)>0.02) unique.push(d);

  const samples=[];
  let segment=1;
  for(const distance of unique){
    while(segment<distances.length-1 && distances[segment]<distance) segment++;
    const from=vertices[segment-1],to=vertices[segment];
    const duration=distances[segment]-distances[segment-1];
    const t=duration ? Math.max(0,Math.min(1,(distance-distances[segment-1])/duration)) : 0;
    samples.push({
      x:from.x+(to.x-from.x)*t,
      y:from.y+(to.y-from.y)*t,
      distance,
    });
  }
  return samples;
}

function framePath(samples, seed, frame) {
  if (!samples.length) return '';
  const total=samples.at(-1).distance;
  const result=[];
  for(let i=0;i<samples.length;i++){
    const sample=samples[i];
    const prev=samples[Math.max(0,i-1)];
    const next=samples[Math.min(samples.length-1,i+1)];
    const tx=next.x-prev.x, ty=next.y-prev.y;
    const len=Math.hypot(tx,ty)||1;
    // Anchored endpoints, local tangent-normal boil instead of translating
    // the full stroke; smooth spatial noise keeps corners continuous.
    const fade=total ? Math.min(1,sample.distance/13,(total-sample.distance)/13) : 0;
    const offset=BOIL_AMPLITUDE*fade*field(seed,frame,sample.distance);
    const x=sample.x-ty/len*offset;
    const y=sample.y+tx/len*offset;
    result.push((i===0?'M':' L')+fmt(x)+' '+fmt(y));
  }
  return result.join('')+(samples.length===1?' l0.1 0.1':'');
}

/**
 * Returns exactly three display-only SVG variants.
 * The seed may be a stable stroke ID; the function is pure and repeatable.
 * Each variant contains at most MAX_VERTICES coordinates.
 */
export function buildLineBoilFrames(points,{seed=1}={}) {
  const samples=buildSamples(points);
  return [5.15,5.65,4.9].map((width,frame)=>({
    d:framePath(samples,seed>>>0,frame),
    width,
  }));
}
