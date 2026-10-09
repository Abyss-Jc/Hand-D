/** HD-09 native editable ink-only file contract.
 * Camera imagery, gesture runtime and undo history are never serialized.
 * SVG export is a static clean illustration, never a composite preview.
 */
const FORMAT='handd-whiteboard';
const VERSION=1;
const MAX_BYTES=8_000_000;
const VALID_TOOLS=new Set(['draw','erase','wiggly']);
const validPoint=p=>p&&typeof p==='object'
  &&Number.isFinite(p.x)&&Number.isFinite(p.y)
  &&p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1;
function validate(paths){
 if(!Array.isArray(paths)||paths.length>10000)throw Error('Invalid stroke count');
 let total=0;
 return paths.map(path=>{
   if(!path||!VALID_TOOLS.has(path.tool)||!Array.isArray(path.points)
       ||path.points.length<1)throw Error('Invalid stroke');
   total+=path.points.length;
   if(total>100000||path.points.some(p=>!validPoint(p)))
     throw Error('Invalid point coordinates or document size');
   return {tool:path.tool,points:path.points.map(p=>({x:p.x,y:p.y}))};
 });
}
export function encodeWhiteboard(paths){
 const json=JSON.stringify({format:FORMAT,version:VERSION,
   canvas:{width:1000,height:600},strokes:validate(paths)});
 if(json.length>MAX_BYTES)throw Error('Drawing is too large to save');
 return json;
}
export function decodeWhiteboard(json){
 if(typeof json!=='string'||json.length>MAX_BYTES)throw Error('Invalid drawing file');
 let doc;
 try{doc=JSON.parse(json);}catch{throw Error('Invalid drawing JSON');}
 if(doc?.format!==FORMAT||doc.version!==VERSION
    ||doc.canvas?.width!==1000||doc.canvas?.height!==600)
   throw Error('Unsupported Hand-D drawing format');
 return validate(doc.strokes);
}
const pos=n=>n.toFixed(2);
function geometry(points){
 return points.map((p,i)=>(i?' L':'M')+pos(p.x*1000)+' '+pos(p.y*600)).join('')
   +(points.length===1?' l0.1 0.1':'');
}
export function exportCleanSvg(paths){
 const strokes=validate(paths);
 const masks=[];
 let ink='';
 let id=0;
 for(const stroke of strokes){
   const d=geometry(stroke.points);
   if(stroke.tool==='erase'){
     const maskId='handd-export-erase-'+(id++);
     masks.push('<mask id="'+maskId+'" maskUnits="userSpaceOnUse"'
       +' maskContentUnits="userSpaceOnUse" mask-type="luminance"'
       +' x="0" y="0" width="1000" height="600">'
       +'<rect x="0" y="0" width="1000" height="600" fill="white"/>'
       +'<path d="'+d+'" fill="none" stroke="black" stroke-width="32"'
       +' stroke-linecap="round" stroke-linejoin="round"/></mask>');
     ink='<g mask="url(#'+maskId+')">'+ink+'</g>';
   }else{
     ink+='<path d="'+d+'" fill="none"'
       +' stroke="'+(stroke.tool==='wiggly'?'#5263e6':'#1a1d1c')+'"'
       +' stroke-width="'+(stroke.tool==='wiggly'?'5':'4')+'"'
       +' stroke-linecap="round" stroke-linejoin="round"/>';
   }
 }
 return '<svg xmlns="http://www.w3.org/2000/svg"'
   +' width="1000" height="600" viewBox="0 0 1000 600">'
   +'<defs>'+masks.join('')+'</defs>'+ink+'</svg>\n';
}
