/** Orthographic view of a stored MediaPipe world-space hand; display only. */
export function projectWorldLandmarks(points, yawDegrees=0, pitchDegrees=0) {
  if (!Array.isArray(points)||points.length!==21||
      points.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v))))
    return null;
  const yaw=yawDegrees*Math.PI/180,pitch=pitchDegrees*Math.PI/180;
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch);
  const rotated=points.map(([x,y,z])=>{
    const nx=x*cy+z*sy, nz=z*cy-x*sy;
    return {x:nx,y:y*cx-nz*sx,depth:y*sx+nz*cx};
  });
  const xs=rotated.map(p=>p.x),ys=rotated.map(p=>p.y);
  const minX=Math.min(...xs),minY=Math.min(...ys);
  const extent=Math.max(Math.max(...xs)-minX,Math.max(...ys)-minY);
  if (!(extent>1e-9)) return null;
  const centerX=(Math.min(...xs)+Math.max(...xs))/2;
  const centerY=(Math.min(...ys)+Math.max(...ys))/2;
  return rotated.map(p=>({
    x:Math.max(0,Math.min(1,0.5+(p.x-centerX)/extent*0.8)),
    y:Math.max(0,Math.min(1,0.5+(p.y-centerY)/extent*0.8)),
  }));
}
