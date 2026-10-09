/** Lightweight two-hand landmark presenter; no camera ownership or snapshots. */
export const HAND_EDGES = Object.freeze([
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
]);

export function fitScene(containerWidth, containerHeight, imageWidth=640, imageHeight=480) {
  if (![containerWidth, containerHeight, imageWidth, imageHeight].every(x=>Number.isFinite(x)&&x>0))
    return {left:0,top:0,width:Math.max(0,containerWidth),height:Math.max(0,containerHeight)};
  const scale=Math.min(containerWidth/imageWidth,containerHeight/imageHeight);
  const width=imageWidth*scale, height=imageHeight*scale;
  return {left:(containerWidth-width)/2,top:(containerHeight-height)/2,width,height};
}

export class HandOverlay {
  constructor(svg,{createSvg = tag=>document.createElementNS('http://www.w3.org/2000/svg',tag)}={}) {
    this.svg=svg;
    this.groups={};
    this.visible=true;
    for(const [role,color] of [['drawing','#dfff58'],['modifier','#6977ff']]){
      const group=createSvg('g');
      group.setAttribute('stroke',color);
      group.setAttribute('fill',color);
      group.setAttribute('stroke-width','.005');
      group.style.display='none';
      const edges=[];
      for(const [a,b] of HAND_EDGES){
        const line=createSvg('line');
        line.setAttribute('stroke-width','.006');
        line.setAttribute('data-bone',a+'-'+b);
        group.append(line);
        edges.push({element:line,a,b});
      }
      const dots=[];
      for(let i=0;i<21;i++){
        const circle=createSvg('circle');
        circle.setAttribute('r',i===8 ? '.011' : '.006');
        circle.setAttribute('data-joint',String(i));
        group.append(circle);
        dots.push(circle);
      }
      svg.append(group);
      this.groups[role]={group,edges,dots};
    }
  }
  setVisible(show){
    this.visible=Boolean(show);
    this.svg.style.display=this.visible?'':'none';
  }
  update(payload){
    for(const role of ['drawing','modifier']){
      const entry=this.groups[role];
      const points=payload?.[role]?.landmarks;
      if(!Array.isArray(points)||points.length!==21
         ||!points.every(p=>p && Number.isFinite(p.x)&&Number.isFinite(p.y)
                            && p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1)){
        entry.group.style.display='none';
        continue;
      }
      entry.group.style.display='';
      for(const {element,a,b} of entry.edges){
        element.setAttribute('x1',points[a].x);
        element.setAttribute('y1',points[a].y);
        element.setAttribute('x2',points[b].x);
        element.setAttribute('y2',points[b].y);
      }
      for(let i=0;i<21;i++){
        entry.dots[i].setAttribute('cx',points[i].x);
        entry.dots[i].setAttribute('cy',points[i].y);
      }
    }
  }
  clear(){this.update({});}
}
