import test from 'node:test';
import assert from 'node:assert/strict';

import {projectWorldLandmarks} from '../web/world-landmarks.mjs';

test('review projection presents all 21 stored 3D points and can rotate without changing the source',()=>{
  const input=Array.from({length:21},(_,i)=>[i*.01,(i%5)*.02,(i%7)*.01]);
  const original=JSON.stringify(input);
  const front=projectWorldLandmarks(input,0);
  const turned=projectWorldLandmarks(input,65);
  assert.equal(front.length,21);
  assert.ok(front.every(p=>p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1));
  assert.notDeepEqual(turned,front);
  assert.equal(JSON.stringify(input),original);
});

test('review does not render malformed world landmarks',()=>{
  assert.equal(projectWorldLandmarks([],0),null);
  assert.equal(projectWorldLandmarks(Array.from({length:21},()=>[0,0,0]),0),null);
  assert.equal(projectWorldLandmarks(Array.from({length:21},()=>[0,NaN,1]),30),null);
});

test('independent yaw and pitch allow real two-axis inspection',()=>{
  const points=Array.from({length:21},(_,i)=>[.03*i,.01*(i%4),.02*(i%6)]);
  const front=projectWorldLandmarks(points,0,0);
  const yaw=projectWorldLandmarks(points,50,0);
  const pitch=projectWorldLandmarks(points,0,50);
  assert.notDeepEqual(front,yaw);
  assert.notDeepEqual(front,pitch);
  assert.notDeepEqual(yaw,pitch);
});

test('canonical camera front view places wrist below fingers without changing saved XYZ',()=>{
  // Canonical +Y is wrist->middle MCP; SVG +Y is downward.
  const canonical=Array.from({length:21},(_,i)=>[0,1,.01*i]);
  canonical[0]=[0,0,0]; // wrist
  canonical[9]=[0,1,0]; // middle MCP
  const original=JSON.stringify(canonical);
  const normalized=projectWorldLandmarks(canonical,0,0,true);
  assert.ok(normalized[0].y > normalized[9].y,
    'wrist must visually sit below the palm in canonical front view');
  assert.ok(normalized[0].y>0.5,'wrist stays in lower half of the viewer');
  assert.equal(JSON.stringify(canonical),original,'display rotation never rewrites raw data');
  const raw=projectWorldLandmarks(canonical,0,0,false);
  assert.ok(raw[0].y<raw[9].y,'raw world mode preserves previous image-axis mapping');
});
