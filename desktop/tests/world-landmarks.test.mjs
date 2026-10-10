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
