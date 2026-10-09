import test from 'node:test';
import assert from 'node:assert/strict';
import { RuntimeGate, Strokes } from '../web/runtime-state.mjs';

test('new sidecar snapshot rejects old and out-of-order pointer events', () => {
  const gate = new RuntimeGate();
  gate.install({ runtime_session_id: 'old', seq: 4, timestamp_ms: 100 });
  assert.equal(gate.accept({type:'runtime.update', runtime_session_id:'old',seq:5,timestamp_ms:150}), true);
  gate.install({ runtime_session_id:'new',seq:0,timestamp_ms:-1 });
  assert.equal(gate.accept({type:'runtime.update',runtime_session_id:'old',seq:6,timestamp_ms:200}), false);
  assert.equal(gate.accept({type:'runtime.update',runtime_session_id:'new',seq:1,timestamp_ms:5}), true);
  assert.equal(gate.accept({type:'runtime.update',runtime_session_id:'new',seq:1,timestamp_ms:5}), false);
});

test('strokes and undo history stay intact across runtime reconnections', () => {
  const strokes = new Strokes();
  strokes.add([{x:.1,y:.2}, {x:.3,y:.4}]);
  const before = strokes.serialize();
  const gate = new RuntimeGate();
  gate.install({runtime_session_id:'new',seq:0,timestamp_ms:0});
  assert.equal(strokes.serialize(), before);
  strokes.undo();
  assert.equal(strokes.paths.length, 0);
  strokes.redo();
  assert.equal(strokes.serialize(), before);
});
