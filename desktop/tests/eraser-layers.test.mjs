import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const css=readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
const markup=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');

function layer(selector) {
  const rules=css.split(selector+'{');
  assert.ok(rules.length>1,'missing style rule for '+selector);
  // Responsive rules can override spacing without overriding z-index.
  const values=rules.slice(1)
    .map(part=>part.split('}')[0].match(/z-index:\s*(\d+)/))
    .filter(Boolean);
  assert.ok(values.length,'missing z-index for '+selector);
  return Number(values.at(-1)[1]);
}

test('camera and bottom toolbar are independent from transparent ink',()=>{
  assert.match(markup,/<div class="camera-scene" id="camera-scene">/);
  assert.match(markup,/<svg id="drawing"/);
  assert.match(markup,/<div class="tools">/);
  assert.ok(layer('.tools')>layer('.camera-scene'));
  assert.ok(layer('.tools')>layer('#drawing'));
  assert.ok(layer('.canvas-tag')>layer('.camera-scene'));
});
