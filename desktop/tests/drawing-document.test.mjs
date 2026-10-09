import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeWhiteboard,decodeWhiteboard,exportCleanSvg} from '../web/drawing-document.mjs';

const paths=[
 {tool:'draw',source:'pointer',points:[{x:.1,y:.5},{x:.9,y:.5}]},
 {tool:'wiggly',source:'gesture',points:[{x:.2,y:.25},{x:.7,y:.3}]},
 {tool:'erase',source:'pointer',points:[{x:.5,y:.49},{x:.5,y:.51}]},
 {tool:'draw',points:[{x:.5,y:.5},{x:.6,y:.55}]},
];

test('native Save/Open contract returns an editable versioned ink-only document',()=>{
 const text=encodeWhiteboard(paths);
 const doc=JSON.parse(text);
 assert.equal(doc.format,'handd-whiteboard');
 assert.equal(doc.version,1);
 assert.equal(doc.strokes.length,4);
 assert.equal(JSON.stringify(doc).includes('frame'),false);
 assert.equal(JSON.stringify(doc).includes('camera'),false);
 assert.equal(doc.strokes[1].tool,'wiggly');
 assert.deepEqual(decodeWhiteboard(text),paths.map(p=>({
   tool:p.tool,points:p.points,
 })));
});
test('Open rejects corrupt, unsupported, and oversized documents without changing ink',()=>{
 for(const bad of ['not json','{}',JSON.stringify({format:'handd-whiteboard',version:2,strokes:[]}),
  JSON.stringify({format:'handd-whiteboard',version:1,strokes:[{tool:'evil',points:[{x:.2,y:.3}]}]}),
  JSON.stringify({format:'handd-whiteboard',version:1,strokes:[{tool:'draw',points:[{x:NaN,y:-.1}]}]}),
  'x'.repeat(8_000_100)]){
  assert.throws(()=>decodeWhiteboard(bad));
 }
});
test('clean SVG exports chronological ink masks and NEVER camera or UI',()=>{
 const svg=exportCleanSvg(paths);
 assert.match(svg,/viewBox="0 0 1000 600"/);
 assert.match(svg,/<mask id="handd-export-erase-0"/);
 assert.match(svg,/mask="url\(#handd-export-erase-0\)"/);
 assert.match(svg,/stroke="black"/);
 assert.match(svg,/stroke="#5263e6"/);
 assert.match(svg,/M500\.00 300\.00 L600\.00 330\.00/);
 assert.doesNotMatch(svg,/(camera|preview|video|<image|<script)/i);
 assert.equal((svg.match(/fill="white"/g)||[]).length,1,
   'the only white fill is inside the eraser definition, never the canvas');
 assert.ok(svg.indexOf('mask="url(#handd-export-erase-0)"') <
   svg.indexOf('M500.00 300.00 L600.00 330.00'),
   'new ink is ABOVE previous eraser mask');
});
