import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

// On macOS the native dialog runs as a child process and waits for user input.
// Tauri 2 runs plain synchronous commands on its UI thread, so those commands
// must explicitly opt into asynchronous dispatch to keep WebKit responsive.
const main=readFileSync(new URL('../src-tauri/src/main.rs',import.meta.url),'utf8');

test('workspace and drawing filesystem commands never block the Tauri UI thread',()=>{
  for(const name of ['select_workspace','pick_workspace','create_workspace',
                     'save_drawing','open_drawing','export_svg']) {
    assert.ok(new RegExp(`#\\[tauri::command\\(async\\)\\]\\s*fn ${name}\\(`).test(main),
      `${name} must use Tauri async dispatch for filesystem/dialog I/O`);
  }
});
