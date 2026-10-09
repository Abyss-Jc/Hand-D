import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const markup=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
const script=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');

test('the entire desktop UI uses English even in accessibility, status and Studio',()=>{
  assert.match(markup,/<html lang="en">/);
  const spanish=[
    'Cámara','cámara','Lienzo','lienzo','Ocultar','Mostrar','Dibujar','Borrar',
    'Ampliar','Cerrar','mano derecha','Reiniciar','Esperando','Sin detección',
    'Iniciando','reconectando','Capturar','Revisar','por integrar',
    'SIN CLASIFICACIÓN','Sin sesión','CONECTANDO','Dibujar:','Modificar:',
    'ANTIGUO ACTIVO','SIN VALIDACIÓN','MODELO:',
  ];
  for(const literal of spanish){
    assert.ok(!markup.toLowerCase().includes(literal.toLowerCase()),
      'HTML visible/accessible UI must be English: '+literal);
    assert.ok(!script.toLowerCase().includes(literal.toLowerCase()),
      'runtime UI messages must be English: '+literal);
  }
  assert.match(markup,/Camera/);
  assert.match(markup,/Whiteboard/);
  assert.match(markup,/Studio/);
});
