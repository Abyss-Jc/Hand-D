# Hand-D Whiteboard: colores, nudillo y modificador de dos manos (12-oct-2026)

**Estado:** implementado y probado mediante contratos sintéticos en Linux; interacción física con ambas manos en WebKit/macOS **No verificado**. Rama `feature/hand-d-v2-modernization`.

## Alcance y decisiones revisadas (ask-matt → grilling → junior-to-senior)

Problema del operador: la UI v2 no ofrece colores para Pen/Wiggly, el gesto Fist usa la punta del índice para borrar y los gestos de la segunda mano Ruler/Thumb Up no producen cambios visibles. Prioridad: dejar un flujo comprobable antes del gate Mac, sin alterar entrenamiento ni la cámara.

| Decisión examinada | Evidencia, contraargumento y resolución |
|---|---|
| ¿Reimplementar el visualizador legacy? | No: `visualizer_app/canvas.py:183-209,251-279,294-302` demuestra reglas útiles, pero la arquitectura v2 separa predicción y dibujo. Implementar en `desktop/web/app.js` consumiendo `handd_core/runtime_v2.py` y conservar UI English-only (ADR 0005). |
| ¿Dónde posicionar el borrador? | `visualizer_app/gesture_engine.py:367-371` elegía landmark 9, Middle Finger MCP, para Fist, y 8 para Index Finger. Se restaura 9 **solo cuando Fist se estabiliza**. El modificador Thumb Up usa el landmark 4 (Thumb Tip), evitando confundir un índice doblado con el pulgar. [Índices oficiales de MediaPipe](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/hand_landmarker/HandLandmark). El detector mantiene `num_hands=2` en `handd_core/sidecar_main.py:59`. |
| ¿Cómo se interpreta una regla? | No añade un modelo ni dibuja la trayectoria a mano alzada: `drawing.action=draw` + `modifier.action=ruler` crea un trazo con ancla fija y extremo móvil (dos puntos). Al soltar o salir del modo, conserva una línea editable con Undo/Redo; Pen/Wiggly son el pincel seleccionado. No se guarda una UI efímera de preview en los documentos. |
| ¿Cómo cambia grosor? | `Thumb_Up` sobre Modifier Hand normaliza su posición X en [0,1]: rango de trazo 1–20 unidades o radio del borrador 10–80 (legacy `canvas.py:56-60,294-302`). La UI indica acción, valor y dirección; se captura el grosor por segmento para no cambiar trazos antiguos. Si se ajusta mientras se dibuja, empieza un segmento nuevo con el nuevo grosor, sin reescalar historia. |
| ¿Cómo conservar ink y export limpio? | Colores explícitos por trazo y ancho numérico en el formato existente v1 `.handd.json`, **opcional** para que los archivos viejos sigan abriendo, valores `#RRGGBB` y ancho finito 1–160. El renderer, Save/Open y el SVG estático usan los mismos estilos. Eraser sigue como máscara sobre tinta, nunca sobre cámara. [SVG stroke-width](https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorials/SVG_from_scratch/Fills_and_strokes). |
| ¿Qué NO debe afirmarse? | Un test con eventos WS sintéticos comprueba la interfaz, **no** la exactitud humana del clasificador legacy (diagnóstico unverified), la calibración Left/Right del Mac, la compatibilidad `.app`, el rendimiento ONNX/MPS, ni la latencia extremo a extremo. Se necesitan clics/cámara reales. |

## Interfaz y pruebas automatizadas

- `handd_core/runtime_v2.py` emite `pointer` en MCP 9 para Fist, index tip 8 para dibujo y thumb tip 4 para Modifier Thumb Up. No cambia las 69 features ni la clasificación; mantiene comprobaciones de rango y las dos manos separadas.
- `desktop/web/app.js`, `index.html`, `style.css`: seis swatches Black/Red/Blue/Green/Purple/Orange; control Brush + Eraser radius; aviso de regla activa, gesto pendiente de estabilización o ajuste numérico. Los controles están en `canvas-shell` para sobrevivir al modal ampliado y al cambio Camera/Clean.
- `desktop/web/stroke-renderer.mjs`, `drawing-document.mjs`: estilos por Stroke y regla con geometría editable; Wiggly conserva frames de line-boil y ancho relativo, el borrador máscaras cronológicas; documentos previos conservan sus estilos por defecto sin migración forzosa.
- Contratos: `tests/test_runtime_v2.py` para MCP9/Thumb4 y acciones simultáneas; `desktop/tests/app-ui.test.mjs` para swatches, Wiggly, tamaño y WS ruler/adjust; `desktop/tests/drawing-document.test.mjs` para roundtrip, estilos y sanitización; `desktop/tests/stroke-renderer.test.mjs` para estilos en las tres variantes Wiggly. Pruebas de borrado y UI English-only siguen formando parte de la suite normal.

**Resultado de gates automáticos (Linux, 2026-10-12):** `node --test desktop/tests/*.test.mjs` **56/56**; `uv run --frozen --group training python -m unittest discover -s tests -p 'test_*.py' -q` **151 tests, OK (3 skipped)**; `cargo test --locked --manifest-path desktop/src-tauri/Cargo.toml` **7/7**; `npm --prefix desktop run check` **PASS**. Los tests de MCP9 y persistencia/colores fallaron antes de su implementación (RED), y pasaron después (GREEN). No representan webcam real ni prueba visual de WebKit/macOS.

## Aceptación física pendiente (Mac)

1. Abrir cámara con **ambas** manos: derecha Index Finger dibuja; derecha Fist borra siguiendo el **nudillo central**, no la punta de índice; izquierda Ruler mantiene una recta hasta release. Probar transiciones de modo y Undo/Redo.
2. Izquierda Thumb Up con mano derecha en Index Finger: mover el pulgar horizontalmente y observar `Adjusting brush thickness` con número actualizado; repetir con mano derecha Fist, debe cambiar **Eraser radius** y la anchura de la máscara sin blanquear video.
3. Cambiar colores Pen y Wiggly, aplicar distintos grosores, guardar `.handd.json`, abrir y exportar SVG; validar que las tintas conservan estilo y los frames/cámara nunca se guardan. Probar modal ampliado, Clean Canvas y Reduced Motion.
4. Si no responde la segunda mano, registrar salud del modelo, `raw_gesture`, `stable_gesture`, rol/physical hand y detecciones, **sin** inferir que el clasificador distingue Ruler/Thumb Up por tener landmarks visibles. No guardar fotos/video ni etiquetar samples sin revisión humana.

Fuente operativa y estado del Mac: [`../runbook/mac-lab-oct09.md`](../runbook/mac-lab-oct09.md). ONNX permanece experimental sin alteración de runtime: [`../runbook/hd12-onnx-experimental.md`](../runbook/hd12-onnx-experimental.md).
