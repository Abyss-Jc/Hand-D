# Pruebas Hand-D v2 en Macs — 9 de octubre de 2026

**Estado:** guía de laboratorio, no validación macOS. Rama: `feature/hand-d-v2-modernization`.
**Objetivo:** comprobar la ventana Tauri, el sidecar Python, cámara, 21 puntos, Camera/Limpio, pincel Wiggly y gestos sin guardar imágenes ni video.

## Preparación de cada Mac

Instalar Git, Rust/cargo, Node/npm, uv y Xcode Command Line Tools (ejecutar `xcode-select --install` solo si faltan). **No copiar** las carpetas `.venv`, `desktop/node_modules` o `desktop/src-tauri/target` desde Linux.

```bash
git clone --branch feature/hand-d-v2-modernization https://github.com/Abyss-Jc/Hand-D.git
cd Hand-D
bash scripts/mac-preflight.sh
```

El script instala dependencias **en el proyecto** mediante `uv sync --frozen` y `npm ci --prefix desktop`, verifica Torch, MediaPipe, OpenCV, los tests y `cargo check --locked`; **no abre la cámara**. Anotar versiones de macOS y chip (sin serial ni identificadores personales) y cualquier error. La app Tauri incluye `NSCameraUsageDescription` en `desktop/src-tauri/Info.plist`, pero el permiso real para OpenCV/Python en macOS aún debe probarse.

## Secuencia física de prueba

1. Cerrar Zoom/Meet y Hand-D. Probar la cámara sin guardar frames:
   ```bash
   uv run --frozen python -m handd_core.camera_smoke --camera 0 --seconds 10
   ```
   Registrar `CAMERA_OPEN`, `CAPTURE_FPS`, callbacks, muestras con manos y errores. `camera_smoke` y `hand_mapping` ya usan captura nativa en macOS (sin `CAP_V4L2` de Linux); falta confirmarlo físicamente. `--preview` es opcional; Qt puede comportarse distinto en Macs.
2. Cerrar el probe y lanzar la app (la **UI está completamente en inglés**, ADR 0005):
   ```bash
   npm --prefix desktop run dev
   ```
   Verificar Tauri, estado `SIDECAR / CONNECTED`, MJPEG sobre cámara y diagnóstico `LEGACY MODEL ACTIVE`. El checkpoint legacy sirve solo como diagnóstico: las etiquetas no tienen validación independiente.
3. Mostrar la **mano derecha, después izquierda y después ambas**. Verificar 21 puntos por mano, conexiones y colores (lima Drawing, azul Modifier). Confirmar manualmente el espejo y los roles; no asumir que la inversión física está probada para la webcam Mac.
4. Alternar **Hide hands / Show hands** y **Camera overlay / Clean canvas**: debe conservarse el mismo documento y no deben detenerse MediaPipe ni los eventos WS. En limpio el MJPEG debe dejar de transmitirse al no tener consumidor. Abrir y cerrar el modal ampliado con Escape.
5. Dibujar con ratón o trackpad: trazo largo y rápido, **Undo/Redo**, **Pen**, **Eraser** y **Wiggly** sobre cámara y en limpio. **Regresión crítica de borrador:** dibujar una línea, borrarla parcialmente encima de la cámara; comprobar que reaparece el video real (NO una mancha blanca), que los controles inferiores nunca se tapan ni dejan de responder, que se puede volver a dibujar sobre la zona borrada y que Undo/Redo restaura la tinta. Repetir sobre Clean canvas y sobre un trazo Wiggly. El nuevo Wiggly debe tener un aspecto orgánico **line-boil A/B/C ~12 FPS** (no vibración continua de toda la línea): probar línea recta, círculo, zigzag, escritura, gestos y trazo largo. Comparar calidad y FPS percibidos con Pen normal. Con **Ajustes del Sistema → Accesibilidad → Pantalla → Reducir movimiento**, Wiggly debe mostrar la geometría original, estática y editable.
6. Probar `Index_Finger` y `Fist` y registrar etiquetas observadas **sin atribuirles precisión**. Los gestos sobre cámara no necesitan video almacenado.
7. Pulsar **Restart camera**: sidecar debe reconectarse con sesión nueva, el documento debe sobrevivir y el overlay no debe dejar manos fantasma.
   Si Python nunca anuncia READY, el supervisor interrumpe el arranque tras 45 s; reintenta una vez y después muestra `RUNTIME FAILED - USE RESTART CAMERA`. Probar el botón de recuperación sin perder la tinta (comportamiento validado en tests de contrato Linux, **No verificado** en macOS).
8. Si existe un proyecto de pruebas con `handd.sqlite`, entrar a **Studio → Existing project workspace directory → Open workspace** usando su ruta local. Verificar que **Review** muestra conteos y Sample IDs reales, que **Accept/Reject/Drop/Restore** requieren clicks explícitos y que **Build Development Snapshot** no entrena ni incluye P003. No hacer acciones sobre un dataset real sin intención/backup. Sin proyecto, Studio debe mostrar un empty state honesto y Whiteboard debe seguir funcionando.
9. Con **un workspace temporal de laboratorio, nunca datos de producción**, mantener cámara activa y usar **Studio → Collect → Participant P001 → Intended gesture → Physical hand → Start Capture**. Comprobar contador con la mano seleccionada, **Pause**, **Resume** y **Finish**. Luego pulsar **Refresh** en Review: las nuevas observaciones deben aparecer como unreviewed. **No** debe abrirse una segunda cámara ni almacenarse video. Start debe rechazar cámara desconectada o P003. Captura física de Studio aún NO verificada en Mac ni Linux hasta ejecutar este paso.
10. Con el mismo workspace de laboratorio, usar el selector nativo **Browse… / Create workspace…**, luego **Save drawing… / Open drawing… / Export SVG…**. Comprobar que los archivos guardados contienen solo trazos editables, que abrir un archivo corrupto no borra el lienzo y que el SVG exportado no incluye la cámara. **Model Artifacts → Refresh Models → Make Active** requiere un `candidate-*` real de la salida de entrenamiento; un artefacto inválido no sustituye al actual y el modelo compatible elegido sobrevive **Restart camera**. No hacer activación sobre un dataset/modelo de producción durante una prueba.
    Comprobar que cancelar el diálogo no modifica nada y que un error real de permisos se muestra como error, no como cancelación. Esta distinción cuenta con test Rust, no con comprobación en Mac.
11. En **cada Mac de destino**, construir el sidecar PyInstaller y la app nativos, nunca reutilizar binarios Linux. Fuera del flujo principal de macOS, probar el bundle instalado antes de afirmar compatibilidad:
    ```bash
    bash scripts/build-sidecar-bundle.sh
    npm --prefix desktop run build -- --bundles app
    ```
    La firma, permisos TCC para el child congelado, notarización y eventual salida del sandbox deben verificarse físicamente. `NSCameraUsageDescription` en Info.plist no demuestra por sí solo el permiso efectivo.

## Registro mínimo

| Observación | Mac A | Mac B |
|---|---|---|
| macOS / chip / arquitectura sin serial | No verificado | No verificado |
| Script preflight: imports, tests, cargo | No verificado | No verificado |
| Camera smoke: FPS / callbacks / manos | No verificado | No verificado |
| Tauri / WS / MJPEG / sidecar READY | No verificado | No verificado |
| 21 puntos Drawing + Modifier, alineados | No verificado | No verificado |
| Camera/Limpio, toggle manos, modal | No verificado | No verificado |
| Mouse, undo/redo, Wiggly, reduced-motion | No verificado | No verificado |
| Borrador transparente: no tapa cámara ni controles, re-dibujo y Undo/Redo | No verificado | No verificado |
| Sidecar restart, trazos conservados | No verificado | No verificado |
| UI English-only: accessible controls and messages | No verificado | No verificado |
| Existing workspace / manual review / Development Snapshot (test project only) | No verificado | No verificado |
| Studio Collect: Start/Pause/Resume/Finish + unreviewed Samples, no second camera | No verificado | No verificado |
| Studio Active Model (compatible, rollback, restart) | No verificado | No verificado |
| Native workspace dialog + Save/Open editable + clean SVG | No verificado | No verificado |
| Native .app sidecar subprocess + camera permissions/TCC | No verificado | No verificado |
| Gestos reales / fluidez percibida | No verificado | No verificado |

No compartir tokens del sidecar, seriales, fotos ni video; solo comandos, errores y medidas agregadas.

## Bitacora de la primera sesion en Mac y regresion del selector (12-oct-2026)

**Origen de las observaciones fisicas:** reportes del operador en la conversacion del 9-12 de octubre; no se dispone de logs completos, version exacta de macOS ni identificacion de la Mac. Los campos `No verificado` de la tabla anterior siguen abiertos como *gates de aceptacion*: un reporte visual no sustituye una prueba repetible.

- El operador pudo abrir la interfaz de Hand-D en su Mac y reporto que los puntos de manos eran visibles correctamente. **Reportado, sin medicion de FPS, calibracion Left/Right ni comprobacion del bundle `.app`.**
- En esa sesion, los botones **Create workspace...**, **Open / change workspace...** y el flujo de ruta introducida no produjeron el resultado esperado. En contraste, un comando `osascript` ejecutado por separado **si abrio el selector nativo de carpetas**. No se guardo la transcripcion exacta de ese comando ni un log de errores de la app.
- Diagnostico por codigo: `desktop/src-tauri/src/main.rs` invocaba funciones sincronas que esperan la seleccion humana de `osascript` y la inicializacion del sidecar. En Tauri 2 esos comandos se ejecutan en el hilo UI por defecto; bloquearlo puede dejar la app aparentemente sin reaccion. Ademas, `desktop/web/app.js` interpretaba la ausencia de `window.__TAURI__.core.invoke` como cancelacion silenciosa. **Causa probable respaldada por codigo, pero no confirmada aun en Mac.** Ver [Tauri 2: Async commands](https://v2.tauri.app/develop/calling-rust/).
- Correccion Linux del 12-oct-2026: `select_workspace`, `pick_workspace`, `create_workspace`, `save_drawing`, `open_drawing` y `export_svg` ahora se despachan fuera del hilo principal mediante `#[tauri::command(async)]`; el frontend muestra error explicito si falta el puente nativo. Mientras espera el dialogo, Studio muestra `Opening native workspace folder picker...`, deshabilita temporalmente los botones y evita dialogs duplicados; Cancel preserva la seleccion anterior. RED antes del cambio para los tres contratos; GREEN despues. Ejecutados: `node --test desktop/tests/native-threading.test.mjs desktop/tests/app-ui.test.mjs` (**26/26**), `node --test desktop/tests/*.test.mjs` (**52/52**), `cargo test --locked --manifest-path desktop/src-tauri/Cargo.toml` (**7/7**) y `npm --prefix desktop run check` (PASS). Ninguna de estas pruebas ejecuta `osascript` en macOS.

### Repetir en macOS despues de actualizar la rama

Usar el checkout de la Mac en `feature/hand-d-v2-modernization` (sin copiar `.venv` ni `target` desde Linux):

```bash
git pull --ff-only
npm --prefix desktop run dev
```

1. Abrir **Studio** y elegir **Create workspace...**. Confirmar y seleccionar una carpeta **existente, vacia y de laboratorio**, nunca el repositorio ni un proyecto real. Mientras se abre el selector, Studio debe indicar que lo esta abriendo y evitar clics repetidos; la ventana no debe quedar congelada. Comprobar que se crea `handd.sqlite`, que Studio indica **Connecting workspace**, y finalmente **Connected workspace**. Esperar al nuevo READY del sidecar; no interpretar el reinicio como fallo.
2. Probar **Open / change workspace...** seleccionando ese mismo directorio; comprobar que vuelve el contenido de Studio. En **Open using an existing folder path**, pegar la ruta y pulsar **Use entered path**; debe conseguir el mismo resultado.
3. Cancelar el dialogo y confirmar que no cambia el workspace activo. Elegir una carpeta sin `handd.sqlite` con Open: debe verse un error real, no una cancelacion falsa. Si aparece una solicitud de permisos de macOS, registrar solo el mensaje y la accion tomada.
4. En Whiteboard, probar **Save drawing...**, **Open drawing...** y **Export SVG...**. La ventana no debe quedar congelada mientras espera el selector. Comprobar que guardar/exportar excluye el video de la camara.

**Todavia No verificado en Mac despues del cambio:** ejecucion de estos cuatro pasos, permisos nativos de camara/TCC, `.app` empaquetada, MPS, benchmark de ONNX y pruebas de HD-10 con humanos. ONNX continua estrictamente opt-in segun [HD-12](hd12-onnx-experimental.md).

### Gate del 13 de octubre: controles de Whiteboard y segunda mano

Los cambios del 12-oct-2026 estan descritos en [Whiteboard controls](../changes/hand-d-v2-visualizer-controls-2026-10-12.md). **Contrato Linux sintético aprobado; Mac físico No verificado.** Tras hacer `git pull --ff-only` y reiniciar la app:

1. Mostrar ambas manos y mantener **Index Finger (Drawing/Right)** mientras la **Modifier/Left** hace **Ruler**: debe verse `Ruler active · straight line preview`, una recta anclada en el punto de inicio, sin zigzag; al salir del gesto la recta queda editable con Undo/Redo. Si no hay feedback, mirar `raw_gesture`/`stable_gesture` y salud del modelo (legacy no es accuracy validada).
2. Con **Thumb Up de Modifier/Left**, mover el pulgar a izquierda y derecha. Con Drawing Index Finger, el indicador **Brush** debe recorrer 1–20; con Drawing Fist debe verse **Eraser radius** 10–80. El HUD indica que se esta ajustando y el numero. El borrador tiene que seguir el **Middle Finger MCP (joint 9)**; el dibujo conserva **Index Finger Tip (joint 8)**. Repetir borrando sobre camara: no hay manchas blancas ni controles cubiertos.
3. En Pen y Wiggly, elegir dos colores distintos entre los seis swatches y ajustar grosor. Guardar, abrir y exportar SVG: cada trazo conserva color y grosor, la tinta borrada sigue siendo transparente y Wiggly mantiene sus tres frames con Reduced Motion estatico. La seleccion sigue accesible dentro del modal ampliado.
4. Alternar Camera/Clean, cambiar de Studio a Whiteboard, pulsar Restart camera. Documento y controles permanecen operativos incluso sin sidecar. Verificar que el input de un gesto no dibuja sobre Whiteboard cuando Studio esta visible.

No cambiar ni empaquetar ONNX/MPS para completar esta prueba: el gate es sobre interaccion y estabilidad, no sobre backend experimental. Si la clasificacion falla en un Mac donde los 21 puntos aparecen correctamente, registrar etiquetas/estado observados antes de tocar pesos o umbrales.

## Limitaciones que no deben confundirse con bugs ya resueltos

- El modo `npm --prefix desktop run dev` usa el Python local del clon. El empaquetado nativo ya tiene receta con PyInstaller y Rust ResourceDir, pero **solo existe prueba de construcción y lanzamiento del .deb Linux**. No hay todavía `.app` validado en Mac. Firma, sandbox, TCC/cámara del child PyInstaller y notarización permanecen pendientes de evidencia real.
- La geometría de overlay y cámara comparte plano de imagen, pero JPEG y eventos WS llegan a ritmos distintos; posible desfase temporal, todavía sin p95 end-to-end.
- Wiggly es un efecto visual propio, opt-in, sin sonidos, GIF ni grabación.
- HD-09 está **DONE para el alcance Linux implementado y empaquetado**, con pruebas automatizadas y sidecar con cámara real; el flujo de clicks GUI sobre datos reales y los paquetes macOS/Windows **no** están verificados. HD-10 sigue **TODO** (datos reales → entrenamiento → modelo verificado → Whiteboard). No declarar macOS Tier 1 validado antes de los resultados físicos.

Actualizar `docs/changes/hand-d-v2-tickets.md` después de la prueba de laboratorio con evidencia de cada Mac.
