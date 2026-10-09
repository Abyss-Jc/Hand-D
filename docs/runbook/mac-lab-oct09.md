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
   Registrar `CAMERA_OPEN`, `CAPTURE_FPS`, callbacks, muestras con manos y errores. `--preview` es opcional; Qt puede comportarse distinto en Macs.
2. Cerrar el probe y lanzar la app (la **UI está completamente en inglés**, ADR 0005):
   ```bash
   npm --prefix desktop run dev
   ```
   Verificar Tauri, estado `SIDECAR / CONNECTED`, MJPEG sobre cámara y diagnóstico `LEGACY MODEL ACTIVE`. El checkpoint legacy sirve solo como diagnóstico: las etiquetas no tienen validación independiente.
3. Mostrar la **mano derecha, después izquierda y después ambas**. Verificar 21 puntos por mano, conexiones y colores (lima Drawing, azul Modifier). Confirmar manualmente el espejo y los roles; no asumir que la inversión física está probada para la webcam Mac.
4. Alternar **Hide hands / Show hands** y **Camera overlay / Clean canvas**: debe conservarse el mismo documento y no deben detenerse MediaPipe ni los eventos WS. En limpio el MJPEG debe dejar de transmitirse al no tener consumidor. Abrir y cerrar el modal ampliado con Escape.
5. Dibujar con ratón o trackpad: trazo largo y rápido, **Undo/Redo**, **Pen**, **Eraser** y **Wiggly** sobre cámara y en limpio. **Regresión crítica de borrador:** dibujar una línea, borrarla parcialmente encima de la cámara; comprobar que reaparece el video real (NO una mancha blanca), que los controles inferiores nunca se tapan ni dejan de responder, que se puede volver a dibujar sobre la zona borrada y que Undo/Redo restaura la tinta. Repetir sobre Clean canvas y sobre un trazo Wiggly. Wiggly debe moverse visiblemente más que antes, sin desplazar los puntos guardados. Con **Ajustes del Sistema → Accesibilidad → Pantalla → Reducir movimiento**, Wiggly debe quedarse estático sin perder puntos.
6. Probar `Index_Finger` y `Fist` y registrar etiquetas observadas **sin atribuirles precisión**. Los gestos sobre cámara no necesitan video almacenado.
7. Pulsar **Restart camera**: sidecar debe reconectarse con sesión nueva, el documento debe sobrevivir y el overlay no debe dejar manos fantasma.
8. Si existe un proyecto de pruebas con `handd.sqlite`, entrar a **Studio → Existing project workspace directory → Open workspace** usando su ruta local. Verificar que **Review** muestra conteos y Sample IDs reales, que **Accept/Reject/Drop/Restore** requieren clicks explícitos y que **Build Development Snapshot** no entrena ni incluye P003. No hacer acciones sobre un dataset real sin intención/backup. Sin proyecto, Studio debe mostrar un empty state honesto y Whiteboard debe seguir funcionando.

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
| Gestos reales / fluidez percibida | No verificado | No verificado |

No compartir tokens del sidecar, seriales, fotos ni video; solo comandos, errores y medidas agregadas.

## Limitaciones que no deben confundirse con bugs ya resueltos

- `desktop/src-tauri/src/main.rs` ejecuta el Python del clon local. `npm --prefix desktop run dev` es una prueba **desde el repositorio**; aún **no** existe un `.app` empaquetado para distribución. Firma, sandbox, permisos de cámara del proceso Python y notarización quedan por verificar en hardware.
- La geometría de overlay y cámara comparte plano de imagen, pero JPEG y eventos WS llegan a ritmos distintos; posible desfase temporal, todavía sin p95 end-to-end.
- Wiggly es un efecto visual propio, opt-in, sin sonidos, GIF ni grabación.
- HD-09 sigue **IN PROGRESS** (Studio Collect/Review/Snapshot y empaquetado); HD-10 sigue **TODO** (datos reales → entrenamiento → modelo verificado → Whiteboard). No declarar macOS Tier 1 validado antes de los resultados físicos.

Actualizar `docs/changes/hand-d-v2-tickets.md` después de la prueba de laboratorio con evidencia de cada Mac.
