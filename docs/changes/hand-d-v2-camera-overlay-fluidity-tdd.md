# Hand-D v2 — Camera-first, hands overlay y fluidez (plan TDD)

**Estado:** planificación aprobada; sin implementación ni benchmarking nuevos.
**Decisiones:** UX-Q116=A, UX-Q117=A (2026-10-08); opción creativa inspirada en Wigglypaint solicitada (2026-10-08).
**Cobertura:** Whiteboard Tauri, runtime Python y canal transitorio de landmarks; no modificar Studio, entrenamiento, SQLite ni el modelo.
**Documentos de autoridad:** [Requirements](../requirements/hand-d-v2.md), [Architecture](../reference/architecture.md), [UX Shape](../design/hand-d-v2-ux-shape.md).

## Decisiones y límites

1. **UX-Q116=A — ambas manos visibles.** En Cámara mostrar los 21 puntos + conexiones anatómicas por mano. Colores contrastantes distintos para **Drawing Hand** y **Modifier Hand**, identificados por rol, no por una inferencia visual de identidad. Toggle **Mostrar manos / Ocultar manos**: solo presentación; nunca suspender el modelo ni alterar acciones.
2. **UX-Q117=A — guardar/exportar.** **Guardar** conserva el documento nativo **editable**, no una foto del stream. **Exportar** produce por defecto solo los trazos, sin fondo de cámara, independientemente del modo. Una futura exportación que incorpore un fotograma de cámara requerirá selección **explícita**, consentimiento apropiado y un cambio de alcance; **no se implementará ahora**.
3. **Mantener camera-first.** **Cámara** muestra el feed detrás del mismo dibujo; **Lienzo limpio** oculta el video, sin parar inferencia/tracking, perder trazos ni cambiar documentos/undo. El modal inmersivo no crea otra sesión ni otra superficie de dibujo. La indisponibilidad de cámara no bloquea el ratón.
4. **Nueva opción — pincel animado tipo Wiggly (referencia visual, NO implementación externa).** Agregar una herramienta **opt-in** de trazo animado, junto al lápiz normal y borrador, disponible con mouse o gestos sobre ambos fondos. Su animación modifica solo la **presentación**, nunca puntos originales, Feature Transform, posición del dedo, acciones o historial. Lápiz normal sigue siendo el predeterminado. Una primera variante **Wiggle suave** (sin sonido por defecto) permite comprobar experiencia y presupuesto antes de más pinceles, paletas o efectos.
5. **Movimiento accesible y rendimiento.** Animación desactivable, sin movimiento si se solicita **reduced motion**, pausada en superficies ocultas/minimizadas. Undo/redo, guardar/abrir y redimensionar deben conservar estilo y geometría estable sin convertir una secuencia de frames en el documento.
6. **WigglyPaint como inspiración, no dependencia.** [Referencia compartida](https://wigglypaint.net/): líneas que oscilan y exportación GIF. La página consultada declara que es un sitio de aficionados no afiliado al programa original; no asumir licencia de código, marca, sonidos o assets. Diseñar efectos propios, sin copiar material de terceros. **GIF animado opcional es trabajo futuro** dentro de Export, nunca reemplaza la exportación limpia ni altera la decisión UX-Q117=A.

La cantidad de pinceles extra, los sonidos y el formato/frecuencia de exportación animada **no están especificados**. No crear esos requisitos por anticipado.

## Observaciones del código (no son benchmarks)

| Riesgo | Evidencia leída | Hipótesis que debe medirse |
|---|---|---|
| Dibujo en una superficie y video en panel independiente | `desktop/web/index.html:32,44` | El modo Cámara no respeta todavía la composición camera-first |
| Cada nuevo punto reconstruye todos los SVG | `desktop/web/app.js:28-60` (`redraw`) | El coste DOM puede crecer con la longitud del documento y producir jank |
| La vista previa comprime solo cada segundo frame | `handd_core/sidecar_main.py:88-97` | La cadencia visible queda limitada respecto a la cámara; medir antes de ajustar |
| No existe payload de los 21 puntos para el frontend | `handd_core/runtime_v2.py:~190-235` | Falta contrato transitorio para el esqueleto; no duplicar MediaPipe |
| Superficie legacy sí dibujaba sobre cámara | `visualizer_app/main.py` (loop de dibujo) | Referencia de interacción, no referencia de arquitectura ni de rendimiento |
| La resolución del gesto puede estar degradada | `docs/changes/hand-d-v2-tickets.md` (HD-09 connection/recognition regression) | Medir cámara, callbacks con mano, inferencia, WS y render por separado antes de culpar al preview |

## Contratos propuestos para acordar antes de escribir tests

- **Runtime (Python):** `GestureRuntime.process_result` emite como máximo dos manos etiquetadas por rol físico, 21 coordenadas normalizadas x/y válidas cada una, orden anatómico fijo, tiempo y `runtime_session_id`. Los datos se descartan cuando pierden frescura. No se vuelven Samples de SQLite.
- **Transporte:** WebSocket con eventos más recientes, secuencia y Session ID; MJPEG entrega frames aparte. El consumidor descarta eventos viejos y no bloquea inferencia esperando el próximo JPEG.
- **Visualizador frontend:** un documento editable y un solo motor de trazos; capas de video (opcional), esqueleto (opcional), trazos/cursor y controles. Alternar Cámara/Limpio y Mostrar/Ocultar manos no muta el documento.
- **Geometría:** la conversión desde coordenadas de imagen a vista tiene en cuenta espejo, aspect ratio, letterbox/crop y resize. La misma calibración aplica al video, esqueleto y posición del dedo. Un trazo dibujado sobre Cámara conserva su posición semántica en Lienzo limpio.
- **Pincel Wiggly:** los datos guardados contienen geometría editable + identificador de estilo y parámetros limitados; el desplazamiento animado se deriva de ellos/tiempo con semilla determinista si es necesario, nunca se acumula en el trazo original. Reducir o desactivar movimiento muestra un trazo estable. Export estático limpia el fondo; no se requiere GIF en este corte.

Estos son los **seams TDD propuestos** para aprobación al entrar en implementación; no se han creado tests ni cambiado firmas durante esta fase documental.

## Plan vertical TDD — un RED → GREEN a la vez

| Orden | Primer comportamiento observable que debe fallar | Implementación mínima al pasar a GREEN |
|---|---|---|
| 0. Establecer baseline | Reproducir un escenario fijo de trazo largo con cámara 640×480 y ambos roles; medir sin inventar cifras | Registrar hardware/cámara, FPS fuente y preview, callbacks con manos, coste DOM/render, tiempo desde resultado MediaPipe hasta estado frontend. Mantener un benchmark repetible; documentar datos medidos o **No verificado** |
| 1. Mano completa | Test público de `GestureRuntime`: dos manos generan 21 puntos cada una, roles estables y desaparición sin residuos; eventos stale descartados | Añadir un payload transitorio versionado con campos de x/y, sin cambiar acción ni almacenamiento |
| 2. Una superficie Camera-first | Test UI: modo Cámara muestra video detrás del mismo documento y Lienzo limpio lo oculta; alternar no reinicia sesión ni afecta undo/redo | Ubicar preview como fondo del Whiteboard, no en panel separado; mantener control de cámara independiente |
| 3. Esqueleto superpuesto | Test UI: ambas manos de 21 puntos y conexiones correctas, colores por rol; toggle Mostrar manos modifica solo esa capa | Renderer de overlay ligero, anclado al resultado válido más reciente, sin duplicar inferencia |
| 4. Geometría sin deriva | Casos conocidos: borde, centro, espejo, cámara 4:3 en vista ancha, redimensionar, modal; posiciones de puntero/landmarks coinciden con video | Transformación central única incluyendo recorte y escala, validada por fixtures geométricos |
| 5. Dibujo incremental | Test UI: cientos/miles de puntos mantienen las rutas anteriores y añaden puntos a la ruta activa sin reconstruir todos los nodos; undo/redo sin regresión | Agrupar actualizaciones en `requestAnimationFrame` y actualizar solo trazos afectados; preservar estado y cursor |
| 6. Preview independiente | Test transporte: ocultar Camera/MJPEG detiene trabajo de encoding cuando no hay consumidores, sin parar MediaPipe/WS; backlog nunca crece ilimitado | Preview a demanda con últimos frames y degradación controlada antes que afectar inferencia |
| 7. Pincel Wiggly opcional | Test visual/funcional: mismo trazo editable admite estilo animado, dibujo normal sin cambio, undo/redo funciona; reduced-motion y pausa detienen solo la animación | Primer estilo propio de wiggle determinista, opt-in, bajo coste y respetuoso de movimiento reducido |
| 8. Export limpio | Test Save/Open conserva geometría y estilo; Export produce dibujo sin pixels de cámara aun en Camera; futura exportación GIF/composición **no disponible** por defecto | Contratos separados para documento y exportación; evitar capturar/reproducir frames de la webcam |
| 9. Hardware y regresiones | Smoke en CachyOS real con mano izquierda/derecha, Camera/Limpio, modal, un gesto Draw/Erase estable, movimiento rápido/largo, pérdida de cámara y reinicio sidecar | Comparar baseline vs nuevo, captura de logs de salud y métricas sin grabar frames; no declarar completado sin evidencias |

### Métricas y definición de terminado

El plan **no fija cifras nuevas** sin evidencia; reutiliza los presupuestos ya aprobados de `docs/reference/architecture.md`:

- **Preview** visible objetivo **30 FPS** estable, suelo normal **24 FPS**; opción 60 FPS solo cuando la cámara y el equipo lo soportan sin empeorar inferencia.
- **Inferencia**: >=30 Hz preferido cuando el hardware permite; <20 Hz sostenido se declara degradado. **Render frontend**: normalmente 60 Hz, sin fingir tasa de inferencia a partir de `requestAnimationFrame`.
- **Respuesta tras MediaPipe** (transformación → predicción → estado temporal → IPC → frontend): **p95 <50 ms preferido** y **p95 <100 ms techo de milestone**. Tiempos de cámara/sensor se registran por separado; no usar p95 de Python como si fuera extremo a extremo.
- Comprobar FPS de preview, variación de inter-frame, p50/p95 de render, input-to-visual p95, tasa de callbacks entregados/descartados y latencia del gesto; medir **antes y después** en el mismo escenario, incluyendo cientos/miles de puntos y Brush Wiggly ON/OFF.
- **Salida segura:** datos y Saved native editables y compatibles, ninguna foto o video grabada, fondo fuera de Export, roles y acciones intactos, movimiento reducido respetado, sin cambios a `main` ni al modelo/dataset.

### Rollback, prioridades y fuera de alcance

- Cada slice es reversible; el modo normal debe seguir funcionando si el efecto animado o el overlay quedan deshabilitados. No sustituir MJPEG por WebRTC sin evidencia de benchmark ni introducir un motor de animación que dependa de MediaPipe.
- **Bloqueante para corregir experiencia:** slices 0–6 y 9, con reconocimiento/WS estables. **Mejora creativa opt-in:** slices 7–8; no debe retrasar una corrección de jank ni habilitar accidentalmente captura de cámara.
- Fuera de este corte: copiar código/assets/sonidos originales de WigglyPaint, GIF animado, timeline, biblioteca de pinceles extensa, cambios en Studio, alteraciones a Feature Transform/dataset, validación de generalización P003 y empaquetado de release.

**Evidencia de esta iteración documental:** lectura de código, requisitos y referencia externa, sin ejecutar benchmarks, tests ni cámara y sin modificar implementación. La worktree mantiene un cambio ajeno sin confirmar en `docs/design/hand-d-v2-ux-shape.md`, que no debe mezclarse con este plan.
