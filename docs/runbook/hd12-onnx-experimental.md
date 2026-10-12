# HD-12 - evaluación experimental de ONNX y MediaPipe 1.1

**Estado:** IN PROGRESS / EXPERIMENTAL. Este experimento **no activa ONNX ni actualiza MediaPipe en producción**. El backend de inferencia sigue siendo PyTorch CPU, con MediaPipe `0.10.33` bloqueado en `pyproject.toml` y `uv.lock`. No hay que modificar un Candidate ni un Project Workspace para evaluar los backends.

**Fecha de los datos:** 2026-10-12. **Host medido:** CachyOS x86_64 / Python 3.13.16 / PyTorch 2.14.1+cpu / ONNX 1.23.2 / ONNX Runtime 1.31.0 / onnxscript 0.7.2, CPU (1 thread). **Comparaciones macOS Apple Silicon y gestos humanos: pendientes.**

## Instalación opcional sin alterar la app

No añadir estos paquetes a las dependencias normales de `[project]` ni actualizar `uv.lock` para el spike. `uv run --frozen --with` instala extras en un entorno temporal gestionado por uv; no convierte ONNX en una dependencia del sidecar. Con un Candidate existente, la instalación experimental reproducible es:

```bash
uv run --frozen \
  --with 'onnx==1.23.2' \
  --with 'onnxruntime==1.31.0' \
  --with 'onnxscript==0.7.2' \
  python -m handd_core.benchmark_inference \
  --candidate "/ruta/al/workspace/models/candidate-ID" \
  --warmup 40 --iterations 250 > hd12-onnx-report.json
```

Cambia `candidate-ID` por un directorio de Candidate real con `manifest.json`, `weights.pth` y `metrics.json`; la rutina valida la procedencia mediante el loader del artefacto y aborta si el checksum no coincide. Si aún no hay Candidate, se pueden ejecutar únicamente las pruebas sintéticas:

```bash
uv run --frozen \
  --with 'onnx==1.23.2' \
  --with 'onnxruntime==1.31.0' \
  --with 'onnxscript==0.7.2' \
  python -m unittest tests.test_hd12_benchmark -q
```

**En macOS M4:** ejecuta los mismos comandos desde el checkout de Hand-D en la Mac después de `uv sync --frozen`. El runtime que ejecutas con `npm --prefix desktop run dev` **no necesita** esos paquetes. Guarda los reportes fuera de un workspace real si los vas a comparar; no subas datos sensibles ni archivos `handd.sqlite`. En caso de que uv no encuentre un wheel para tu arquitectura/versión de Python, registra el error en vez de instalar dependencias globales o cambiar los pins de la aplicación.

Si más adelante se decide **mantener una implementación ONNX permanente**, antes de cambiar producción habrá que agregar un grupo opcional, por ejemplo `[dependency-groups].onnx-benchmark`, a `pyproject.toml`, actualizar y versionar `uv.lock`, y ejecutar las matrices de CI y packaging. **No hacerlo ahora** evita sumar `onnxruntime` al bundle o provocar un cambio de inferencia sin pruebas suficientes.

## Qué valida el código experimental

- `handd_core/benchmark_inference.py` carga un Candidate comprobando hashes, exporta su MLP a ONNX mediante `torch.export` (dynamo, opset 18, batch dinámico), comprueba el grafo y exige `CPUExecutionProvider`.
- Compara los logits y la clase predicha de PyTorch y ONNX Runtime sobre los **mismos features float32 sintéticos de 69 valores**, lotes 1, 8 y 32. Los datos se generan con una semilla reproducible; **no proceden de personas ni de un Snapshot**.
- Registra medianas y p95 de inferencia pura (después de warmup), tiempo de carga del modelo, inicio de la sesión ONNX y exportación. No mide callbacks MediaPipe, render Tauri, cámara ni IPC.
- Cuenta tanto `model.onnx` como los pesos externalizados `model.onnx.data`, y los tamaños de las distribuciones Python instaladas. **Estos últimos NO son tamaños de ejecutables empaquetados.** El export ONNX es un proceso de evaluación en una carpeta temporal: ni el Candidate ni la activación actual se modifican.
- En la suite ordinaria, los tests ONNX se saltan si no están sus dependencias opcionales. Se ejecutan completos con el comando `--with` anterior.

## Resultado del primer benchmark CPU sintético

MLP de tres clases con pesos sintéticos, semilla `2026`, 40 warmups, 250 iteraciones por tamaño de lote, un hilo CPU por backend:

| Batch | PyTorch mediana / p95 (ms) | ONNX Runtime mediana / p95 (ms) |
|---|---:|---:|
| 1 | 0.05654 / 0.14124 | 0.01197 / 0.02349 |
| 8 | 0.10775 / 0.20410 | 0.02532 / 0.02895 |
| 32 | 0.10626 / 0.16393 | 0.03084 / 0.06167 |

**Equivalencia:** 0 discrepancias en la clase predicha, máximo error absoluto de logits **1.49×10⁻⁸**. Modelo: `weights.pth` 72,341 bytes frente a ONNX grafo + datos externos **76,842 bytes** (7,466 + 69,376). Distribuciones instaladas (no empaquetadas): PyTorch **715,498,121 bytes**; onnxruntime **64,529,736 bytes**. Export ONNX ~2.7 s; carga inicial PyTorch y sesión ONNX no son comparables a arranque de un proceso completo porque el intérprete/import ya están calientes.

**Conclusión limitada:** ONNX Runtime fue más rápido en inferencia CPU sintética de este MLP pequeño. Esto **no prueba** un beneficio perceptible de FPS/latencia completa de Hand-D, mejora de clasificación en humanos, tamaño menor del bundle Tauri ni estabilidad Mac. **No hay razón suficiente para sustituir PyTorch todavía.**

## MediaPipe 1.1.0: prueba opt-in sin webcam

Para probar la versión actual congelada (`0.10.33`):

```bash
HANDD_HD12_MP_LIVE=1 uv run --frozen \
  python -m unittest tests.test_hd12_mediapipe_spike -q
```

Para probar `1.1.0` en **otro entorno temporal** sin tocar `pyproject.toml`, `uv.lock` ni la `.venv` de producción:

```bash
HANDD_HD12_MP_LIVE=1 uv run --no-project --python 3.13 \
  --with 'mediapipe==1.1.0' --with 'numpy>=2.2,<3' \
  python -m unittest tests.test_hd12_mediapipe_spike -q
```

El test abre el mismo `models/hand_landmarker.task` en modo `LIVE_STREAM`, envía dos imágenes negras de prueba (128×128), comprueba timestamps, estructura de salida y contrato del Feature Transform de 69 valores. No abre una webcam, ni confirma handedness/gestos reales. **Puede requerir inicialización de GPU/Metal/EGL aunque no use cámara**. Por ello el test LIVE_STREAM requiere `HANDD_HD12_MP_LIVE=1` y no bloquea la suite CI normal; la prueba de normalización NumPy sigue ejecutándose. Ambas versiones pasaron esta prueba aislada en Linux el 2026-10-12. Los avisos internos de TFLite sobre `NORM_RECT` y feedback tensors aparecieron y **no significaron fallo del contrato**.

En macOS M4 aún debemos verificar webcam, permisos, interpretación Left/Right, recorte, callbacks bajo carga, consumo de recursos y bundle firmado. Que el log diga `Running target/debug/hand-d-desktop` o `TensorFlow Lite XNNPACK delegate` no demuestra reconocimiento correcto.

## Criterios pendientes para considerar una migración

1. Comparar sobre **Candidate real y verificado** (sin revelar muestras sensibles), con varias clases y feeds de features representativos, comprobando límites de tolerancia y orden de labels.
2. Medir callback→inferencia→WS→Tauri en Linux y Mac M4 con cámara real, guardando p50/p95 y la latencia percibida; priorizar fluidez sin perder la cámara por una optimización prematura.
3. Construir y lanzar los sidecars PyTorch y ONNX en macOS ARM64 y Linux, comparar bytes reales del bundle, dependencias nativas, startup y fallback seguro CPU.
4. Verificar que MediaPipe 1.1.x preserva handedness, 21 landmarks, timestamps, comportamiento de tracking y los contratos de captura, especialmente ante pérdida temporal de cámara.
5. Tomar la decisión explícita en un nuevo ticket/ADR. **Por ahora HD-12 permanece abierto como spike opcional; HD-10 validación humana también sigue aparte.**
