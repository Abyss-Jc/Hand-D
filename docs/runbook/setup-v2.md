# Instalar y ejecutar Hand-D v2 desde cero

**Tipo:** Runbook / SETUP

**Revisado:** 2026-10-11

**Rama que contiene v2:** `feature/hand-d-v2-modernization`
**Estado de plataforma:** Linux x86_64: código, tests y empaquetado probados; macOS Apple Silicon: preflight disponible, pero cámara/Tauri nativo **no verificados físicamente** en el último corte; Windows: **no verificado**.

> **Empieza aquí si eres nuevo:** [handoff y estado del proyecto](../reference/handoff-v2-2026-10-11.md). Este documento instala **v2 (Tauri + Python)**, no la aplicación antigua Tkinter de `visualizer_app/main.py`.

## 1. Qué instalar en tu equipo

- **Git** para obtener el repositorio y colaborar.
- **Rust estable** (`rustc`, `cargo`, preferiblemente [rustup](https://rustup.rs/)); Tauri compila su shell nativo.
- **Node.js + npm** ([sitio oficial](https://nodejs.org/en/download)); npm instala la CLI Tauri según `desktop/package-lock.json`.
- **uv** ([instalación oficial](https://docs.astral.sh/uv/getting-started/installation/)); administra Python **3.13** y el entorno local según `pyproject.toml` / `uv.lock`.
- **Bibliotecas nativas de Tauri 2** de tu sistema operativo, más acceso a una webcam para las pruebas físicas (los tests normales no la necesitan).

No uses `requirements.txt`, `requirementsGPU.txt`, `pip install` global ni el `venv/` antiguo para trabajar en v2. El intérprete correcto será `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`). Rust, Node y uv se instalan una sola vez por equipo; las dependencias del proyecto se instalan por clon.

Si en Linux aún no tienes uv o Rust, sigue sus instaladores oficiales
(revisa la URL antes de ejecutar un script remoto):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Para Node/npm instala una versión estable compatible desde su [sitio oficial](https://nodejs.org/en/download).
Abre una terminal nueva después de instalar rustup o uv para recuperar el PATH.

### Linux: Ubuntu / Debian

Instala las dependencias de Tauri 2 (además de Rust, Node, uv y Git):

```bash
sudo apt update
sudo apt install -y git build-essential curl wget file pkg-config \
  libwebkit2gtk-4.1-dev libxdo-dev libssl-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf
```

### Linux: Arch / CachyOS

```bash
sudo pacman -Syu
sudo pacman -S --needed git base-devel curl wget file openssl \
  webkit2gtk-4.1 appmenu-gtk-module libappindicator-gtk3 \
  librsvg xdotool patchelf pkgconf
```

Los nombres y variantes de paquetes pueden cambiar con la distribución. **Fuente externa consultada:** [requisitos oficiales de Tauri 2](https://v2.tauri.app/start/prerequisites/). Estos comandos son una guía de instalación para un equipo nuevo; no se han ejecutado en el equipo del nuevo contribuidor.

### macOS Apple Silicon

Instala primero las herramientas de desarrollo de Apple:

```bash
xcode-select --install
```

Con [Homebrew](https://brew.sh/) disponible, puedes instalar Git/Node/uv:

```bash
brew install git node uv
```

Instala Rust con [rustup](https://rustup.rs/) y comprueba que `cargo` está disponible; abre una terminal nueva si el PATH no se actualiza. Dependiendo de tu entorno, puede ser necesario instalar/configurar Xcode completo. **No copies la carpeta `.venv`, el sidecar ni un binario Linux al Mac**: reconstruye allí.

Después de clonar y antes de abrir la cámara, ejecuta el preflight que ya existe:

```bash
bash scripts/mac-preflight.sh
```

El script comprueba uv, Node/npm, cargo, imports de Python, tests y `cargo check`; **no abre la cámara ni valida permisos macOS**. Continúa con [la lista de verificación Mac](mac-lab-oct09.md) cuando haya hardware disponible.

### Windows

La estructura del launcher contempla un Python local en `.venv/Scripts`, pero **este repositorio aún no tiene una instalación Tauri/sidecar/hardware Windows verificada**. No afirmes que el build funciona por contar con dependencias de Tauri; un contribuidor Windows deberá documentar su primera prueba nativa por separado.

## 2. Obtener exactamente la versión v2

En una carpeta de trabajo vacía, desde tu terminal:

```bash
git clone --branch feature/hand-d-v2-modernization \
  https://github.com/Abyss-Jc/Hand-D.git
cd Hand-D
git status
```

Debería mostrar la rama `feature/hand-d-v2-modernization` (sin cambios locales). No uses `main` ni supongas que un ZIP del branch principal contiene v2.

## 3. Instalar dependencias reproducibles

Desde **la raíz del repositorio**:

```bash
uv --version
cargo --version
node --version
npm --version

uv python install 3.13
uv lock --check
uv sync --frozen
npm ci --prefix desktop --no-audit --no-fund
```

- `uv sync --frozen` prepara `.venv` usando el lockfile, con Python >=3.13,<3.14. **No hace falta activar `.venv` manualmente**.
- `npm ci` usa `desktop/package-lock.json` sin seleccionar versiones nuevas.
- Linux resuelve **PyTorch CPU** por defecto. Tener GPU no implica que esta rama ya haya validado inferencia acelerada.
- Los modelos de diagnóstico `models/hand_landmarker.task` y `models/gesture_mlp.pth` están versionados en Git. El checkpoint legacy no es un modelo v2 con precisión validada.

**Perfil opcional de entrenamiento (pandas/scikit-learn/reportes):**

```bash
uv sync --frozen --group training
```

Al ejecutar un comando que dependa de esos extras, añade también `--group training` a `uv run`. No lances entrenamiento contra un workspace de datos reales sin acordar su procedencia.

## 4. Verificar la instalación sin cámara

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_*.py' -q
node --test desktop/tests/*.test.mjs
cargo test --manifest-path desktop/src-tauri/Cargo.toml --locked -q
npm --prefix desktop run check
```

**Referencia del corte anterior (2026-10-09):** 138 Python, 45 JavaScript, 7 Rust superados en Linux. Son cifras de ese corte, no una promesa para otros equipos. Si falta una librería de GTK/WebKit, vuelve a la sección de prerequisitos antes de modificar código.

## 5. Abrir Hand-D v2

```bash
npm --prefix desktop run dev
```

En desarrollo, Tauri ejecuta **el Python local `.venv`** del repositorio y su propio sidecar; no debes abrir manualmente un segundo proceso de cámara. Comprueba que aparece una ventana Hand-D con **Whiteboard** y **Studio**.

- **Whiteboard** funciona sin seleccionar workspace. Tiene cámara/overlay de 21 puntos, Pen, Eraser, Wiggly, Undo/Redo y guardado/exportación de dibujos.
- **Studio** ofrece Collect, Browse, Review, Snapshots, Models y Runtime. **Open / change workspace…** abre un selector nativo de carpetas; **Create workspace…** necesita **una carpeta existente y totalmente vacía**.
- La cámara, MJPEG y WebSocket se sirven localmente por `127.0.0.1`, con token temporal; no copies tokens a informes.
- Si el runtime deja de responder, comprueba el estado y usa **Restart camera**; el lienzo debe permanecer independiente de ese fallo.

Primero ejecuta un diagnóstico de cámara **sin almacenar datos**:

```bash
uv run --frozen python -m handd_core.camera_smoke --camera 0 --seconds 8
```

Si quieres ventana de preview, añade `--preview` (necesita una sesión gráfica). Si no hay manos/frames, comprueba permisos, índice de cámara, otra app usando la webcam y la [guía Mac](mac-lab-oct09.md), según corresponda. Un smoke de cámara exitoso **no demuestra exactitud del clasificador**.

## 6. Workspace de laboratorio: no mezclar código con datos

Un **Project Workspace** es una carpeta aparte del checkout: contiene `handd.sqlite`, `snapshots/`, `models/candidate-*/`, `reports/` y evaluaciones derivadas. No compartas el workspace real de otro usuario sin copia de seguridad y acuerdo explícito. La app no requiere workspace para dibujar.

Ruta fácil: crea una **carpeta nueva y vacía** fuera del repo y selecciónala en Studio → **Create workspace…**. Por CLI, usando un ejemplo local que no exista previamente:

```bash
mkdir -p "$HOME/HandD-Lab-Workspace"
uv run --frozen python -m handd_core.workspace_init \
  --workspace "$HOME/HandD-Lab-Workspace"
```

**Atención:** la CLI rechaza carpetas no vacías y no sobreescribe workspaces. La carpeta de ejemplo es para una instalación nueva; usa otro nombre si ya existe. Para un workspace existente usa **Open / change workspace…**, nunca **Create**.

### Primer piloto de recolección/revisión

1. En Studio → **Collect**, elige `P001`, un gesto real, mano física e inicia **10–20 Samples** (la cuota por defecto es 120 y se ajusta en opciones avanzadas). Asegura **los 21 landmarks dentro del encuadre**.
2. El nuevo filtro ignora observaciones de mano incompleta o fuera de cámara y avisa por qué. **Omitir callbacks antiguos para mantener la latencia baja no equivale a marcar Samples como `Dropped`.**
3. Abre **Review this Capture**. Con un primer modelo todavía no verificado, revisa la submuestra aleatoria de **QC** y confirma el resto mediante **Accept remaining batch** solo si las etiquetas se verificaron durante la toma. Cada aceptación queda auditada; una QC rechazada bloquea el lote.
4. Si hay un **Candidate v2 verificado**, Suggested for Review prioriza desacuerdos e incertidumbre en **muestras nuevas fuera de su Snapshot de entrenamiento**. Esto **no** es una aprobación automática ni prueba de precisión.
5. La vista 3D canónica debe mostrar **muñeca abajo**; rotación X/Y, comparación con coordenadas originales y navegación entre muestras. Se guardan landmarks, **no fotos ni video**.
6. Snapshots se construyen explícitamente, nunca al capturar. Training sigue siendo un comando CLI; Model → **Make Active** requiere confirmación.

### Entrenar solamente después de una curación real

```bash
uv run --frozen --group training python -m handd_core.train_cli \
  --workspace "$HOME/HandD-Lab-Workspace" \
  --snapshot "dev-ID-REAL-DEL-SNAPSHOT"
```

Reemplaza el ID por el que realmente generó **Snapshots**. Si no hay suficientes sesiones de desarrollo revisadas, el comando puede bloquearse con razón; no inventes datos ni participantes. **P003 se mantiene sellado para prueba final** y no entra en Development.

## 7. Empaquetado opcional

**No es necesario para desarrollar.** En cada sistema operativo/arquitectura hay que construir su propio sidecar:

```bash
bash scripts/build-sidecar-bundle.sh
```

Después, en Linux Debian/Ubuntu:

```bash
npm --prefix desktop run build -- --bundles deb
```

En macOS:

```bash
npm --prefix desktop run build -- --bundles app
```

El build `.deb` y un portable Linux se probaron anteriormente; la instalación/ejecución del `.app` y permisos de su child en macOS siguen **no verificados**. Las operaciones de firma/notarización tampoco están listas para declararse completadas. Ver [runbook extendido](development.md) y [laboratorio macOS](mac-lab-oct09.md).

## 8. Diagnóstico rápido

| Síntoma | Comprobar |
|---|---|
| `uv sync` falla o usa otra versión | `uv python install 3.13`, `uv lock --check` y acceso a red |
| `cargo` no existe | Rust/rustup y PATH de una terminal nueva |
| Tauri no compila en Linux | Paquetes WebKitGTK 4.1, GTK/Rust, SSL y AppIndicator de tu distribución |
| Ventana abre pero no aparece cámara | Permisos/ocupación de webcam, índice `--camera 0`, logs y `camera_smoke` |
| `RUNTIME FAILED - USE RESTART CAMERA` | `.venv`, assets en `models/`, salida del sidecar; reintenta manualmente |
| `Open workspace` falla | Selecciona una carpeta con **`handd.sqlite`**, no el checkout |
| `Create workspace` falla | Selecciona una carpeta **ya existente y vacía** |
| Review marca `Dropped` | Estado de lifecycle explícito: recuperable con `Restore`, **no** es un callback temporal perdido |
| No hay sugerencias ML | Sin Candidate verificable o sin datos fuera de entrenamiento: QC bootstrap, no inferir seguridad |
| No puedes formar Snapshot/Train | Verificar que existan Samples **accepted + active**, sesiones P001/P002 y folds elegibles |
| Windows/macOS build no probado | Recopila logs y sigue checklist de plataforma; no declares soporte validado |

## 9. Cómo contribuir con seguridad

- Antes de trabajar: `git fetch origin`, `git status`, relee [handoff](../reference/handoff-v2-2026-10-11.md) y [tickets](../changes/hand-d-v2-tickets.md).
- Evita editar `main`: crea una rama hija del trabajo v2, ejecuta tests y manda PR/revisión. Coordina los cambios del mismo branch con la otra persona.
- Respeta `AGENTS.md`, especialmente UI del producto completamente en **inglés**, datos brutos inmutables, P003 sellado, pruebas TDD y CPU funcional.
- No subas `.venv/`, `node_modules/`, resultados grandes de bundle, SQLite real, identificadores privados de dispositivo ni material de cámara.
- Una instalación correcta no equivale a aceptación de cámara/UI ni a buen rendimiento del clasificador en humanos.

**Fuente de código y contratos:** `pyproject.toml:1-37`, `desktop/package.json:1-17`, `desktop/src-tauri/src/main.rs:75-165`, `handd_core/workspace_init.py:10-30`, `handd_core/studio_collection.py:17-145`, `handd_core/capture_sampling.py:77-129`, `handd_core/train_cli.py:10-56`, `scripts/mac-preflight.sh:1-29`, `scripts/build-sidecar-bundle.sh:1-24`. **Fuente de paquetes de sistema:** [Tauri 2 — Prerequisites](https://v2.tauri.app/start/prerequisites/).
