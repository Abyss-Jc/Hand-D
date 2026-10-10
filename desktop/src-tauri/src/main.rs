//! HD-09 native supervisor and explicit workspace/document commands.
mod native_files;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SidecarReady {
    #[serde(rename = "type")]
    kind: String,
    host: String,
    port: u16,
    token: String,
    runtime_session_id: String,
    workspace: Option<String>,
}

#[derive(Clone, Default)]
struct Supervisor {
    ready: Arc<Mutex<Option<SidecarReady>>>,
    failure: Arc<Mutex<Option<String>>>,
    restart: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    active_child: Arc<Mutex<Option<Child>>>,
    workspace: Arc<Mutex<Option<PathBuf>>>,
    portable_root: Arc<Mutex<Option<PathBuf>>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum SidecarStatus {
    Ready(SidecarReady),
    Failed { error: String },
}

fn wait_for_sidecar_ready<R: Read + Send + 'static>(
    stdout: R, state: &Supervisor, timeout: Duration,
) -> Option<SidecarReady> {
    // The reader can block on Python; the supervisor still needs to handle retries.
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let ready = reader.read_line(&mut line).ok()
            .filter(|&n| n > 0)
            .and_then(|_| serde_json::from_str::<SidecarReady>(&line).ok());
        let _ = tx.send(ready);
        let _ = std::io::copy(&mut reader, &mut std::io::sink());
    });
    let deadline = Instant::now() + timeout;
    loop {
        if state.shutdown.load(Ordering::SeqCst) || state.restart.load(Ordering::SeqCst) {
            return None;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return None; }
        match rx.recv_timeout(remaining.min(Duration::from_millis(100))) {
            Ok(Some(ready)) if ready.kind == "sidecar.ready"
                && ready.host == "127.0.0.1"
                && ready.port > 0 && !ready.token.is_empty()
                && !ready.runtime_session_id.is_empty() => return Some(ready),
            Ok(_) | Err(RecvTimeoutError::Disconnected) => return None,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

fn source_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bundled_sidecar(resources: &PathBuf) -> Option<PathBuf> {
    let suffix=if cfg!(windows) {"handd-sidecar.exe"} else {"handd-sidecar"};
    [
        resources.join("sidecar/handd-sidecar").join(suffix),
        resources.join("sidecar").join(suffix),
    ].into_iter().find(|path|path.is_file())
}

fn sidecar_program(resources: Option<&PathBuf>, packaged: bool) -> Result<(Command, bool), String> {
    if packaged {
        if let Some(resources)=resources {
            if let Some(binary)=bundled_sidecar(resources) {
                let mut command=Command::new(binary);
                command.current_dir(resources);
                return Ok((command, true));
            }
        }
        // Never silently fall back to the developer's compile-time home
        // directory when an installed release is missing its bundled sidecar.
        return Err("The packaged Hand-D Python sidecar is missing".into());
    }
    // Tauri dev *always* uses local source Python, even after a packaged
    // sidecar was built. Changes to Python must not silently run stale code.
    let root=source_root();
    let python=if cfg!(windows) {root.join(".venv/Scripts/python.exe")}
                else {root.join(".venv/bin/python")};
    let mut command=Command::new(python);
    command.current_dir(root);
    Ok((command, false))
}

#[tauri::command]
fn sidecar_status(state: tauri::State<'_, Supervisor>) -> Option<SidecarStatus> {
    if let Some(ready) = state.ready.lock().expect("state lock").clone() {
        return Some(SidecarStatus::Ready(ready));
    }
    state.failure.lock().expect("failure lock").clone()
        .map(|error| SidecarStatus::Failed { error })
}

#[tauri::command]
fn restart_sidecar(state: tauri::State<'_, Supervisor>) {
    state.restart.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn select_workspace(path: String, state: tauri::State<'_, Supervisor>) -> Result<String, String> {
    apply_workspace(path, &state)
}

fn apply_workspace(path: String, state: &Supervisor) -> Result<String, String> {
    // This command runs only from the trusted local Tauri webview. It does
    // not create new databases and never changes a workspace implicitly.
    let canonical = PathBuf::from(path).canonicalize()
        .map_err(|_| "Workspace directory was not found".to_string())?;
    if !canonical.is_dir() || !canonical.join("handd.sqlite").is_file() {
        return Err("Select a Hand-D workspace containing handd.sqlite".into());
    }
    *state.workspace.lock().expect("workspace lock") = Some(canonical.clone());
    state.restart.store(true, Ordering::SeqCst);
    Ok(canonical.to_string_lossy().to_string())
}

#[tauri::command]
fn pick_workspace(state: tauri::State<'_, Supervisor>) -> Result<Option<String>, String> {
    let Some(path)=native_files::pick_folder()? else {return Ok(None)};
    apply_workspace(path.to_string_lossy().to_string(),&state).map(Some)
}

#[tauri::command]
fn create_workspace(state: tauri::State<'_, Supervisor>) -> Result<Option<String>, String> {
    let Some(path)=native_files::pick_folder()? else {return Ok(None)};
    let canonical=path.canonicalize().map_err(|e|e.to_string())?;
    if !canonical.is_dir() || canonical.read_dir().map_err(|e|e.to_string())?.next().is_some() {
        return Err("Choose an existing empty directory for a new workspace".into());
    }
    let portable=state.portable_root.lock().expect("resource lock").clone();
    let (mut command, frozen)=sidecar_program(portable.as_ref(),!cfg!(debug_assertions))?;
    if frozen { command.arg("--init-workspace"); }
    else { command.args(["-m","handd_core.workspace_init"]); }
    let output=command.arg("--workspace").arg(&canonical).output()
        .map_err(|e|format!("Could not initialize workspace: {e}"))?;
    if !output.status.success(){
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    apply_workspace(canonical.to_string_lossy().to_string(),&state).map(Some)
}

#[tauri::command]
fn save_drawing(document: String) -> Result<Option<String>, String> {
    native_files::save_drawing(document)
}

#[tauri::command]
fn open_drawing() -> Result<Option<native_files::DrawingOpen>, String> {
    native_files::open_drawing()
}

#[tauri::command]
fn export_svg(svg: String) -> Result<Option<String>, String> {
    native_files::export_svg(svg)
}
fn spawn_supervisor(state: Supervisor) {
    thread::spawn(move || {
        let mut failures = 0u8;
        while !state.shutdown.load(Ordering::SeqCst) {
            if failures >= 2 {
                *state.failure.lock().expect("failure lock") =
                    Some("RUNTIME FAILED - USE RESTART CAMERA".into());
                while !state.shutdown.load(Ordering::SeqCst)
                    && !state.restart.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(250));
                }
                if state.shutdown.load(Ordering::SeqCst) { break; }
                state.restart.store(false, Ordering::SeqCst);
                failures = 0;
            }
            *state.ready.lock().expect("state lock") = None;
            *state.failure.lock().expect("failure lock") = None;
            // Kill the Python process directly, not a uv wrapper which could
            // leave its child orphaned when the desktop exits.
            let portable=state.portable_root.lock().expect("resource lock").clone();
            let (mut command, frozen)=match sidecar_program(
                portable.as_ref(),!cfg!(debug_assertions)) {
                Ok(launcher)=>launcher,
                Err(error)=>{
                    eprintln!("Hand-D packaged sidecar unavailable: {error}");
                    failures += 1;
                    thread::sleep(Duration::from_millis(250));
                    continue;
                }
            };
            if frozen {
                if let Some(resources)=portable.as_ref() {
                    command.arg("--task").arg(resources.join("models/hand_landmarker.task"));
                    // Diagnostic legacy predictor only: Sidecar reports
                    // legacy_unverified until a verified v2 Candidate is
                    // explicitly selected. Never present as evaluated v2.
                    let legacy=resources.join("models/gesture_mlp.pth");
                    if legacy.is_file() {
                        command.arg("--legacy-checkpoint").arg(legacy);
                    }
                }
            } else {
                command.args([
                    "-u", "-m", "handd_core.sidecar_main",
                    "--legacy-checkpoint", "models/gesture_mlp.pth",
                ]);
            }
            command.stdin(Stdio::null())
                .stdout(Stdio::piped()).stderr(Stdio::inherit());
            if let Some(workspace) = state.workspace.lock().expect("workspace lock").clone() {
                command.arg("--workspace").arg(workspace);
            }
            let started = command.spawn();
            match started {
                Ok(mut child) => {
                    let stdout = child.stdout.take();
                    *state.active_child.lock().expect("child lock") = Some(child);
                    let started_at = Instant::now();
                    let ready = stdout.and_then(|stdout|
                        wait_for_sidecar_ready(stdout, &state, Duration::from_secs(45)));
                    let launched = ready.is_some();
                    *state.ready.lock().expect("state lock") = ready;
                    if !launched {
                        eprintln!("Hand-D sidecar did not announce READY within 45 seconds");
                    }
                    let mut manual_restart = false;
                    while !state.shutdown.load(Ordering::SeqCst) {
                        if state.restart.swap(false, Ordering::SeqCst) {
                            manual_restart = true;
                            break;
                        }
                        if !launched { break; }
                        let mut child = state.active_child.lock().expect("child lock");
                        match child.as_mut().map(|process| process.try_wait()) {
                            None | Some(Ok(Some(_))) | Some(Err(_)) => break,
                            Some(Ok(None)) => {
                                drop(child);
                                thread::sleep(Duration::from_millis(250));
                            }
                        }
                    }
                    if let Some(mut child) = state.active_child.lock().expect("child lock").take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    *state.ready.lock().expect("state lock") = None;
                    if manual_restart || state.restart.swap(false, Ordering::SeqCst) {
                        failures = 0;
                    } else if !state.shutdown.load(Ordering::SeqCst) {
                        failures = if launched && started_at.elapsed() >= Duration::from_secs(30) {
                            1
                        } else {
                            failures.saturating_add(1)
                        };
                    }
                }
                Err(error) => {
                    eprintln!("Hand-D sidecar could not start: {error}");
                    failures = failures.saturating_add(1);
                }
            }
            for _ in 0..2 {
                if state.shutdown.load(Ordering::SeqCst) {
                    break;
                }
                thread::sleep(Duration::from_millis(250));
            }
        }
    });
}

fn main() {
    let supervisor = Supervisor::default();
    let run_state = supervisor.clone();
    tauri::Builder::default()
        .manage(supervisor.clone())
        .setup(move |app| {
            *supervisor.portable_root.lock().expect("resource lock") =
                app.path().resource_dir().ok();
            spawn_supervisor(supervisor.clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar_status, restart_sidecar, select_workspace, pick_workspace,
            create_workspace, save_drawing, open_drawing, export_svg
        ])
        .build(tauri::generate_context!())
        .expect("error building Hand-D desktop")
        .run(move |_handle, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                run_state.shutdown.store(true, Ordering::SeqCst);
                if let Some(mut child) = run_state.active_child.lock().expect("child lock").take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}

#[cfg(test)]
mod supervisor_tests {
    use super::*;
    use std::io::Cursor;
    use std::time::Instant;

    #[test]
    fn ready_reader_accepts_only_valid_local_handshake() {
        let state = Supervisor::default();
        let valid = br#"{"type":"sidecar.ready","host":"127.0.0.1","port":39100,"token":"abc","runtime_session_id":"session","workspace":null}"#;
        let result = wait_for_sidecar_ready(Cursor::new(valid), &state, Duration::from_secs(1));
        assert_eq!(result.unwrap().port, 39100);
        let wrong_host = br#"{"type":"sidecar.ready","host":"0.0.0.0","port":39100,"token":"abc","runtime_session_id":"session","workspace":null}"#;
        assert!(wait_for_sidecar_ready(Cursor::new(wrong_host), &state, Duration::from_secs(1)).is_none());
    }

    #[test]
    fn stalled_handshake_times_out_instead_of_blocking_restart() {
        struct StalledReader;
        impl std::io::Read for StalledReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                std::thread::sleep(Duration::from_millis(250));
                Ok(0)
            }
        }
        let state = Supervisor::default();
        let before = Instant::now();
        assert!(wait_for_sidecar_ready(StalledReader, &state, Duration::from_millis(25)).is_none());
        assert!(before.elapsed() < Duration::from_millis(200));
    }

    #[test]
    fn package_prefers_local_frozen_sidecar_over_developer_python() {
        let temp=std::env::temp_dir().join(format!(
            "handd-sidecar-resources-test-{}",std::process::id()));
        let bin=temp.join("sidecar/handd-sidecar").join(
            if cfg!(windows) {"handd-sidecar.exe"} else {"handd-sidecar"});
        std::fs::create_dir_all(bin.parent().unwrap()).unwrap();
        std::fs::write(&bin,b"test marker").unwrap();
        let (command,frozen)=sidecar_program(Some(&temp),true).unwrap();
        assert!(frozen);
        assert_eq!(command.get_program(),bin.as_os_str());
        let (development,development_frozen)=sidecar_program(Some(&temp),false).unwrap();
        assert!(!development_frozen);
        assert!(development.get_program().to_string_lossy().contains(".venv"));
        assert!(sidecar_program(None,true).is_err());
        std::fs::remove_dir_all(temp).unwrap();
    }
}
