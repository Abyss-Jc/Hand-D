//! HD-09 native supervisor and explicit workspace/document commands.
mod native_files;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    restart: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    active_child: Arc<Mutex<Option<Child>>>,
    workspace: Arc<Mutex<Option<PathBuf>>>,
    portable_root: Arc<Mutex<Option<PathBuf>>>,
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
fn sidecar_status(state: tauri::State<'_, Supervisor>) -> Option<SidecarReady> {
    state.ready.lock().expect("state lock").clone()
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
        while !state.shutdown.load(Ordering::SeqCst) {
            *state.ready.lock().expect("state lock") = None;
            // Kill the Python process directly, not a uv wrapper which could
            // leave its child orphaned when the desktop exits.
            let portable=state.portable_root.lock().expect("resource lock").clone();
            let (mut command, frozen)=match sidecar_program(
                portable.as_ref(),!cfg!(debug_assertions)) {
                Ok(launcher)=>launcher,
                Err(error)=>{
                    eprintln!("Hand-D packaged sidecar unavailable: {error}");
                    thread::sleep(Duration::from_secs(2));
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
                    if let Some(stdout) = stdout {
                        let mut reader = BufReader::new(stdout);
                        let mut line = String::new();
                        match reader.read_line(&mut line) {
                            Ok(n) if n > 0 => {
                                if let Ok(ready) = serde_json::from_str::<SidecarReady>(&line) {
                                    if ready.kind == "sidecar.ready"
                                        && ready.host == "127.0.0.1"
                                        && ready.port > 0
                                    {
                                        *state.ready.lock().expect("state lock") = Some(ready);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    while !state.shutdown.load(Ordering::SeqCst) {
                        if state.restart.swap(false, Ordering::SeqCst) {
                            break;
                        }
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
                }
                Err(error) => {
                    eprintln!("Hand-D sidecar could not start: {error}");
                }
            }
            for _ in 0..8 {
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
