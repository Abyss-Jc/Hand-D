//! HD-09 native supervisor. Development uses Python from the repo venv;
//! distributing a packaged Python sidecar is a separate release gate.
use serde::{Deserialize, Serialize};
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

fn spawn_supervisor(state: Supervisor) {
    thread::spawn(move || {
        let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("Hand-D repository root");
        while !state.shutdown.load(Ordering::SeqCst) {
            *state.ready.lock().expect("state lock") = None;
            // Kill the Python process directly, not a uv wrapper which could
            // leave its child orphaned when the desktop exits.
            let mut command = Command::new(root.join(".venv/bin/python"));
            command.args([
                    "-u", "-m", "handd_core.sidecar_main",
                    "--legacy-checkpoint", "models/gesture_mlp.pth",
                ])
                .current_dir(&root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
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
        .setup(move |_app| {
            spawn_supervisor(supervisor.clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar_status, restart_sidecar, select_workspace
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
