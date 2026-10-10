//! Native, opt-in OS file/folder dialogs without changing Tauri's capabilities.
//! No webview-supplied raw path is used for an initial Save/Open/Export.
//! Linux relies on zenity (with kdialog fallback), macOS uses osascript.
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
pub struct DrawingOpen {
    pub path: String,
    pub document: String,
}

#[cfg(any(test, target_os = "macos"))]
fn macos_picker_result(ok: bool, stdout: &[u8], stderr: &[u8]) -> Result<Option<PathBuf>, String> {
    if !ok {
        let message = String::from_utf8_lossy(stderr);
        if message.contains("(-128)") { return Ok(None); }
        return Err(format!("macOS file picker failed: {}",
            message.trim().chars().take(180).collect::<String>()));
    }
    let value = String::from_utf8(stdout.to_vec())
        .map_err(|_| "Invalid native file selection")?;
    Ok(value.lines().next().filter(|s| !s.is_empty()).map(PathBuf::from))
}

fn picker(kind: &str, title: &str, filename: Option<&str>) -> Result<Option<PathBuf>, String> {
    #[cfg(target_os = "linux")]
    {
        let mut command=Command::new("zenity");
        command.arg("--file-selection").arg(format!("--title={title}"));
        match kind {
            "directory" => {command.arg("--directory");}
            "open" => {command.arg("--file-filter=Hand-D drawing | *.handd.json");}
            "save" | "export" => {
                command.arg("--save").arg("--confirm-overwrite");
                if let Some(name)=filename {command.arg(format!("--filename={name}"));}
            }
            _ => return Err("Unsupported file dialog".into()),
        }
        let output=match command.output() {
            Ok(output)=>output,
            Err(_) => {
                let mut alt=Command::new("kdialog");
                match kind {
                    "directory" => {alt.arg("--getexistingdirectory");}
                    "open" => {alt.arg("--getopenfilename").arg(".")
                        .arg("Hand-D drawings (*.handd.json)");}
                    _ => {alt.arg("--getsavefilename").arg(filename.unwrap_or("."))
                        .arg(if kind=="export" {"SVG files (*.svg)"}
                             else {"Hand-D drawings (*.handd.json)"});}
                }
                alt.arg("--title").arg(title);
                alt.output().map_err(|e|format!("No native file picker available: {e}"))?
            }
        };
        if !output.status.success() {return Ok(None);}
        let value=String::from_utf8(output.stdout)
            .map_err(|_|"Invalid native file selection")?;
        return Ok(value.lines().next().filter(|s|!s.is_empty()).map(PathBuf::from));
    }
    #[cfg(target_os="macos")]
    {
        let script=match kind {
            "directory" => "POSIX path of (choose folder with prompt \"Select Hand-D workspace\")",
            "open" => "POSIX path of (choose file with prompt \"Open Hand-D drawing\")",
            "save" => "POSIX path of (choose file name with prompt \"Save Hand-D drawing\" default name \"drawing.handd.json\")",
            "export" => "POSIX path of (choose file name with prompt \"Export clean drawing\" default name \"drawing.svg\")",
            _ => return Err("Unsupported file dialog".into()),
        };
        let output=Command::new("osascript").args(["-e",script]).output()
            .map_err(|e|format!("macOS file picker unavailable: {e}"))?;
        return macos_picker_result(
            output.status.success(), &output.stdout, &output.stderr);
    }
    #[cfg(target_os="windows")]
    {
        // Native Shell dialog via PowerShell; return only a chosen path.
        let script=match kind {
            "directory" => {
                "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}"
            }
            "open" => {
                "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Filter='Hand-D drawings|*.handd.json'; if($d.ShowDialog() -eq 'OK'){$d.FileName}"
            }
            "save" | "export" => {
                "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.SaveFileDialog; if($d.ShowDialog() -eq 'OK'){$d.FileName}"
            }
            _ => return Err("Unsupported file dialog".into()),
        };
        let output=Command::new("powershell").args(["-NoProfile","-Command",script])
            .output().map_err(|e|format!("Windows file picker unavailable: {e}"))?;
        if !output.status.success(){return Ok(None);}
        let value=String::from_utf8(output.stdout)
            .map_err(|_|"Invalid native file selection")?;
        return Ok(value.lines().next().filter(|s|!s.is_empty()).map(PathBuf::from));
    }
}

pub fn pick_folder() -> Result<Option<PathBuf>, String> {
    picker("directory","Select Hand-D Project Workspace",None)
}

fn enforce_extension(path: PathBuf, suffix: &str) -> PathBuf {
    if path.to_string_lossy().to_ascii_lowercase().ends_with(suffix) {
        path
    } else {
        PathBuf::from(format!("{}{}",path.display(),suffix))
    }
}

fn atomic_write(path: &Path, text: &str) -> Result<(), String> {
    if text.len()>8_000_000 {return Err("Drawing is too large".into());}
    let nanos=SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|e|e.to_string())?.as_nanos();
    let tmp=path.with_extension(format!("handd-{}-{nanos}.tmp",std::process::id()));
    fs::write(&tmp,text).map_err(|e|format!("Could not write drawing: {e}"))?;
    if let Err(e)=fs::rename(&tmp,path) {
        let _=fs::remove_file(&tmp);
        return Err(format!("Could not save drawing: {e}"));
    }
    Ok(())
}

fn is_valid_native_drawing(value: &str) -> bool {
    if value.len()>8_000_000 {return false;}
    let Ok(doc)=serde_json::from_str::<serde_json::Value>(value) else {return false};
    doc.get("format").and_then(|s|s.as_str())==Some("handd-whiteboard")
        && doc.get("version").and_then(|v|v.as_u64())==Some(1)
        && doc.get("strokes").and_then(|v|v.as_array()).is_some()
}

pub fn save_drawing(document: String) -> Result<Option<String>, String> {
    if !is_valid_native_drawing(&document) {return Err("Invalid editable drawing".into());}
    let Some(path)=picker("save","Save editable Hand-D drawing",Some("drawing.handd.json"))?
    else {return Ok(None)};
    let path=enforce_extension(path,".handd.json");
    atomic_write(&path,&document)?;
    Ok(Some(path.to_string_lossy().to_string()))
}

pub fn open_drawing() -> Result<Option<DrawingOpen>, String> {
    let Some(path)=picker("open","Open editable Hand-D drawing",None)?
    else {return Ok(None)};
    let metadata=fs::metadata(&path).map_err(|e|e.to_string())?;
    if metadata.len()>8_000_000 {return Err("Drawing file is too large".into());}
    let document=fs::read_to_string(&path).map_err(|e|format!("Cannot open drawing: {e}"))?;
    if !is_valid_native_drawing(&document) {return Err("Invalid Hand-D drawing file".into());}
    Ok(Some(DrawingOpen{path:path.to_string_lossy().to_string(),document}))
}

pub fn export_svg(svg: String) -> Result<Option<String>, String> {
    if svg.len()>8_000_000 || !svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\"")
      || !svg.ends_with("</svg>\n") || svg.contains("<image") || svg.contains("<script") {
        return Err("Invalid clean SVG drawing".into());
    }
    let Some(path)=picker("export","Export clean SVG artwork",Some("drawing.svg"))?
    else {return Ok(None)};
    let path=enforce_extension(path,".svg");
    atomic_write(&path,&svg)?;
    Ok(Some(path.to_string_lossy().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn macos_dialog_distinguishes_user_cancel_from_real_error() {
        let canceled = macos_picker_result(false, b"", b"execution error: User canceled. (-128)");
        assert_eq!(canceled.unwrap(), None);
        let broken = macos_picker_result(false, b"", b"Not authorized to send Apple events. (-1743)");
        assert!(broken.unwrap_err().contains("Not authorized"));
        let chosen = macos_picker_result(true, b"/Users/demo/drawing.handd.json\n", b"");
        assert_eq!(chosen.unwrap(), Some(PathBuf::from("/Users/demo/drawing.handd.json")));
    }
    #[test]
    fn native_drawing_rejects_unstructured_or_oversized_payloads(){
        assert!(is_valid_native_drawing(r#"{"format":"handd-whiteboard","version":1,"strokes":[]}"#));
        assert!(!is_valid_native_drawing("{}"));
        assert!(!is_valid_native_drawing(r#"{"format":"handd-whiteboard","version":2,"strokes":[]}"#));
        assert!(!is_valid_native_drawing(&"x".repeat(8_000_001)));
    }
    #[test]
    fn extensions_are_added_only_when_missing(){
        assert_eq!(enforce_extension(PathBuf::from("/tmp/a"),".svg"),PathBuf::from("/tmp/a.svg"));
        assert_eq!(enforce_extension(PathBuf::from("/tmp/a.svg"),".svg"),PathBuf::from("/tmp/a.svg"));
    }
    #[test]
    fn atomic_save_keeps_previous_bytes_when_input_exceeds_limit(){
        let dir=std::env::temp_dir().join(format!("handd-test-{}",std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path=dir.join("document.handd.json");
        fs::write(&path,"previous").unwrap();
        assert!(atomic_write(&path,&"X".repeat(8_000_001)).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(),"previous");
        let _=fs::remove_dir_all(&dir);
    }
}
