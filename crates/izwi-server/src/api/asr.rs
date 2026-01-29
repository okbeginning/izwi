//! Qwen3-ASR API endpoints for speech-to-text transcription

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::{info, warn};

use crate::error::ApiError;
use crate::state::AppState;

const ASR_SOCKET_PATH: &str = "/tmp/izwi_qwen3_asr_daemon.sock";

/// ASR transcription request
#[derive(Debug, Deserialize)]
pub struct TranscribeRequest {
    pub audio_base64: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
}

/// ASR transcription response
#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub transcription: String,
    pub language: Option<String>,
}

/// ASR daemon status response
#[derive(Debug, Serialize)]
pub struct AsrStatusResponse {
    pub running: bool,
    pub status: String,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
}

/// Send a message to the ASR daemon via Unix socket
fn send_daemon_message(message: &serde_json::Value) -> Result<serde_json::Value, ApiError> {
    let mut stream = UnixStream::connect(ASR_SOCKET_PATH)
        .map_err(|e| ApiError::internal(format!("ASR daemon not running: {}", e)))?;

    stream.set_read_timeout(Some(Duration::from_secs(120))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    let msg_bytes = serde_json::to_vec(message)
        .map_err(|e| ApiError::internal(format!("Failed to serialize message: {}", e)))?;

    let length = (msg_bytes.len() as u32).to_be_bytes();
    stream
        .write_all(&length)
        .map_err(|e| ApiError::internal(format!("Failed to write length: {}", e)))?;
    stream
        .write_all(&msg_bytes)
        .map_err(|e| ApiError::internal(format!("Failed to write message: {}", e)))?;

    let mut length_buf = [0u8; 4];
    stream
        .read_exact(&mut length_buf)
        .map_err(|e| ApiError::internal(format!("Failed to read response length: {}", e)))?;
    let response_length = u32::from_be_bytes(length_buf) as usize;

    let mut response_buf = vec![0u8; response_length];
    stream
        .read_exact(&mut response_buf)
        .map_err(|e| ApiError::internal(format!("Failed to read response: {}", e)))?;

    serde_json::from_slice(&response_buf)
        .map_err(|e| ApiError::internal(format!("Failed to parse response: {}", e)))
}

/// Check if the ASR daemon is running
fn is_daemon_running() -> bool {
    Path::new(ASR_SOCKET_PATH).exists() && UnixStream::connect(ASR_SOCKET_PATH).is_ok()
}

/// Get ASR daemon status
pub async fn status(State(_state): State<AppState>) -> Result<Json<AsrStatusResponse>, ApiError> {
    if !is_daemon_running() {
        return Ok(Json(AsrStatusResponse {
            running: false,
            status: "stopped".to_string(),
            device: None,
            cached_models: vec![],
        }));
    }

    let message = serde_json::json!({
        "command": "status"
    });

    match send_daemon_message(&message) {
        Ok(response) => {
            let device = response
                .get("device")
                .and_then(|v| v.as_str())
                .map(String::from);
            let cached_models = response
                .get("cached_models")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            Ok(Json(AsrStatusResponse {
                running: true,
                status: "running".to_string(),
                device,
                cached_models,
            }))
        }
        Err(_) => Ok(Json(AsrStatusResponse {
            running: false,
            status: "error".to_string(),
            device: None,
            cached_models: vec![],
        })),
    }
}

/// Start the ASR daemon
pub async fn start_daemon(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if is_daemon_running() {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon already running"
        })));
    }

    info!("Starting Qwen3-ASR daemon");

    let scripts_dir = std::env::current_dir().unwrap_or_default().join("scripts");
    let daemon_script = scripts_dir.join("qwen3_asr_daemon.py");

    if !daemon_script.exists() {
        return Err(ApiError::internal(format!(
            "ASR daemon script not found: {:?}",
            daemon_script
        )));
    }

    let _child = Command::new("python3")
        .arg(&daemon_script)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| ApiError::internal(format!("Failed to start ASR daemon: {}", e)))?;

    // Wait for daemon to start
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(100));
        if is_daemon_running() {
            return Ok(Json(serde_json::json!({
                "success": true,
                "message": "ASR daemon started"
            })));
        }
    }

    Err(ApiError::internal(
        "ASR daemon failed to start within timeout",
    ))
}

/// Stop the ASR daemon
pub async fn stop_daemon(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !is_daemon_running() {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon not running"
        })));
    }

    let message = serde_json::json!({
        "command": "shutdown"
    });

    match send_daemon_message(&message) {
        Ok(_) => Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon stopped"
        }))),
        Err(_) => {
            warn!("Error stopping ASR daemon");
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "ASR daemon stop requested"
            })))
        }
    }
}

/// Transcribe audio to text
pub async fn transcribe(
    State(_state): State<AppState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResponse>, ApiError> {
    if !is_daemon_running() {
        return Err(ApiError::internal(
            "ASR daemon not running. Please start it first.",
        ));
    }

    let message = serde_json::json!({
        "command": "transcribe",
        "audio_base64": request.audio_base64,
        "model_id": request.model_id,
        "language": request.language,
    });

    let response = send_daemon_message(&message)?;

    if let Some(error) = response.get("error").and_then(|v| v.as_str()) {
        return Err(ApiError::internal(error.to_string()));
    }

    let transcription = response
        .get("transcription")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let language = response
        .get("language")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(Json(TranscribeResponse {
        transcription,
        language,
    }))
}
