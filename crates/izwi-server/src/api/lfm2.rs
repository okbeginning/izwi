//! LFM2-Audio API endpoints
//!
//! Handles TTS, ASR, and audio-to-audio chat via the LFM2 daemon.
//! Uses the InferenceEngine's LFM2Bridge for daemon lifecycle management.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::state::AppState;

/// LFM2 TTS request
#[derive(Debug, Deserialize)]
pub struct LFM2TTSRequest {
    pub text: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub audio_temperature: Option<f32>,
    #[serde(default)]
    pub audio_top_k: Option<u32>,
}

fn default_voice() -> String {
    "us_female".to_string()
}

/// LFM2 ASR request
#[derive(Debug, Deserialize)]
pub struct LFM2ASRRequest {
    pub audio_base64: String,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
}

/// LFM2 Audio Chat request
#[derive(Debug, Deserialize)]
pub struct LFM2AudioChatRequest {
    #[serde(default)]
    pub audio_base64: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub audio_temperature: Option<f32>,
    #[serde(default)]
    pub audio_top_k: Option<u32>,
}

/// LFM2 TTS response
#[derive(Debug, Serialize)]
pub struct LFM2TTSResponse {
    pub audio_base64: String,
    pub sample_rate: u32,
    pub format: String,
}

/// LFM2 ASR response
#[derive(Debug, Serialize)]
pub struct LFM2ASRResponse {
    pub transcription: String,
}

/// LFM2 Audio Chat response
#[derive(Debug, Serialize)]
pub struct LFM2AudioChatResponse {
    pub text: String,
    pub audio_base64: Option<String>,
    pub sample_rate: u32,
    pub format: String,
}

/// LFM2 Status response
#[derive(Debug, Serialize)]
pub struct LFM2StatusResponse {
    pub running: bool,
    pub status: String,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
    pub voices: Vec<String>,
}

/// LFM2 Daemon response
#[derive(Debug, Serialize)]
pub struct LFM2DaemonResponse {
    pub success: bool,
    pub message: String,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
}

/// Get LFM2 daemon status
pub async fn status(
    State(state): State<AppState>,
) -> Result<Json<LFM2StatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let engine = state.engine.read().await;

    match engine.get_lfm2_daemon_status() {
        Ok(response) => {
            if let Some(error) = &response.error {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: error.clone(),
                        },
                    }),
                ));
            }

            Ok(Json(LFM2StatusResponse {
                running: response.status.as_deref() == Some("ok"),
                status: response.status.unwrap_or_else(|| "unknown".to_string()),
                device: response.device,
                cached_models: response.cached_models.unwrap_or_default(),
                voices: response.voices.unwrap_or_else(|| {
                    vec![
                        "us_male".to_string(),
                        "us_female".to_string(),
                        "uk_male".to_string(),
                        "uk_female".to_string(),
                    ]
                }),
            }))
        }
        Err(e) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("LFM2 daemon not available: {}", e),
                },
            }),
        )),
    }
}

/// Start the LFM2 daemon
pub async fn start_daemon(
    State(state): State<AppState>,
) -> Result<Json<LFM2DaemonResponse>, (StatusCode, Json<LFM2DaemonResponse>)> {
    info!("Starting LFM2 daemon via API");

    let engine = state.engine.read().await;

    match engine.ensure_lfm2_daemon_running() {
        Ok(_) => Ok(Json(LFM2DaemonResponse {
            success: true,
            message: "LFM2 daemon started successfully".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(LFM2DaemonResponse {
                success: false,
                message: format!("Failed to start LFM2 daemon: {}", e),
            }),
        )),
    }
}

/// Stop the LFM2 daemon
pub async fn stop_daemon(
    State(state): State<AppState>,
) -> Result<Json<LFM2DaemonResponse>, (StatusCode, Json<LFM2DaemonResponse>)> {
    info!("Stopping LFM2 daemon via API");

    let engine = state.engine.read().await;

    match engine.stop_lfm2_daemon() {
        Ok(_) => Ok(Json(LFM2DaemonResponse {
            success: true,
            message: "LFM2 daemon stopped successfully".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(LFM2DaemonResponse {
                success: false,
                message: format!("Failed to stop LFM2 daemon: {}", e),
            }),
        )),
    }
}

/// Generate TTS with LFM2
pub async fn tts(
    State(state): State<AppState>,
    Json(req): Json<LFM2TTSRequest>,
) -> Result<Json<LFM2TTSResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!(
        "LFM2 TTS request: {} chars, voice: {}",
        req.text.len(),
        req.voice
    );

    let engine = state.engine.read().await;

    match engine.lfm2_generate_tts(
        &req.text,
        Some(&req.voice),
        req.max_new_tokens,
        req.audio_temperature,
        req.audio_top_k,
    ) {
        Ok(response) => Ok(Json(LFM2TTSResponse {
            audio_base64: response.audio_base64.unwrap_or_default(),
            sample_rate: response.sample_rate.unwrap_or(24000),
            format: response.format.unwrap_or_else(|| "wav".to_string()),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("{}", e),
                },
            }),
        )),
    }
}

/// Transcribe audio with LFM2 ASR
pub async fn asr(
    State(state): State<AppState>,
    Json(req): Json<LFM2ASRRequest>,
) -> Result<Json<LFM2ASRResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("LFM2 ASR request");

    let engine = state.engine.read().await;

    match engine.lfm2_transcribe(&req.audio_base64, req.max_new_tokens) {
        Ok(response) => Ok(Json(LFM2ASRResponse {
            transcription: response.transcription.unwrap_or_default(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("{}", e),
                },
            }),
        )),
    }
}

/// Audio-to-audio chat with LFM2
pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<LFM2AudioChatRequest>,
) -> Result<Json<LFM2AudioChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    info!("LFM2 Audio Chat request");

    let engine = state.engine.read().await;

    match engine.lfm2_audio_chat(
        req.audio_base64.as_deref(),
        req.text.as_deref(),
        req.max_new_tokens,
        req.audio_temperature,
        req.audio_top_k,
    ) {
        Ok(response) => Ok(Json(LFM2AudioChatResponse {
            text: response.text.unwrap_or_default(),
            audio_base64: response.audio_base64,
            sample_rate: response.sample_rate.unwrap_or(24000),
            format: response.format.unwrap_or_else(|| "wav".to_string()),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("{}", e),
                },
            }),
        )),
    }
}
