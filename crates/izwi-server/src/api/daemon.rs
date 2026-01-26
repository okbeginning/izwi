//! Daemon management API endpoints

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::state::AppState;

/// Daemon status response
#[derive(Debug, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
}

/// Preload model request
#[derive(Debug, Deserialize)]
pub struct PreloadRequest {
    pub model_path: String,
}

/// Generic response
#[derive(Debug, Serialize)]
pub struct DaemonResponse {
    pub success: bool,
    pub message: String,
}

/// Get daemon status
pub async fn get_status(State(state): State<AppState>) -> Json<DaemonStatus> {
    let engine = state.engine.read().await;

    match engine.get_daemon_status() {
        Ok(response) => Json(DaemonStatus {
            running: response.status.as_deref() == Some("ok"),
            device: response.device,
            cached_models: response.cached_models.unwrap_or_default(),
        }),
        Err(_) => Json(DaemonStatus {
            running: false,
            device: None,
            cached_models: vec![],
        }),
    }
}

/// Start the daemon
pub async fn start_daemon(
    State(state): State<AppState>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Starting TTS daemon via API");

    let engine = state.engine.read().await;

    match engine.ensure_daemon_running() {
        Ok(_) => Ok(Json(DaemonResponse {
            success: true,
            message: "Daemon started successfully".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(DaemonResponse {
                success: false,
                message: format!("Failed to start daemon: {}", e),
            }),
        )),
    }
}

/// Stop the daemon
pub async fn stop_daemon(
    State(state): State<AppState>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Stopping TTS daemon via API");

    let engine = state.engine.read().await;

    match engine.stop_daemon() {
        Ok(_) => Ok(Json(DaemonResponse {
            success: true,
            message: "Daemon stopped successfully".to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(DaemonResponse {
                success: false,
                message: format!("Failed to stop daemon: {}", e),
            }),
        )),
    }
}

/// Preload a model into the daemon cache
pub async fn preload_model(
    State(state): State<AppState>,
    Json(request): Json<PreloadRequest>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Preloading model via API: {}", request.model_path);

    let engine = state.engine.read().await;

    match engine.preload_model(&request.model_path) {
        Ok(_) => Ok(Json(DaemonResponse {
            success: true,
            message: format!("Model preloaded: {}", request.model_path),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(DaemonResponse {
                success: false,
                message: format!("Failed to preload model: {}", e),
            }),
        )),
    }
}
