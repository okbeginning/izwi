//! LFM2-Audio bridge for inference
//! Connects to a persistent Python daemon for LFM2-Audio model inference

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, info};

use crate::error::{Error, Result};

/// Default socket path for the LFM2 daemon
const DEFAULT_SOCKET_PATH: &str = "/tmp/izwi_lfm2_daemon.sock";

/// Request to LFM2 daemon
#[derive(Debug, Serialize)]
pub struct LFM2Request {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_top_k: Option<u32>,
}

impl Default for LFM2Request {
    fn default() -> Self {
        Self {
            command: String::new(),
            text: None,
            voice: None,
            audio_base64: None,
            max_new_tokens: None,
            audio_temperature: None,
            audio_top_k: None,
        }
    }
}

/// Response from LFM2 daemon
#[derive(Debug, Deserialize, Clone)]
pub struct LFM2Response {
    pub audio_base64: Option<String>,
    pub sample_rate: Option<u32>,
    pub format: Option<String>,
    pub text: Option<String>,
    pub transcription: Option<String>,
    pub error: Option<String>,
    pub status: Option<String>,
    pub device: Option<String>,
    pub cached_models: Option<Vec<String>>,
    pub voices: Option<Vec<String>>,
}

/// LFM2-Audio bridge for calling the LFM2 daemon
pub struct LFM2Bridge {
    socket_path: PathBuf,
    daemon_script_path: PathBuf,
    python_cmd: String,
    daemon_process: Mutex<Option<Child>>,
}

impl LFM2Bridge {
    /// Create a new LFM2 bridge
    pub fn new() -> Self {
        let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            daemon_script_path: base_dir.join("scripts/lfm2_daemon.py"),
            python_cmd: "python3".to_string(),
            daemon_process: Mutex::new(None),
        }
    }

    /// Check if the daemon is running
    fn is_daemon_running(&self) -> bool {
        self.socket_path.exists() && self.connect_to_daemon().is_ok()
    }

    /// Start the daemon if not running
    pub fn ensure_daemon_running(&self) -> Result<()> {
        if self.is_daemon_running() {
            debug!("LFM2 daemon already running");
            return Ok(());
        }

        info!("Starting LFM2 daemon...");

        let child = Command::new(&self.python_cmd)
            .arg(&self.daemon_script_path)
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::InferenceError(format!("Failed to start LFM2 daemon: {}", e)))?;

        // Store the child process
        {
            let mut guard = self.daemon_process.lock().unwrap();
            *guard = Some(child);
        }

        // Wait for daemon to be ready (up to 60 seconds - LFM2 model loading can be slow)
        for i in 0..600 {
            std::thread::sleep(Duration::from_millis(100));
            if self.socket_path.exists() {
                if let Ok(mut stream) = self.connect_to_daemon() {
                    // Send a status command to verify it's responding
                    let request = LFM2Request {
                        command: "status".to_string(),
                        ..Default::default()
                    };
                    if self.send_request(&mut stream, &request).is_ok() {
                        info!("LFM2 daemon started successfully");
                        return Ok(());
                    }
                }
            }
            if i % 100 == 0 && i > 0 {
                debug!("Waiting for LFM2 daemon to start... ({}/60s)", i / 10);
            }
        }

        Err(Error::InferenceError(
            "LFM2 daemon failed to start within 60 seconds".to_string(),
        ))
    }

    /// Stop the daemon
    pub fn stop_daemon(&self) -> Result<()> {
        if !self.is_daemon_running() {
            return Ok(());
        }

        info!("Stopping LFM2 daemon...");

        // Send shutdown command
        if let Ok(mut stream) = self.connect_to_daemon() {
            let request = LFM2Request {
                command: "shutdown".to_string(),
                ..Default::default()
            };
            let _ = self.send_request(&mut stream, &request);
        }

        // Kill the process if we started it
        {
            let mut guard = self.daemon_process.lock().unwrap();
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        // Clean up socket file
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        Ok(())
    }

    /// Connect to the daemon socket
    fn connect_to_daemon(&self) -> Result<UnixStream> {
        let stream = UnixStream::connect(&self.socket_path).map_err(|e| {
            Error::InferenceError(format!("Failed to connect to LFM2 daemon: {}", e))
        })?;

        // Set longer timeouts for LFM2 which can be slow
        stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(60))).ok();
        stream.set_nonblocking(false).ok();

        Ok(stream)
    }

    /// Read exactly n bytes with retry on EAGAIN/WouldBlock
    fn read_exact_with_retry(
        stream: &mut UnixStream,
        buf: &mut [u8],
        max_retries: u32,
    ) -> std::io::Result<()> {
        let mut total_read = 0;
        let mut retries = 0;

        while total_read < buf.len() {
            match stream.read(&mut buf[total_read..]) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Connection closed",
                    ));
                }
                Ok(n) => {
                    total_read += n;
                    retries = 0;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    retries += 1;
                    if retries > max_retries {
                        return Err(e);
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Send request to daemon and receive response
    fn send_request(&self, stream: &mut UnixStream, request: &LFM2Request) -> Result<LFM2Response> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| Error::InferenceError(format!("Failed to serialize request: {}", e)))?;

        // Send length-prefixed message
        let data = request_json.as_bytes();
        let length = (data.len() as u32).to_be_bytes();

        stream
            .write_all(&length)
            .map_err(|e| Error::InferenceError(format!("Failed to write length: {}", e)))?;
        stream
            .write_all(data)
            .map_err(|e| Error::InferenceError(format!("Failed to write request: {}", e)))?;
        stream
            .flush()
            .map_err(|e| Error::InferenceError(format!("Failed to flush: {}", e)))?;

        // Read length-prefixed response
        let mut length_buf = [0u8; 4];
        Self::read_exact_with_retry(stream, &mut length_buf, 3000)
            .map_err(|e| Error::InferenceError(format!("Failed to read response length: {}", e)))?;
        let response_len = u32::from_be_bytes(length_buf) as usize;

        let mut response_buf = vec![0u8; response_len];
        Self::read_exact_with_retry(stream, &mut response_buf, 3000)
            .map_err(|e| Error::InferenceError(format!("Failed to read response body: {}", e)))?;

        let response: LFM2Response = serde_json::from_slice(&response_buf).map_err(|e| {
            Error::InferenceError(format!(
                "Failed to parse response: {} - {}",
                e,
                String::from_utf8_lossy(&response_buf)
            ))
        })?;

        Ok(response)
    }

    /// Call daemon with request
    fn call_daemon(&self, request: &LFM2Request) -> Result<LFM2Response> {
        // Ensure daemon is running
        self.ensure_daemon_running()?;

        // Connect and send request
        let mut stream = self.connect_to_daemon()?;
        self.send_request(&mut stream, request)
    }

    /// Get daemon status
    pub fn get_status(&self) -> Result<LFM2Response> {
        let request = LFM2Request {
            command: "status".to_string(),
            ..Default::default()
        };
        self.call_daemon(&request)
    }

    /// Generate TTS audio
    pub fn generate_tts(
        &self,
        text: &str,
        voice: Option<&str>,
        max_new_tokens: Option<u32>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<u32>,
    ) -> Result<LFM2Response> {
        info!("LFM2 TTS for text: {}", text);

        let request = LFM2Request {
            command: "tts".to_string(),
            text: Some(text.to_string()),
            voice: voice.map(|s| s.to_string()),
            max_new_tokens,
            audio_temperature,
            audio_top_k,
            ..Default::default()
        };

        let response = self.call_daemon(&request)?;

        if let Some(err) = &response.error {
            return Err(Error::InferenceError(format!("LFM2 TTS error: {}", err)));
        }

        Ok(response)
    }

    /// Transcribe audio (ASR)
    pub fn transcribe(
        &self,
        audio_base64: &str,
        max_new_tokens: Option<u32>,
    ) -> Result<LFM2Response> {
        info!("LFM2 ASR transcription");

        let request = LFM2Request {
            command: "asr".to_string(),
            audio_base64: Some(audio_base64.to_string()),
            max_new_tokens,
            ..Default::default()
        };

        let response = self.call_daemon(&request)?;

        if let Some(err) = &response.error {
            return Err(Error::InferenceError(format!("LFM2 ASR error: {}", err)));
        }

        Ok(response)
    }

    /// Audio chat (audio-to-audio)
    pub fn audio_chat(
        &self,
        audio_base64: Option<&str>,
        text: Option<&str>,
        max_new_tokens: Option<u32>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<u32>,
    ) -> Result<LFM2Response> {
        info!("LFM2 Audio Chat");

        let request = LFM2Request {
            command: "audio_chat".to_string(),
            audio_base64: audio_base64.map(|s| s.to_string()),
            text: text.map(|s| s.to_string()),
            max_new_tokens,
            audio_temperature,
            audio_top_k,
            ..Default::default()
        };

        let response = self.call_daemon(&request)?;

        if let Some(err) = &response.error {
            return Err(Error::InferenceError(format!("LFM2 chat error: {}", err)));
        }

        Ok(response)
    }
}

impl Default for LFM2Bridge {
    fn default() -> Self {
        Self::new()
    }
}
