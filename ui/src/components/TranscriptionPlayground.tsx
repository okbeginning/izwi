import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Mic,
  MicOff,
  Upload,
  Download,
  RotateCcw,
  Loader2,
  Copy,
  Check,
} from "lucide-react";
import { api } from "../api";
import clsx from "clsx";

interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  onModelRequired: () => void;
}

export function TranscriptionPlayground({
  selectedModel,
  onModelRequired,
}: TranscriptionPlaygroundProps) {
  const [transcription, setTranscription] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startRecording = useCallback(async () => {
    if (!selectedModel) {
      onModelRequired();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [selectedModel, onModelRequired]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError(null);

    const url = URL.createObjectURL(audioBlob);
    setAudioUrl(url);

    try {
      const reader = new FileReader();
      const audioBase64 = await new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(",")[1];
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
      });

      const response = await api.asrTranscribe({
        audio_base64: audioBase64,
        model_id: selectedModel || undefined,
      });

      setTranscription(response.transcription);
      setDetectedLanguage(response.language || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcription failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!selectedModel) {
      onModelRequired();
      return;
    }

    await processAudio(file);
  };

  const handleReset = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setTranscription("");
    setDetectedLanguage(null);
    setAudioUrl(null);
    setError(null);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([transcription], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `transcription-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/20">
            <FileText className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-white">Transcription</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Convert speech to text with Qwen3-ASR
            </p>
          </div>
        </div>

        {(transcription || audioUrl) && (
          <button onClick={handleReset} className="btn btn-ghost text-xs">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        )}
      </div>

      {/* Audio Preview */}
      {audioUrl && (
        <div className="mb-4 p-3 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a]">
          <div className="text-xs text-gray-500 mb-2">Audio Input</div>
          <audio src={audioUrl} controls className="w-full h-8" />
        </div>
      )}

      {/* Transcription Result */}
      {transcription && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Transcription</span>
              {detectedLanguage && (
                <span className="text-[10px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded">
                  {detectedLanguage}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={handleCopy}
                className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-green-500" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
              <button
                onClick={handleDownload}
                className="p-1.5 rounded hover:bg-white/5 text-gray-500 hover:text-gray-300"
              >
                <Download className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
          <div className="p-3 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a]">
            <p className="text-sm text-gray-300 whitespace-pre-wrap">
              {transcription}
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="p-2 rounded bg-red-950/50 border border-red-900/50 text-red-400 text-xs mb-4"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Processing indicator */}
      {isProcessing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center gap-2 text-gray-400 text-sm py-8"
        >
          <Loader2 className="w-5 h-5 animate-spin" />
          Transcribing...
        </motion.div>
      )}

      {/* Input Controls */}
      {!isProcessing && (
        <div className="space-y-4">
          {/* Record / Upload buttons */}
          <div className="flex items-center justify-center gap-6">
            {/* Record button */}
            <div className="flex flex-col items-center gap-2">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={clsx(
                  "p-4 rounded-full transition-all",
                  isRecording
                    ? "bg-red-500 hover:bg-red-600 animate-pulse"
                    : "bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2a2a2a]",
                )}
              >
                {isRecording ? (
                  <MicOff className="w-6 h-6 text-white" />
                ) : (
                  <Mic className="w-6 h-6 text-gray-300" />
                )}
              </button>
              <span className="text-xs text-gray-500">
                {isRecording ? "Stop" : "Record"}
              </span>
            </div>

            <div className="text-gray-600 text-xs">or</div>

            {/* Upload button */}
            <div className="flex flex-col items-center gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-4 rounded-full bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#2a2a2a] transition-all"
              >
                <Upload className="w-6 h-6 text-gray-300" />
              </button>
              <span className="text-xs text-gray-500">Upload</span>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>

          {isRecording && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center justify-center gap-2 text-red-400 text-sm"
            >
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Recording...
            </motion.div>
          )}

          <p className="text-center text-xs text-gray-500">
            {isRecording
              ? "Click the microphone to stop recording"
              : "Record audio or upload a file for transcription"}
          </p>
        </div>
      )}
    </div>
  );
}
