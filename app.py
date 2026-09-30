import os
import time
import tempfile
import gradio as gr

# PyAV compatibility fix: PyAV 14.0+ dropped 'metadata_errors' from av.open, which faster-whisper uses
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception:
    pass

from faster_whisper import WhisperModel


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# Default execution target: "cpu" (Safe for all systems out-of-the-box).
# To run on NVIDIA GPU:
#   1. Set DEFAULT_DEVICE = "cuda"
#   2. Set DEFAULT_COMPUTE_TYPE = "float16" (or "int8_float16" / "int8")
#   3. Ensure CUDA toolkit and cuDNN drivers are installed.
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"  # "int8" is optimized for CPU efficiency

# Model cache to prevent reloading the model from disk/RAM when settings haven't changed.
MODEL_CACHE = {}

# Supported languages for manual selection (None = Auto-detect)
LANGUAGE_OPTIONS = {
    "Auto-Detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar",
    "Tamil": "ta"
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """
    Loads and caches the Faster-Whisper model.
    Re-uses the cached model if the parameters remain identical.
    """
    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key not in MODEL_CACHE:
        print(f"Loading Whisper model '{model_size}' on {device.upper()} ({compute_type})...")
        MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print(f"Model '{model_size}' loaded successfully!")
    return MODEL_CACHE[cache_key]


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS,mmm format for SRT subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments) -> str:
    """Convert transcribed segments into SRT subtitle format string."""
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        srt_content.append(f"{i}\n{start_str} --> {end_str}\n{segment.text.strip()}\n")
    return "\n".join(srt_content)


def generate_txt(segments) -> str:
    """Convert transcribed segments into clean, continuous text."""
    return " ".join(segment.text.strip() for segment in segments)


# ==============================================================================
# MAIN TRANSCRIPTION LOGIC
# ==============================================================================
def transcribe_audio(
    audio_path: str,
    model_size: str,
    language_display: str,
    vad_filter: bool,
    beam_size: int,
    device_choice: str
):
    """
    Processes audio input using Faster-Whisper and returns formatted outputs.
    """
    if not audio_path:
        return (
            "⚠️ Please upload an audio file or record sound from your microphone.",
            "",
            "",
            None,
            None
        )

    # Determine execution device & quantization
    device = device_choice.lower()
    compute_type = "float16" if device == "cuda" else "int8"
    language_code = LANGUAGE_OPTIONS.get(language_display, None)

    try:
        start_time = time.time()

        # Load or retrieve cached model
        model = get_whisper_model(model_size, device, compute_type)

        # Execute transcription
        # vad_filter=True automatically crops leading/trailing silences using Silero VAD
        segments_generator, info = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=beam_size,
            vad_filter=vad_filter
        )

        # Evaluate generator into list
        segments = list(segments_generator)
        elapsed_time = time.time() - start_time

        # Generate outputs
        full_text = generate_txt(segments)
        srt_text = generate_srt(segments)

        # Build segment breakdown with timestamps
        timestamp_lines = []
        for s in segments:
            timestamp_lines.append(
                f"[{format_timestamp(s.start)} --> {format_timestamp(s.end)}]  {s.text.strip()}"
            )
        timestamp_breakdown = "\n".join(timestamp_lines)

        # Build execution summary
        status_msg = (
            f"✅ **Transcription Complete!**\n"
            f"- **Detected Language:** {info.language.upper()} (Probability: {info.language_probability:.2%})\n"
            f"- **Audio Duration:** {info.duration:.2f} seconds\n"
            f"- **Processing Time:** {elapsed_time:.2f} seconds\n"
            f"- **Hardware Used:** {device.upper()} ({compute_type})\n"
            f"- **Speed Ratio:** {info.duration / elapsed_time:.2f}x real-time"
        )

        # Create temporary downloadable files
        temp_dir = tempfile.gettempdir()
        txt_file_path = os.path.join(temp_dir, "transcript.txt")
        srt_file_path = os.path.join(temp_dir, "transcript.srt")

        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write(srt_text)

        return full_text, timestamp_breakdown, status_msg, txt_file_path, srt_file_path

    except Exception as e:
        error_msg = f"❌ **Error during transcription:** {str(e)}"
        print(error_msg)
        return error_msg, "", "", None, None


# ==============================================================================
# GRADIO WEB INTERFACE
# ==============================================================================
def build_app():
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(title="Faster-Whisper ASR Studio") as app:
        gr.Markdown(
            """
            # 🎙️ Faster-Whisper ASR Studio
            *High-Performance Local Automatic Speech Recognition powered by **Faster-Whisper** and **CTranslate2**.*
            """
        )

        with gr.Row():
            # LEFT COLUMN: Input & Parameters
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Audio Input")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload Audio or Record Voice"
                )

                with gr.Accordion("⚙️ Model & Performance Settings", open=True):
                    model_dropdown = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="base",
                        label="Whisper Model Size",
                        info="Larger models increase accuracy but require more RAM/Compute."
                    )

                    device_radio = gr.Radio(
                        choices=["cpu", "cuda"],
                        value=DEFAULT_DEVICE,
                        label="Compute Device",
                        info="Default is CPU. Select CUDA if an NVIDIA GPU with drivers is configured."
                    )

                    language_dropdown = gr.Dropdown(
                        choices=list(LANGUAGE_OPTIONS.keys()),
                        value="Auto-Detect",
                        label="Language",
                        info="Select specific language or allow automatic detection."
                    )

                    vad_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable VAD (Voice Activity Detection)",
                        info="Filters out silence and background noise before transcription."
                    )

                    beam_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Beam Size",
                        info="Higher values improve quality but slow down processing."
                    )

                transcribe_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")

            # RIGHT COLUMN: Results & Export
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Results")
                status_box = gr.Markdown(value="*Upload audio and click 'Start Transcription' to begin.*")

                with gr.Tabs():
                    with gr.Tab("Full Text"):
                        output_text = gr.Textbox(
                            label="Transcribed Text",
                            placeholder="Transcription results will appear here...",
                            lines=10
                        )
                    with gr.Tab("Timestamped Segments"):
                        timestamp_text = gr.Textbox(
                            label="Segments with Timestamps",
                            placeholder="Timestamped segments will appear here...",
                            lines=10
                        )

                gr.Markdown("### 💾 Export Files")
                with gr.Row():
                    txt_download = gr.File(label="Download .TXT File")
                    srt_download = gr.File(label="Download .SRT File")

        # Event trigger
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                model_dropdown,
                language_dropdown,
                vad_checkbox,
                beam_slider,
                device_radio
            ],
            outputs=[
                output_text,
                timestamp_text,
                status_box,
                txt_download,
                srt_download
            ]
        )

        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666;'>
                Built with <strong>Faster-Whisper</strong> & <strong>Gradio</strong> | Local & Privacy-focused ASR
            </div>
            """
        )

    return app, theme


if __name__ == "__main__":
    # Launch Gradio interface locally
    app, theme = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)

