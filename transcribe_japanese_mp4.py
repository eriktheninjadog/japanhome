#!/usr/bin/env python3
"""
transcribe_japanese_mp4.py

Extract audio from a Japanese MP4 file and produce an English SRT subtitle file
using a local Whisper model downloaded from HuggingFace.

Usage:
    python transcribe_japanese_mp4.py <input.mp4> [options]

Examples:
    python transcribe_japanese_mp4.py video.mp4
    python transcribe_japanese_mp4.py video.mp4 --output subtitles.srt
    python transcribe_japanese_mp4.py video.mp4 --model openai/whisper-medium
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_audio(mp4_path: str, audio_path: str) -> None:
    """Extract audio from an MP4 file and save it as a 16 kHz mono WAV.

    Whisper models expect 16 kHz, single-channel PCM audio.

    Args:
        mp4_path:   Path to the source MP4 file.
        audio_path: Destination path for the extracted WAV file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
        FileNotFoundError: If ffmpeg is not installed / not on PATH.
    """
    cmd = [
        "ffmpeg",
        "-i", mp4_path,
        "-vn",           # drop video stream
        "-acodec", "pcm_s16le",  # 16-bit PCM (standard WAV)
        "-ar", "16000",  # 16 kHz sample rate required by Whisper
        "-ac", "1",      # mono channel
        "-y",            # overwrite output without asking
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )


def format_srt_timestamp(seconds: float) -> str:
    """Convert a duration in seconds to an SRT timestamp string.

    Args:
        seconds: Duration in seconds (may include fractional part).

    Returns:
        Timestamp in ``HH:MM:SS,mmm`` format as required by the SRT spec.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_and_translate(audio_path: str, model_name: str) -> list:
    """Transcribe Japanese audio and translate it to English using Whisper.

    Uses the HuggingFace *transformers* ``automatic-speech-recognition`` pipeline
    with ``task="translate"`` so the model outputs English text regardless of the
    source language.

    Args:
        audio_path: Path to the WAV file to transcribe.
        model_name: HuggingFace model ID, e.g. ``"openai/whisper-large-v3"``.

    Returns:
        A list of chunk dicts with keys ``"text"`` and ``"timestamp"``
        (a ``(start, end)`` tuple of floats in seconds).
    """
    # Import here so the module can be imported without transformers installed
    # (e.g. during unit testing of helper functions).
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required. "
            "Install it with:  pip install transformers"
        ) from exc

    print(f"Loading model '{model_name}' from HuggingFace …")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        stride_length_s=5,
    )

    print("Transcribing and translating audio …")
    result = pipe(
        audio_path,
        return_timestamps=True,
        generate_kwargs={"task": "translate", "language": "japanese"},
    )

    chunks = result.get("chunks", [])
    return chunks


def write_srt(chunks: list, srt_path: str) -> None:
    """Write recognized chunks to an SRT subtitle file.

    Chunks with a missing end timestamp or empty text are silently skipped.

    Args:
        chunks:   List of dicts with ``"text"`` and ``"timestamp"`` keys.
        srt_path: Destination path for the ``.srt`` file.
    """
    index = 1
    with open(srt_path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp if len(timestamp) == 2 else (None, None)

            if start is None or end is None:
                continue

            text = chunk.get("text", "").strip()
            if not text:
                continue

            fh.write(f"{index}\n")
            fh.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            fh.write(f"{text}\n\n")
            index += 1


def main(argv=None) -> int:
    """Entry point for the command-line tool.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a Japanese MP4 and create an English SRT subtitle "
            "file using a local HuggingFace Whisper model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mp4_file",
        help="Path to the input MP4 file.",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help=(
            "HuggingFace model ID to use for transcription/translation "
            "(default: openai/whisper-large-v3)."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Output SRT file path. "
            "Defaults to the input filename with the extension replaced by .srt."
        ),
    )

    args = parser.parse_args(argv)

    mp4_path = args.mp4_file
    if not os.path.isfile(mp4_path):
        print(f"Error: input file not found: {mp4_path}", file=sys.stderr)
        return 1

    srt_path = args.output or str(Path(mp4_path).with_suffix(".srt"))

    # Use a temporary file for the extracted audio so we don't litter the
    # working directory even if the script is interrupted.
    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        print(f"Extracting audio from '{mp4_path}' …")
        try:
            extract_audio(mp4_path, audio_path)
        except FileNotFoundError:
            print(
                "Error: ffmpeg not found. Please install ffmpeg and make sure it "
                "is available on your PATH.",
                file=sys.stderr,
            )
            return 1
        except RuntimeError as exc:
            print(f"Error extracting audio: {exc}", file=sys.stderr)
            return 1

        try:
            chunks = transcribe_and_translate(audio_path, args.model)
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"Writing SRT file to '{srt_path}' …")
        write_srt(chunks, srt_path)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print(f"Done! SRT subtitle file saved to '{srt_path}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
