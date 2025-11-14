#!/usr/bin/env python3
"""
Interview Analyzer (optimized for free-tier):
 - 1 call to Gemini to generate N questions
 - Local recording + local Whisper transcription for each answer
 - Local audio-based tone/confidence metrics
 - 1 final call to Gemini for consolidated feedback
"""

import os
import sys
import time
import json
import tempfile
import pathlib
import threading
import numpy as np
import sounddevice as sd
import wavio
import whisper
from gtts import gTTS
import google.generativeai as genai
import librosa

# -----------------------
# CONFIG
# -----------------------
NUM_QUESTIONS = 10
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FOLDER = pathlib.Path("answers")
AUDIO_FOLDER.mkdir(exist_ok=True)
TRANSCRIPT_MODEL_NAME = "tiny"  # whisper tiny -> fast and small
GENIE_MODEL = "models/gemini-1.5-flash-latest"  # use flash by default

# Read Gemini API key from environment
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: Set your Gemini API key in environment variable GOOGLE_API_KEY")
    print("  export GOOGLE_API_KEY='YOUR_KEY_HERE'")
    sys.exit(1)
genai.configure(api_key=API_KEY)

# Initialize Whisper model once
print("Loading Whisper model (this may take a while the first time)...")
whisper_model = whisper.load_model(TRANSCRIPT_MODEL_NAME)
print("Whisper model loaded.")

# Utility: text-to-speech (gTTS) playback using system player
def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    path = tmp.name
    try:
        tts.save(path)
        # macOS: afplay; Linux: mpg123 or play; Windows: start
        if sys.platform.startswith("darwin"):
            os.system(f"afplay {path} >/dev/null 2>&1")
        elif sys.platform.startswith("linux"):
            # try mpg123 or ffplay
            if shutil.which("mpg123"):
                os.system(f"mpg123 -q {path}")
            else:
                os.system(f"ffplay -nodisp -autoexit -loglevel quiet {path}")
        elif sys.platform.startswith("win"):
            os.system(f"start {path}")
    finally:
        try:
            os.remove(path)
        except:
            pass

# -----------------------
# Gemini helpers
# -----------------------
def generate_questions(prompt_text, n_questions=NUM_QUESTIONS):
    """
    Ask Gemini to create N concise interview questions for the given prompt_text.
    Returns list of strings.
    """
    model = genai.GenerativeModel(GENIE_MODEL)
    system_prompt = (
        f"You are an interviewer. Based on the job/topic below, create {n_questions} "
        "clear concise interview questions (one line each). Return them as a numbered list.\n\n"
        f"Job/Topic: {prompt_text}\n\n"
        "Only output the questions numbered 1..N, without extra commentary."
    )
    resp = model.generate_content(system_prompt)
    text = resp.text.strip()
    # Try to parse numbered or line-based output
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    questions = []
    for line in lines:
        # strip leading numbering if present
        # e.g., "1. Question..." or "1) Question..."
        q = line
        if q and (q[0].isdigit() or (len(q) > 1 and q[:2].isdigit())):
            # remove a prefix like "1." or "1)"
            parts = q.split(".", 1) if "." in q else q.split(")", 1)
            if len(parts) > 1 and parts[0].strip().isdigit():
                q = parts[1].strip()
        questions.append(q)
        if len(questions) >= n_questions:
            break
    # If not enough parsed, fill from lines
    if len(questions) < n_questions:
        # attempt splitting by sentence punctuation
        flat = " ".join(lines)
        cand = [s.strip() for s in flat.split("?") if s.strip()]
        for c in cand:
            if len(questions) >= n_questions:
                break
            questions.append(c + "?")
    # final trim
    questions = questions[:n_questions]
    return questions

# -----------------------
# Recording utilities (press Enter to start/stop)
# -----------------------
def record_press_enter_save(filename, samplerate=SAMPLE_RATE, channels=CHANNELS):
    """
    Press Enter to start, press Enter to stop. Saves to WAV at filename.
    """
    print("\nPress Enter to start recording (then press Enter again to stop)...")
    input()
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print("Recorder status:", status, file=sys.stderr)
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=callback, dtype="int16")
    stream.start()
    input()  # stop on Enter
    stream.stop()
    stream.close()
    print("Recording stopped, saving...")

    # concat and save
    audio_np = np.concatenate(frames, axis=0)
    # wavio wants shape (Nsamples, channels) and dtype int16
    wavio.write(filename, audio_np, samplerate, sampwidth=2)
    return filename

# -----------------------
# Local transcription (Whisper) and audio metrics
# -----------------------
def transcribe_whisper(path):
    # whisper.load_model done globally
    # whisper expects sampling rate of 16000 or will resample
    result = whisper_model.transcribe(path, language="en")
    text = result.get("text", "").strip()
    duration = result.get("duration", None)
    return text, result

def compute_audio_metrics(audio_path, transcript_text):
    """
    Returns a dict with simple confidence/tone metrics:
      - words_per_minute
      - avg_pitch (Hz)
      - pitch_std
      - rms (volume mean)
      - silence_ratio
    """
    y, sr = librosa.load(audio_path, sr=None)  # keep original sr
    # duration
    duration = librosa.get_duration(y=y, sr=sr) if len(y) > 0 else 0.0001
    # words per minute from transcript
    words = transcript_text.split()
    wpm = (len(words) / duration) * 60 if duration > 0 else 0.0

    # RMS (volume)
    rms = np.mean(librosa.feature.rms(y=y)) if len(y) > 0 else 0.0

    # Pitch estimate via librosa.yin (robust)
    try:
        f0 = librosa.yin(y, fmin=75, fmax=500, sr=sr)
        # filter out unvoiced frames
        voiced = f0[~np.isnan(f0)]
        avg_pitch = float(np.mean(voiced)) if voiced.size > 0 else 0.0
        pitch_std = float(np.std(voiced)) if voiced.size > 0 else 0.0
    except Exception:
        avg_pitch = 0.0
        pitch_std = 0.0

    # silence ratio: proportion of frames below a low-energy threshold
    try:
        intervals = librosa.effects.split(y, top_db=30)  # non-silent intervals
        non_silent = sum((end - start) for start, end in intervals)
        silence_ratio = 1.0 - (non_silent / len(y)) if len(y) > 0 else 0.0
    except Exception:
        silence_ratio = 0.0

    metrics = {
        "duration_s": float(duration),
        "words": len(words),
        "words_per_minute": float(wpm),
        "avg_pitch_hz": float(avg_pitch),
        "pitch_std": float(pitch_std),
        "rms": float(rms),
        "silence_ratio": float(silence_ratio),
    }
    # Simple confidence heuristic (0-10)
    # More WPM between 100-160, lower silence, low pitch variance => higher confidence
    score = 5.0
    # adjust for speaking rate
    if 100 <= wpm <= 160:
        score += 2.0
    elif 80 <= wpm < 100 or 160 < wpm <= 180:
        score += 1.0
    else:
        score -= 1.0
    # silence reduces confidence
    score -= silence_ratio * 2.0
    # pitch variability reduces score
    score -= min(pitch_std / 50.0, 1.5)
    # clamp
    score = max(0.0, min(10.0, score))
    metrics["confidence_score_0_10"] = float(round(score, 2))
    return metrics

# -----------------------
# Final consolidated feedback call
# -----------------------
def request_final_feedback(questions, answers, metrics_list):
    """
    Sends all Q&A + audio-derived metrics to Gemini in a single API call for final feedback & scoring.
    """
    model = genai.GenerativeModel(GENIE_MODEL)

    # Build a clear prompt: include short instructions, questions, answers, and metrics
    payload = "You are an experienced interviewer and coach. For each question+answer below, provide:\n" \
              "1) A one-line concise feedback (strengths + improvement)  \n" \
              "2) A confidence rating 0-10 based on both the transcript and the provided audio-derived metrics  \n" \
              "3) A short suggested follow-up question if applicable.\n\n"

    payload += "Now the data:\n\n"
    for i, q in enumerate(questions, start=1):
        payload += f"Q{i}: {q}\n"
        payload += f"A{i}: {answers[i-1]}\n"
        payload += f"Metrics{i}: {json.dumps(metrics_list[i-1])}\n\n"

    payload += "\nProduce a JSON array 'results' where each element is {question_index, feedback, confidence_score, follow_up} and a final summary at the end."

    # single generate_content call
    resp = model.generate_content(payload)
    return resp.text

# -----------------------
# MAIN INTERVIEW FLOW
# -----------------------
def run_interview():
    topic = input("Enter job/topic for interview (e.g., 'junior data scientist'): ").strip()
    if not topic:
        topic = "junior data scientist"

    print("\nGenerating interview questions (1 API call)...")
    questions = generate_questions(topic, n_questions=NUM_QUESTIONS)
    print("\nQuestions generated:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    # Loop: play Q, record A, transcribe locally, compute metrics, store
    answers = []
    metrics_list = []
    for i, q in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {q}")
        # speak_text(q)  # optional: play audio via gTTS (disabled to be faster); enable if you like
        # record
        audio_path = AUDIO_FOLDER / f"answer_{i}.wav"
        record_press_enter_save(str(audio_path))
        # transcribe
        transcript, _ = transcribe_whisper(str(audio_path))
        answers.append(transcript)
        # metrics
        metrics = compute_audio_metrics(str(audio_path), transcript)
        metrics_list.append(metrics)
        # quick local feedback (optional): show confidence now
        print(f"Local metrics: confidence ~ {metrics['confidence_score_0_10']} /10 (wpm={metrics['words_per_minute']:.1f})")
        # save transcript file
        with open(AUDIO_FOLDER / f"answer_{i}.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

    # Final consolidated feedback via 1 Gemini call
    print("\nRequesting final feedback from Gemini (1 API call)...")
    final_text = request_final_feedback(questions, answers, metrics_list)
    # Save report
    report = {
        "topic": topic,
        "questions": questions,
        "answers": answers,
        "metrics": metrics_list,
        "gemini_feedback_raw": final_text,
        "timestamp": time.time()
    }
    with open("final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nFinal feedback (raw from Gemini):\n")
    print(final_text)
    print("\nSaved full report to final_report.json and individual answers in 'answers/' folder.")

if __name__ == "__main__":
    run_interview()