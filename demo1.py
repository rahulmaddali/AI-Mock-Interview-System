#!/usr/bin/env python3
import os
import sys
import time
import json
import pathlib
import numpy as np
import sounddevice as sd
import wavio
import whisper
import google.generativeai as genai
import librosa

# CONFIG
NUM_QUESTIONS = 2
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FOLDER = pathlib.Path("answers")
AUDIO_FOLDER.mkdir(exist_ok=True)
TRANSCRIPT_MODEL_NAME = "tiny"
GENIE_MODEL = "models/gemini-1.5-flash-latest"

# Hardcode your new API key here:
API_KEY = "AIzaSyBzoXLeZGM1DnANFXYIMyfEphp5VO9gyyw"
if not API_KEY:
    print("Set API_KEY in script")
    sys.exit(1)
genai.configure(api_key=API_KEY)

# Load Whisper model once
print("Loading Whisper model...")
whisper_model = whisper.load_model(TRANSCRIPT_MODEL_NAME)
print("Whisper loaded.")

def record_answer(filename):
    print("\nPress Enter to start recording, then Enter again to stop...")
    input()
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print("Recorder status:", status, file=sys.stderr)
        frames.append(indata.copy())

    print("Recording... press Enter to stop.")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback, dtype="int16")
    stream.start()
    input()
    stream.stop()
    stream.close()
    print("Recording stopped.")

    audio_np = np.concatenate(frames, axis=0)
    wavio.write(filename, audio_np, SAMPLE_RATE, sampwidth=2)

def transcribe_audio(path):
    result = whisper_model.transcribe(path, language="en")
    return result.get("text", "").strip()

def compute_metrics(audio_path, transcript):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr) if len(y) > 0 else 0.001
    words = transcript.split()
    wpm = (len(words) / duration) * 60 if duration > 0 else 0

    rms = np.mean(librosa.feature.rms(y=y)) if len(y) > 0 else 0

    try:
        f0 = librosa.yin(y, fmin=75, fmax=500, sr=sr)
        voiced = f0[~np.isnan(f0)]
        avg_pitch = float(np.mean(voiced)) if voiced.size > 0 else 0
        pitch_std = float(np.std(voiced)) if voiced.size > 0 else 0
    except:
        avg_pitch = 0
        pitch_std = 0

    try:
        intervals = librosa.effects.split(y, top_db=30)
        non_silent = sum((end - start) for start, end in intervals)
        silence_ratio = 1.0 - (non_silent / len(y)) if len(y) > 0 else 0
    except:
        silence_ratio = 0

    score = 5.0
    if 100 <= wpm <= 160:
        score += 2
    elif 80 <= wpm < 100 or 160 < wpm <= 180:
        score += 1
    else:
        score -= 1
    score -= silence_ratio * 2
    score -= min(pitch_std / 50, 1.5)
    score = max(0, min(10, score))

    return {
        "duration_s": float(duration),
        "words": (len(words)),
        "words_per_minute": float(wpm),
        "avg_pitch_hz": float(avg_pitch),
        "pitch_std": float(pitch_std),
        "rms": float(rms),
        "silence_ratio": float(silence_ratio),
        "confidence_score_0_10": round(score, 2)
    }

def generate_questions(topic):
    model = genai.GenerativeModel(GENIE_MODEL)
    prompt = (
        f"You are an interviewer. Based on the job/topic below, create {NUM_QUESTIONS} clear concise interview questions.\n\n"
        f"Job/Topic: {topic}\n\n"
        "Return the questions as a numbered list without extra commentary."
    )
    response = model.generate_content(prompt)
    text = response.text.strip()
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if line and (line[0].isdigit() or line[0] == '-'):
            # remove numbering like '1.' or '1)' or '-'
            parts = line.split('.', 1) if '.' in line else line.split(')', 1)
            if len(parts) > 1 and parts[0].strip().isdigit():
                q = parts[1].strip()
            else:
                q = line.lstrip('- ').strip()
            questions.append(q)
        elif line:
            questions.append(line)
        if len(questions) >= NUM_QUESTIONS:
            break
    return questions[:NUM_QUESTIONS]

def request_feedback(questions, answers, metrics):
    model = genai.GenerativeModel(GENIE_MODEL)
    prompt = "You are an expert interviewer and coach. For each Q&A below, provide:\n" \
             "1) One-line feedback\n2) Confidence rating 0-10 based on transcript and audio metrics\n" \
             "3) A suggested follow-up question if relevant.\n\n"
    for i, q in enumerate(questions, 1):
        prompt += f"Q{i}: {q}\nA{i}: {answers[i-1]}\nMetrics{i}: {json.dumps(metrics[i-1])}\n\n"
    prompt += "Return a JSON array with fields question_index, feedback, confidence_score, follow_up."
    response = model.generate_content(prompt)
    return response.text.strip()

def main():
    topic = input("Enter interview job/topic: ").strip()
    if not topic:
        topic = "software engineer"

    print(f"\nGenerating {NUM_QUESTIONS} interview questions for '{topic}'...")
    questions = generate_questions(topic)
    for idx, q in enumerate(questions, 1):
        print(f"{idx}. {q}")

    answers = []
    metrics_list = []
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q}")
        audio_file = AUDIO_FOLDER / f"answer_{i}.wav"
        record_answer(str(audio_file))
        transcript = transcribe_audio(str(audio_file))
        print(f"Transcribed answer: {transcript}")
        metrics = compute_metrics(str(audio_file), transcript)
        print(f"Local confidence: {metrics['confidence_score_0_10']}/10")
        answers.append(transcript)
        metrics_list.append(metrics)

    print("\nRequesting final feedback from Gemini...")
    feedback = request_feedback(questions, answers, metrics_list)
    print("\nFinal feedback:\n", feedback)

    report = {
        "topic": topic,
        "questions": questions,
        "answers": answers,
        "metrics": metrics_list,
        "feedback": feedback,
        "timestamp": time.time()
    }
    with open("final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full report to final_report.json")

if __name__ == "__main__":
    main()