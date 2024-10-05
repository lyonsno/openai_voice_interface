import os
import asyncio
import sounddevice as sd
import numpy as np
import torch
import queue
import threading
import openai
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import time
import io
from tiktoken import encoding_for_model
import logging
from logging.handlers import RotatingFileHandler
import sys

# Remove this line as we'll be using a different import
# from silero_vad import VAD, collect_chunks

# Add these imports
from silero_vad import load_silero_vad, get_speech_timestamps

# Set your OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration Parameters
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
BLOCK_DURATION = 30  # ms
FRAME_DURATION = 30  # ms
DEVICE = None  # Default input device
VAD_MODE = "3"  # Aggressiveness of VAD: 0-3
WHISPER_MODEL = "whisper-1"
COMPLETIONS_MODEL = "gpt-4o-latest"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"  # Choose your preferred voice
TOKEN_COST_INPUT = 5 / 1e6  # $5 per million input tokens
TOKEN_COST_OUTPUT = 15 / 1e6  # $15 per million output tokens

# Logging Configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('app.log')
                    ])
logger = logging.getLogger(__name__)

# Initialize VAD
vad_model = load_silero_vad()

# Initialize Queues
audio_queue = queue.Queue()
transcript_queue = queue.Queue()
response_queue = queue.Queue()

# Conversation Log
conversation_log = []

# Token Counters
input_tokens = 0
output_tokens = 0

# Tokenizer
tokenizer = encoding_for_model(COMPLETIONS_MODEL)

def audio_callback(indata, frames, time_info, status):
    """Callback function called by sounddevice for each audio block."""
    if status:
        print(f"Audio callback status: {status}")
    # Convert stereo to mono if necessary
    if indata.shape[1] == 2:
        mono_data = np.mean(indata, axis=1)
    else:
        mono_data = indata.flatten()
    audio_queue.put(mono_data)

def transcribe_audio(audio_data):
    """Send audio data to Whisper for transcription."""
    with io.BytesIO() as audio_buffer:
        # Convert numpy array to audio segment
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=audio_data.dtype.itemsize,
            channels=CHANNELS
        )
        audio_segment.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        try:
            transcript = openai.Audio.transcribe(WHISPER_MODEL, audio_buffer)
            logger.info(f"Whisper transcription: {transcript.get('text', '')}")
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""

    return transcript.get('text', "")

async def generate_response(prompt, log):
    """Generate a response using OpenAI's GPT-4 with streaming."""
    global output_tokens

    response_text = ""
    sentence_buffer = ""
    sentences_to_tts = []

    try:
        logger.info(f"Sending prompt to GPT-4: {prompt}")
        async for chunk in openai.ChatCompletion.create(
            model=COMPLETIONS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            delta = chunk.choices[0].delta
            if 'content' in delta:
                response_text += delta['content']
                # Simple sentence detection based on periods
                sentence_buffer += delta['content']
                if '.' in sentence_buffer:
                    sentences = sentence_buffer.split('.')
                    for s in sentences[:-1]:
                        sentence = s.strip() + '.'
                        sentences_to_tts.append(sentence)
                        log.append({"role": "assistant", "content": sentence})
                        output_tokens += len(tokenizer.encode(sentence))  # Accurate token count
                    sentence_buffer = sentences[-1]

                    # Play sentences as they are detected
                    for sentence in sentences_to_tts:
                        asyncio.create_task(handle_tts(sentence))
                    sentences_to_tts = []

        # Handle any remaining sentence buffer
        if sentence_buffer.strip():
            sentences_to_tts.append(sentence_buffer.strip())
            log.append({"role": "assistant", "content": sentence_buffer.strip()})
            output_tokens += len(tokenizer.encode(sentence_buffer.strip()))  # Accurate token count
            for sentence in sentences_to_tts:
                asyncio.create_task(handle_tts(sentence))

        logger.info(f"GPT-4 response: {response_text}")
    except Exception as e:
        logger.error(f"Error during response generation: {e}")

    return response_text

def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS API."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_speech_file:
        speech_file_path = temp_speech_file.name

    try:
        response = openai.Audio.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text
        )
    except Exception as e:
        logger.error(f"Error during TTS: {e}")
        return

    with open(speech_file_path, 'wb') as f:
        f.write(response['audio'])

    # Play the audio
    audio = AudioSegment.from_file(speech_file_path, format="mp3")
    play(audio)

    os.remove(speech_file_path)

async def handle_tts(text):
    """Asynchronous handler for TTS to prevent blocking."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, text_to_speech, text)

async def process_transcriptions():
    """Process transcriptions and generate responses."""
    global input_tokens

    while True:
        transcript = await asyncio.to_thread(transcribe_audio, await transcript_queue.get())
        if transcript:
            print(f"Transcribed: {transcript}")
            conversation_log.append({"role": "user", "content": transcript})
            input_tokens += len(tokenizer.encode(transcript))  # Accurate token count

            # Generate response
            await generate_response(transcript, conversation_log)

async def play_responses():
    """Placeholder for playing responses if needed."""
    pass  # Already handled in handle_tts

def estimate_cost():
    """Estimate the cost based on token usage."""
    cost = (input_tokens * TOKEN_COST_INPUT) + (output_tokens * TOKEN_COST_OUTPUT)
    print(f"Estimated Cost: ${cost:.6f}")

async def main():
    global audio_queue

    print("Starting the voice interface...")
    logger.info("Application started")

    # Start audio stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE, channels=CHANNELS,
                            callback=audio_callback, blocksize=int(SAMPLE_RATE * BLOCK_DURATION / 1000)):
            print("Listening... Press Ctrl+C to stop.")
            processing_task = asyncio.create_task(process_transcriptions())

            buffer = []
            last_speech_time = time.time()
            last_log_time = time.time()
            while True:
                try:
                    audio_block = audio_queue.get(timeout=1)  # 1 second timeout
                    buffer.append(audio_block)

                    # Convert buffer to numpy array
                    audio_np = np.concatenate(buffer)

                    # Apply VAD
                    speech_timestamps = get_speech_timestamps(torch.from_numpy(audio_np.astype(np.float32)), vad_model)

                    current_time = time.time()
                    if speech_timestamps:
                        print("Speech detected!")
                        logger.info("VAD: Speech detected")
                        transcript_queue.put(audio_np)
                        buffer = []
                        last_speech_time = current_time
                        last_log_time = current_time
                    else:
                        # No speech detected, log every 5 seconds
                        if current_time - last_log_time >= 5:
                            print(f"No speech detected for {current_time - last_speech_time:.1f} seconds")
                            logger.info(f"No speech detected for {current_time - last_speech_time:.1f} seconds")
                            last_log_time = current_time

                except queue.Empty:
                    # Queue is empty, just continue
                    pass

                await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping the application...")
        logger.info("Application stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.exception("An unexpected error occurred")
    finally:
        if 'processing_task' in locals():
            processing_task.cancel()
        estimate_cost()
        print("Application terminated.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.exception("An unexpected error occurred in the main execution")