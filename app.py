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
import torch.nn.functional as F
import signal
import librosa  # Add this import at the top of the file

# Remove this import as we'll be using a different import
# from silero_vad import VAD, collect_chunks

# Add these imports
from silero_vad import load_silero_vad, get_speech_timestamps



"""
A real-time voice interface system that creates a seamless conversation flow between users and AI.

This script implements a complete audio processing pipeline that combines voice activity detection,
speech recognition, natural language processing, and speech synthesis. It utilizes various OpenAI
APIs (Whisper, GPT-4, and TTS) to create an interactive voice assistant.

Key Components:
    - Audio Input Processing: Captures and processes microphone input with real-time level monitoring
    - Voice Activity Detection (VAD): Uses Silero VAD to identify speech segments
    - Speech-to-Text: Converts detected speech to text using OpenAI's Whisper
    - Conversational AI: Processes text through GPT-4 for intelligent responses
    - Text-to-Speech: Converts AI responses to natural speech using OpenAI's TTS
    - Monitoring: Includes comprehensive logging and cost estimation

The system processes audio in real-time blocks, detecting speech segments and ignoring silence.
When speech is detected, it's transcribed and processed through GPT-4, with responses being
synthesized and played back to the user immediately.

Requirements:
    - OpenAI API key set in environment variables
    - Python packages: sounddevice, numpy, torch, openai, pydub, librosa, etc.
    - Audio input/output capabilities

Usage:
    Run the script directly to start the voice interface:
    $ python app.py

    The system will first run an audio test, then begin listening for speech.
    Use Ctrl+C to gracefully shut down the application.

Note:
    Token usage is tracked and cost estimates are provided upon shutdown.
"""


# Set your OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration Parameters
SAMPLE_RATE = 44100  # Increased from 16000
CHANNELS = 1
BUFFER_SIZE = 4096  # Increased and aligned to power of 2
BLOCK_DURATION = (BUFFER_SIZE / SAMPLE_RATE) * 50  # ms
FRAME_DURATION = 30  # ms
DEVICE = None  # Default input device
VAD_MODE = "0.3"  # Aggressiveness of VAD: 0-3
WHISPER_MODEL = "whisper-1"
COMPLETIONS_MODEL = "chatGPT-4o-latest"
TOKENIZER_MODEL = "gpt-4o-latest"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"  # Choose your preferred voice
TOKEN_COST_INPUT = 5 / 1e6  # $5 per million input tokens
TOKEN_COST_OUTPUT = 15 / 1e6  # $15 per million output tokens

# Logging Configuration
logging.basicConfig(level=logging.DEBUG,  # Changed from INFO to DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('app.log', mode='w')
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
tokenizer = encoding_for_model(TOKENIZER_MODEL)

# Add this constant
VAD_THRESHOLD = 0.02  # Increased from 0.01 to make it less sensitive

# Add these variables for logging control
AUDIO_BLOCK_LOG_INTERVAL = 5  # Log every 5 seconds for audio blocks
last_audio_block_log_time = 0

# Add this global variable
audio_stream = None

# Add these constants
NORMALIZATION_FACTOR = 0.01  # Adjust as needed

# Add this global variable at the top of your script
should_stop = False

# Add this function for the audio level meter
def print_audio_level(db):
    meter_length = 50
    if db < -60:
        meter = "[" + " " * meter_length + "]"
    else:
        level = int((db + 60) / 60 * meter_length)
        meter = "[" + "#" * level + " " * (meter_length - level) + "]"
    print(f"\rAudio Level: {meter} {db:.2f} dB", end="", flush=True)

def audio_callback(indata, frames, time_info, status):
    """Callback function called by sounddevice for each audio block."""
    global last_audio_block_log_time
    current_time = time.time()

    logger.debug(f"Audio callback called: frames={frames}, time_info={time_info}")

    if status:
        logger.warning(f"Audio callback status: {status}")

    # Convert stereo to mono if necessary
    mono_data = np.mean(indata, axis=1) if indata.shape[1] == 2 else indata.flatten()

    # Normalize audio data
    max_val = np.max(np.abs(mono_data))
    if max_val > 0:
        mono_data = mono_data / max_val * NORMALIZATION_FACTOR

    # Calculate RMS and convert to dB
    rms = np.sqrt(np.mean(mono_data**2))
    db = 20 * np.log10(max(rms, 1e-10))  # Avoid log(0)

    # Print audio level meter
    print_audio_level(db)

    # Log audio block info less frequently
    if current_time - last_audio_block_log_time >= AUDIO_BLOCK_LOG_INTERVAL:
        logger.info(f"Audio block received: shape={mono_data.shape}, dtype={mono_data.dtype}, "
                    f"max={np.max(np.abs(mono_data)):.4f}, min={np.min(mono_data):.4f}, "
                    f"mean={np.mean(mono_data):.4f}, RMS={rms:.4f}, dB={db:.2f}")
        last_audio_block_log_time = current_time

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
    global input_tokens, should_stop

    while not should_stop:
        try:
            transcript = await asyncio.wait_for(
                asyncio.to_thread(transcribe_audio, await transcript_queue.get()),
                timeout=1.0  # Add a timeout to allow checking should_stop
            )
            if transcript:
                print(f"Transcribed: {transcript}")
                conversation_log.append({"role": "user", "content": transcript})
                input_tokens += len(tokenizer.encode(transcript))  # Accurate token count

                # Generate response
                await generate_response(transcript, conversation_log)
        except asyncio.TimeoutError:
            continue  # This allows the loop to check should_stop regularly
        except Exception as e:
            logger.error(f"Error in process_transcriptions: {e}")
            if not should_stop:
                await asyncio.sleep(1)  # Wait a bit before retrying if not stopping

async def play_responses():
    """Placeholder for playing responses if needed."""
    pass  # Already handled in handle_tts

def estimate_cost():
    """Estimate the cost based on token usage."""
    cost = (input_tokens * TOKEN_COST_INPUT) + (output_tokens * TOKEN_COST_OUTPUT)
    print(f"Estimated Cost: ${cost:.6f}")

async def main():
    global audio_queue, audio_stream, should_stop

    logger.info("Entering main function")
    print("Starting the voice interface...")

    try:
        logger.info("Initializing audio stream")
        devices = sd.query_devices()
        input_device = next((i for i, d in enumerate(devices) if "MacBook Pro Microphone" in d['name']), None)
        if input_device is None:
            logger.warning("Could not find MacBook Pro Microphone, using default input device")
            input_device = sd.default.device[0]

        logger.info(f"Using input device: {input_device}")
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=input_device,
            channels=CHANNELS,
            callback=audio_callback,
            blocksize=BUFFER_SIZE
        )
        audio_stream.start()
        logger.info(f"Audio stream initialized successfully with device: {input_device}")
        print("Listening... Press Ctrl+C to stop.")

        # Correct the processing task name
        processing_task = asyncio.create_task(process_transcriptions())

        buffer = []
        last_speech_time = time.time()
        last_log_time = time.time()
        while not should_stop:
            try:
                audio_block = audio_queue.get(timeout=0.1)  # Add a timeout
                buffer.append(audio_block)
                logger.debug(f"Audio block added to buffer. Buffer size: {len(buffer)}")

                # Process if we have at least 1 second of audio
                if len(buffer) * BLOCK_DURATION >= 50:
                    audio_np = np.concatenate(buffer)
                    logger.debug(f"Processing audio chunk: shape={audio_np.shape}, duration={len(audio_np)/SAMPLE_RATE:.2f}s")

                    # After downsampling to 16kHz
                    audio_16k = librosa.resample(audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)

                    # Ensure audio is in the correct format for VAD
                    audio_tensor = torch.from_numpy(audio_16k.astype(np.float32))

                    logger.debug("Applying VAD model")

                    # Detect speech regions directly using the get_speech_timestamps function
                    speech_timestamps = get_speech_timestamps(audio_16k, vad_model, sampling_rate=16000)
                    logger.debug(f"Speech timestamps: {speech_timestamps}")

                    current_time = time.time()
                    if speech_timestamps:
                        print("\nSpeech detected!")
                        logger.info(f"VAD: Speech detected.")
                        transcript_queue.put(audio_np)  # Use original audio for transcription
                        buffer = []
                        last_speech_time = current_time
                        last_log_time = current_time
                    else:
                        # No speech detected, log every 5 seconds
                        if current_time - last_log_time >= 5:
                            print(f"\nNo speech detected for {current_time - last_speech_time:.1f} seconds")
                            logger.info(f"No speech detected for {current_time - last_speech_time:.1f} seconds.")
                            last_log_time = current_time

                    # Clear buffer if it's too large, keeping the last 1 second
                    if len(buffer) * BLOCK_DURATION > 50:
                        buffer = buffer[-int(1000 / BLOCK_DURATION):]

            except queue.Empty:
                logger.debug("Audio queue empty")

            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping the application")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
    finally:
        should_stop = True
        if audio_stream:
            audio_stream.stop()
            audio_stream.close()
        if 'processing_task' in locals():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                logger.info("Processing task canceled successfully.")
        estimate_cost()
        logger.info("Exiting main function")

def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down...")
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    # Instead of sys.exit(0), we'll set a flag to stop the main loop
    asyncio.get_event_loop().stop()

def test_audio():
    try:
        logger.info("Testing audio recording and playback...")
        duration = 3  # seconds
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        print("Recording...")
        sd.wait()

        # Calculate and log audio statistics
        rms = np.sqrt(np.mean(recording**2))
        db = 20 * np.log10(max(rms, 1e-10))
        logger.info(f"Recording complete. Shape: {recording.shape}, Max: {np.max(np.abs(recording)):.4f}, "
                    f"RMS: {rms:.4f}, dB: {db:.2f}")

        # Apply a fade in and fade out to reduce pops
        fade_duration = int(0.01 * SAMPLE_RATE)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_duration).reshape(-1, 1)
        fade_out = np.linspace(1, 0, fade_duration).reshape(-1, 1)

        # Ensure fade_in and fade_out match the number of channels in recording
        if recording.shape[1] > 1:
            fade_in = np.tile(fade_in, (1, recording.shape[1]))
            fade_out = np.tile(fade_out, (1, recording.shape[1]))

        recording[:fade_duration] *= fade_in
        recording[-fade_duration:] *= fade_out

        print("Playing back...")
        sd.play(recording, SAMPLE_RATE)
        sd.wait()
        logger.info("Playback complete")
    except Exception as e:
        logger.error(f"Error in test_audio: {e}")
        logger.error(f"Recording shape: {recording.shape}, fade_in shape: {fade_in.shape}, fade_out shape: {fade_out.shape}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        logger.info("Starting the application")
        test_audio()  # Keep this line
        print("Available audio devices:")
        print(sd.query_devices())
        print(f"Default input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Application shut down.")