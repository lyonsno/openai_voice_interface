# Real-Time Voice Interface with LLMs

This Python project implements a real-time API-based voice interface that interacts with Large Language Models (LLMs).

## Features

- Real-time voice input processing
- Integration with LLM APIs (e.g., OpenAI GPT)
- Text-to-speech output for LLM responses
- Configurable voice recognition and synthesis
- Extensible architecture for multiple LLM providers

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/voice-llm-interface.git
   cd voice-llm-interface
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API keys:
   Copy `.env.example` to `.env` and fill in your API keys for the LLM service you're using.

## Usage

Run the main script:
