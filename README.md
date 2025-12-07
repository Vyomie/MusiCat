# MusiCat

MusiCat is a context-aware Discord bot that combines natural language conversation with emotion-based music recommendation. It analyzes user messages, identifies intent, evaluates emotional tone, and can generate personalized Spotify playlists based on mood or direct music requests.

The bot maintains short-term user memory using SQLite, allowing natural multi-turn dialogue. It integrates with OpenRouter for LLM chat generation and playlist naming. MusiCat is optimized to run even on limited hardware such as Raspberry Pi.

## Features

### Conversation
- Context-aware chat using OpenRouter LLMs
- Per-user memory: last ten messages for up to ten minutes
- Graceful fallback when the LLM is unavailable

### Music Recommendation
- Emotion classification
- Zero-shot music tagging
- Weighted musical relevance scoring
- Sentiment-adjusted ranking
- Playlist name generation via LLM
- Automated Spotify playlist creation
- Optional user confirmation for playlists through embeds and buttons

### Discord Integration
- `$chat <message>` conversational interaction
- `$music <message>` direct playlist generation
- `$stats` view usage statistics
- `$help` list all bot commands
- `$info` system overview
- Rich embeds for all bot outputs

## Requirements

- Python 3.10 or higher  
- Discord bot token  
- Spotify Client ID and Secret  
- OpenRouter API key  

A lightweight classifier model is recommended when running on a Raspberry Pi.

## Installation

Clone the repository:
```bash
git clone https://github.com/Vyomie/MusiCat.git
cd MusiCat
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Add your API keys directly into the bot file if using hardcoded configuration.

## Running the Bot

```bash
python main.py
```

Ensure Spotify, Discord, and OpenRouter credentials are configured correctly.

## Architecture Overview

### LLM Interface
Generates conversational responses, builds context, and proposes playlist names.

### Classifier Engine
Performs zero-shot detection of user intent and music-related tags.

### Music Recommendation System
Computes sentiment, tag compatibility, and relevance scores based on a local music corpus.

### Spotify Integration
Handles user authentication, track search, and playlist construction.

### Discord Command Layer
Manages prefix commands (`$`), embeds, buttons, and all user interactions.

### SQLite Memory Store
Maintains short-term per-user conversation history.

## Commands

`$chat <text>`  
Start a conversational request. Includes emotional analysis automatically.

`$music <text>`  
Generate a playlist based on the described mood, vibe, or type of music.

`$stats`  
Display query count, playlist count, and other metrics.

`$help`  
Provide a description of all commands.

`$info`  
Explain what MusiCat is and how it works.

## Notes

- The zero-shot classifier can be swapped for a smaller model to improve speed on low-power devices.
- Playlist generation requires a valid Spotify authentication flow.
- OpenRouter rate limits may temporarily impact chat quality; fallback logic helps reduce disruptions.

## License

This project is provided under the Apache License.
