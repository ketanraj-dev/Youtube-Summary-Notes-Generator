# 🎬 YouTube Video Notes

A Gradio-based chat app that fetches the transcript of any YouTube video and writes thorough, structured notes using OpenAI's GPT models — with real-time streaming output.

---


## Features

- **Chat interface** — conversational flow that asks for a YouTube URL, then generates the notes
- **Streaming output** — notes stream token-by-token as they are generated
- **Detailed note-taking** — captures every concept, example, quote, and insight; never compresses or skips content
- **Long video support** — handles videos of any length via automatic map-reduce chunking for videos over ~5 hours
- **Multi-language transcripts** — falls back to any available language if English captions are not found
- **Follow-up questions** — ask questions about the video after the notes are generated
- **Supports all YouTube URL formats** — `youtube.com/watch?v=`, `youtu.be/`, Shorts, embeds

---

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A conda or virtual environment (recommended)

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Youtube_summary
```

### 2. Create and activate a conda environment

```bash
conda create -n youtubesummary python=3.11
conda activate youtubesummary
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

The app opens automatically at `http://127.0.0.1:7860`.

---

## Usage

1. Enter your **OpenAI API key** in the field at the top of the page
2. Paste any **YouTube video URL** into the chat box and press Enter
3. The app fetches the transcript and streams detailed notes
4. Ask **follow-up questions** about the video in the same chat

---

## How It Works

| Video length | Word count | Strategy |
|---|---|---|
| Up to ~5 hours | < 40,000 words | Single API call, streamed directly |
| Over ~5 hours | ≥ 40,000 words | Map-reduce: each 10,000-word chunk is noted individually, then combined into one final streamed output |

**Notes format:**

| Section | What it contains |
|---|---|
| 📌 Video at a Glance | Topic, speaker, target audience, core thesis |
| 🗒️ Notes | One subsection per concept — definition, why it matters, key details, examples & analogies, notable quotes |
| 🔑 Key Takeaways | 5–10 standalone statements worth remembering |
| ⚡ Action Items | Concrete steps or advice from the video |
| ❓ Questions & Gaps | What the video leaves unanswered; follow-up topics to explore |

---

## Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Chat UI |
| `openai` | GPT-4o / GPT-4o-mini via streaming API |
| `youtube-transcript-api` | Fetching video transcripts without browser automation |

---

## Notes

- Only videos with **captions enabled** can be processed (auto-generated captions work fine)
- Your OpenAI API key is never stored — it is used only for the current session
- GPT-4o-mini is used by default to keep costs low; swap the model name in `app.py` if you prefer GPT-4o
