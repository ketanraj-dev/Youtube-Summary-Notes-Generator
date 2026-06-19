import re
import math
import gradio as gr
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------------------------
# System prompt — detailed note-taking
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert note-taker who converts YouTube video transcripts into thorough, well-structured study notes. Your goal is NOT to summarise — it is to capture every piece of knowledge in the video so the reader never needs to watch it.

Write the notes in the following format exactly:

---

## 📌 Video at a Glance
- **Topic**: What the video is fundamentally about
- **Speaker / Creator**: Name if mentioned, otherwise "Not specified"
- **Target audience**: Who this video is aimed at
- **Core argument / thesis**: The single most important idea the video is trying to convey

---

## 🗒️ Notes

For EVERY concept, topic, or section in the video, create a subsection. Do not skip or merge any topic.

### [Concept / Topic Name]

**What it is**
Clear, precise definition or explanation in your own words. Write as if explaining to someone who has never heard of this before.

**Why it matters**
The reason the speaker brings this up — the problem it solves or the point it supports.

**Key details**
- Bullet every specific fact, statistic, step, rule, or argument made
- Preserve exact numbers, names, dates, and technical terms
- If the speaker lists steps or a framework, reproduce it in full

**Examples & analogies**
- Note every example, case study, story, or analogy used
- Include enough detail that the example is actually useful

**Quotes worth keeping**
> Any particularly sharp, memorable, or instructive line from the speaker — written verbatim if possible

---

*(Repeat the block above for every distinct topic — do not summarise multiple topics into one)*

---

## 🔑 Key Takeaways
The 5–10 most important things to remember from the entire video, written as standalone statements that make sense without context.

## ⚡ Action Items
Specific, concrete things the viewer is advised to do. Write each as an actionable instruction. If the video contains no action items, write "None."

## ❓ Questions & Gaps
Things the video leaves unanswered, assumptions it makes, or follow-up questions a curious viewer might want to explore.

---

Rules:
- Write notes, not prose. Prefer bullets and structured fields over paragraphs.
- Never compress or gloss over a point — if the speaker spent time on it, so should you.
- Preserve exact terminology, brand names, and proper nouns from the transcript.
- If the transcript is auto-generated and fragmented, reconstruct meaning from context — do not copy garbled text verbatim.
- Do not add opinions or information not present in the transcript.
"""

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "👋 **Welcome to YouTube Video Notes!**\n\n"
        "I turn any YouTube video into thorough, structured notes — "
        "every concept, example, and insight captured so you never need to rewatch.\n\n"
        "**To get started:**\n"
        "1. 🔑 Enter your **OpenAI API key** in the field above\n"
        "2. 📹 Paste any **YouTube video URL** below\n\n"
        "I'll fetch the transcript and write detailed notes for you!"
    ),
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_WORD_LIMIT = 10000        # words per chunk (~13k tokens)
SINGLE_PASS_WORD_LIMIT = 40000  # use single pass below this; chunk above

CHUNK_SYSTEM_PROMPT = """You are taking detailed notes on section {n} of {total} of a YouTube video transcript.

Extract EVERYTHING from this section:
- Every concept or topic introduced — with full explanation
- Every fact, statistic, name, date, or number — preserved exactly
- Every example, story, or analogy — with enough detail to be useful
- Every step, framework, or list the speaker mentions — reproduced in full
- Any memorable quotes — as close to verbatim as possible

Write structured notes with clear headings and bullets. Be thorough — this section's notes will be merged with other sections to form the complete set of video notes."""

COMBINE_PROMPT = """Below are detailed notes from all {total} sections of a YouTube video, in chronological order.

Combine them into ONE complete, well-structured set of notes following the exact format in your system prompt.
- Preserve all detail — do not compress or drop any point
- Merge overlapping content naturally where sections repeat a concept
- Keep the section headings from the notes where they are useful
- Ensure the final output flows as one coherent document"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:embed/)([0-9A-Za-z_-]{11})",
        r"(?:shorts/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def is_youtube_url(text: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", text, re.I))


def chunk_transcript(text: str, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_chunk(client: OpenAI, chunk: str, n: int, total: int) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM_PROMPT.format(n=n, total=total)},
            {"role": "user", "content": chunk},
        ],
        max_tokens=3000,
    )
    return resp.choices[0].message.content


def fetch_transcript(video_id: str) -> tuple[str | None, str | None]:
    try:
        api = YouTubeTranscriptApi()
        # Try fetching in multiple languages; fall back to any available
        try:
            transcript = api.fetch(video_id)
        except Exception:
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_transcript(
                [t.language_code for t in transcript_list]
            ).fetch()
        text = " ".join(snippet.text for snippet in transcript)
        return text, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main chat handler (streaming generator)
# ---------------------------------------------------------------------------
def respond(message: str, history: list, api_key: str, transcript_state: str):
    message = message.strip()
    if not message:
        yield history, transcript_state, ""
        return

    # Guard: API key required
    if not api_key or not api_key.strip():
        updated = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please enter your **OpenAI API key** in the field above to continue."},
        ]
        yield updated, transcript_state, ""
        return

    client = OpenAI(api_key=api_key.strip())
    history = history + [{"role": "user", "content": message}]

    # ── Branch 1: YouTube URL detected ──────────────────────────────────────
    if is_youtube_url(message):
        video_id = extract_video_id(message)
        if not video_id:
            history = history + [{"role": "assistant", "content": "❌ Couldn't extract a video ID from that URL. Please double-check and try again."}]
            yield history, transcript_state, ""
            return

        # Step 1: show fetching status
        history = history + [{"role": "assistant", "content": "⏳ Fetching transcript from YouTube..."}]
        yield history, transcript_state, ""

        # Step 2: fetch transcript
        transcript, error = fetch_transcript(video_id)
        if error or not transcript:
            history[-1] = {
                "role": "assistant",
                "content": (
                    "❌ Could not fetch the transcript.\n\n"
                    "This usually means the video has no captions or they are disabled.\n\n"
                    f"**Details:** {error}"
                ),
            }
            yield history, transcript_state, ""
            return

        transcript_state = transcript
        word_count = len(transcript.split())
        estimated_mins = math.ceil(word_count / 130)

        history[-1] = {
            "role": "assistant",
            "content": (
                f"✅ Transcript fetched! ({word_count:,} words, ~{estimated_mins} min video)\n\n"
                "Writing detailed notes — please wait..."
            ),
        }
        yield history, transcript_state, ""

        # Step 3: single-pass or map-reduce depending on length
        try:
            if word_count <= SINGLE_PASS_WORD_LIMIT:
                # ── Single pass ──────────────────────────────────────────────
                prompt = f"Write complete, detailed notes for the following YouTube video transcript:\n\n{transcript}"
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    max_tokens=8192,
                )
                response = ""
                for chunk_part in stream:
                    delta = chunk_part.choices[0].delta.content
                    if delta:
                        response += delta
                        history[-1] = {"role": "assistant", "content": response}
                        yield history, transcript_state, ""

            else:
                # ── Map-reduce for very long videos ──────────────────────────
                chunks = chunk_transcript(transcript)
                total = len(chunks)
                chunk_summaries = []

                for i, chunk_text_part in enumerate(chunks, 1):
                    history[-1] = {
                        "role": "assistant",
                        "content": (
                            f"📝 Summarizing section **{i} of {total}**...\n\n"
                            f"*(This video is long — processing each section for accuracy)*"
                        ),
                    }
                    yield history, transcript_state, ""
                    summary = summarize_chunk(client, chunk_text_part, i, total)
                    chunk_summaries.append(f"### Section {i} of {total}\n{summary}")

                # Final synthesis — streamed
                history[-1] = {"role": "assistant", "content": f"🔄 Combining all {total} sections into final summary..."}
                yield history, transcript_state, ""

                combined = "\n\n---\n\n".join(chunk_summaries)
                final_prompt = COMBINE_PROMPT.format(total=total) + f"\n\n{combined}"
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": final_prompt},
                    ],
                    stream=True,
                    max_tokens=8192,
                )
                response = ""
                for chunk_part in stream:
                    delta = chunk_part.choices[0].delta.content
                    if delta:
                        response += delta
                        history[-1] = {"role": "assistant", "content": response}
                        yield history, transcript_state, ""

        except Exception as exc:
            history[-1] = {"role": "assistant", "content": f"❌ OpenAI error: {exc}"}
            yield history, transcript_state, ""

    # ── Branch 2: Follow-up question / normal chat ───────────────────────────
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if transcript_state:
            # Cap transcript context for follow-up questions (~50k chars ≈ 12k tokens)
            ctx = transcript_state[:50000]
            messages.append({
                "role": "system",
                "content": f"The user has already shared a video. Here is its transcript for reference:\n\n{ctx}",
            })

        for h in history[:-1]:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        history = history + [{"role": "assistant", "content": ""}]
        try:
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
            )
            response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    response += delta
                    history[-1] = {"role": "assistant", "content": response}
                    yield history, transcript_state, ""
        except Exception as exc:
            history[-1] = {"role": "assistant", "content": f"❌ OpenAI error: {exc}"}
            yield history, transcript_state, ""


# ---------------------------------------------------------------------------
# Gradio UI  (compatible with Gradio 6+)
# ---------------------------------------------------------------------------
CSS = """
    footer { display: none !important; }
    #chatbot { border-radius: 12px; }
"""

with gr.Blocks(title="YouTube Video Summarizer") as demo:

    gr.Markdown(
        "# 🎬 YouTube Video Notes\n"
        "*Paste any YouTube URL to get thorough, structured notes covering every concept, example, and insight.*"
    )

    api_key_input = gr.Textbox(
        label="🔑 OpenAI API Key",
        placeholder="sk-...",
        type="password", 
    )

    transcript_state = gr.State("")

    chatbot = gr.Chatbot(
        value=[WELCOME_MESSAGE],
        height=580,
        show_label=False,
        elem_id="chatbot",
        layout="bubble",
        render_markdown=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Paste a YouTube URL here, or ask a follow-up question...",
            show_label=False,
            scale=8,
            container=False,
        )
        send_btn = gr.Button("Send ▶", scale=1, variant="primary")

    clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")

    # Wire up events
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, api_key_input, transcript_state],
        outputs=[chatbot, transcript_state, msg_box],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, api_key_input, transcript_state],
        outputs=[chatbot, transcript_state, msg_box],
    )
    clear_btn.click(
        lambda: ([WELCOME_MESSAGE], ""),
        outputs=[chatbot, transcript_state],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
        css=CSS,
    )
