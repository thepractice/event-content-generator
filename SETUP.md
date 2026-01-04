# Event Content Generator - Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Set Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_gemini_key_here  # Optional: for image generation
```

**Required:**
- **OPENAI_API_KEY** — GPT-4o for content generation (drafter, critic, verifier nodes)
- **Local sentence-transformers** for embeddings (no API needed)

**Optional:**
- **GEMINI_API_KEY** — Gemini 2.5 Flash for marketing image generation. Get one at [Google AI Studio](https://aistudio.google.com/apikey)

### 3. Run the App
```bash
python3 -m streamlit run app.py
```

Opens at: http://localhost:8501

### 4. Ingest Corpus (First Time)
Click "Ingest Corpus Documents" in the sidebar. This:
- Loads documents from `corpus/` folder
- Chunks them into ~500 char segments
- Stores embeddings in `chroma_db/` (local)

## Project Structure

```
event-content-generator/
├── src/
│   ├── graph.py          # LangGraph pipeline (retrieve→draft→critic→verify→images→export)
│   ├── schemas.py        # Pydantic models (Claim, ChannelDraft, CriticFeedback)
│   ├── prompts.py        # All LLM prompts + image prompts
│   ├── nodes/            # Pipeline nodes
│   │   ├── retriever.py  # RAG chunk retrieval
│   │   ├── drafter.py    # Content generation (GPT-4o)
│   │   ├── critic.py     # Quality scoring
│   │   ├── verifier.py   # Claim verification
│   │   ├── image_generator.py  # Marketing image generation (Gemini)
│   │   └── exporter.py   # Final output packaging
│   └── rag/
│       ├── ingest.py     # Corpus → ChromaDB
│       └── retrieve.py   # Vector search
├── corpus/               # Brand docs (add your own here)
├── chroma_db/            # Vector store (auto-created)
├── app.py                # Streamlit UI
└── .env                  # API keys (not in git)
```

## Adding Your Own Corpus

1. Add `.md` or `.txt` files to `corpus/`
2. Name files to indicate type:
   - `*brand*` or `*voice*` → brand_voice collection
   - `*product*` or `*docs*` → product_docs collection
   - Other → included in both
3. Click "Ingest Corpus Documents" to re-index

## Example Input

| Field | Example |
|-------|---------|
| Event Title | Zero Trust Security Webinar |
| Event Description | Join us for a deep dive into implementing Zero Trust architecture. Learn from security experts about identity-first security approaches. |
| Target Audience | Enterprise Security Leaders and CTOs |
| Key Messages | Zero Trust starts with identity<br>Traditional perimeter security is obsolete<br>Reduces breach risk significantly |
| Relevant URLs | Registration \| https://example.com/register<br>Learn More \| https://example.com/zero-trust |
| Channels | LinkedIn, Email |

## Pipeline Flow

```
INPUT → RETRIEVER → DRAFTER → CRITIC → VERIFIER → [LOOP?] → GENERATE_IMAGES → EXPORTER
                         ↑__________________________|
                         (loops if quality < 7 or unsupported claims, max 3x)
```

**Image Generation:** After content passes quality checks, marketing images are generated for each channel using Gemini 2.5 Flash (if `GEMINI_API_KEY` is set). Images are tailored to each channel's style (professional for LinkedIn, engaging for Facebook, etc.).

## Troubleshooting

**"insufficient_quota" error:**
- Check your OpenAI billing at platform.openai.com

**ChromaDB errors:**
- Delete `chroma_db/` folder and re-ingest

**Slow first run:**
- First ingestion downloads ~90MB embedding model (cached after)

## Optional: LangSmith Tracing

Add to `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=event-content-generator
```

Then view traces at: https://smith.langchain.com
