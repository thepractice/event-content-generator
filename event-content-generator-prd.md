# Event Content Generator - Product Requirements Document

## Executive Summary

Build a LangGraph-powered pipeline that transforms raw event briefs into brand-safe, multi-channel marketing content with citation-backed claims and quality verification loops.

**Target Use Case:** Internal marketing tool for generating event promotional content (LinkedIn posts, Facebook posts, email invites, landing page copy).

**Core Differentiator:** Every factual claim is traced to a source document. No citation = no claim.

---

## Problem Statement

Enterprise content generation fails for two reasons:

1. **Hallucinations** — AI makes claims the company can't defend
2. **Brand drift** — Output sounds like generic ChatGPT, not the company

This tool solves both problems visibly and traceably.

---

## User Personas

**Primary User:** Marketing team member who needs to quickly generate promotional content for an upcoming event (webinar, conference, product launch).

**Secondary User:** Marketing manager who needs to review and approve AI-generated content before publishing.

---

## Functional Requirements

### Inputs

| Field | Required | Description |
|-------|----------|-------------|
| Event Title | Yes | Name of the event |
| Event Description | Yes | What the event is about (can be messy bullet points) |
| Event Date | No | When the event occurs |
| Target Audience | Yes | Who this content is for (e.g., "Enterprise CTOs", "Developers") |
| Key Messages | Yes | 2-5 bullet points of what to communicate |
| Channels | Yes | Multi-select: LinkedIn, Facebook, Email, Web |
| Brand Source | No | Which brand voice to use (default: configured corpus) |

### Outputs

For each selected channel:

| Output | Description |
|--------|-------------|
| Final Copy | Ready-to-paste content |
| Variations | 2-3 alternatives where appropriate |
| Claims Table | Every factual claim with source citation |
| Quality Scorecard | Brand voice (0-10), CTA clarity (0-10), length compliance |
| Audit Log | Full trace of inputs, retrievals, iterations, final output |

### Output Formats

- **Display:** Streamlit UI with tabs per channel
- **Export:** JSON bundle with all outputs + audit log
- **Trace:** LangSmith trace URL for debugging

---

## Technical Architecture

### Pipeline Overview

```
INPUT → RETRIEVER → DRAFTER → CRITIC → VERIFIER → [LOOP?] → EXPORTER
```

### Node Specifications

#### 1. Retriever Node

**Purpose:** Pull relevant brand voice examples and product context from vector store.

**Inputs:**
- Event description
- Target audience
- Key messages

**Outputs:**
- `brand_chunks`: List of {id, text} from brand voice corpus
- `product_chunks`: List of {id, text} from product documentation

**Implementation:**
- Vector store: ChromaDB (local)
- Embeddings: `text-embedding-3-small`
- Top-k: 5-10 chunks per query
- Each chunk must have a unique ID for citation tracking

#### 2. Drafter Node

**Purpose:** Generate content for all selected channels.

**Inputs:**
- Original user input (event details, audience, messages)
- Retrieved brand chunks
- Retrieved product chunks
- Critic feedback (if iterating)

**Outputs:**
- `drafts`: Dict mapping channel → ChannelDraft

**ChannelDraft Schema:**
```python
class ChannelDraft(BaseModel):
    channel: Literal["linkedin", "facebook", "email", "web"]
    headline: str | None  # Not all channels need headlines
    body: str
    cta: str
    subject_line: str | None  # Email only
    claims: list[Claim]  # Inline claim annotations
```

**Claim Schema:**
```python
class Claim(BaseModel):
    text: str  # The factual statement
    source_chunk_id: str | None  # ID of supporting chunk
    is_supported: bool  # Whether source was found
```

**Channel-Specific Requirements:**

| Channel | Max Length | Tone | Required Elements |
|---------|-----------|------|-------------------|
| LinkedIn | 3000 chars | Professional, thought-leadership | Hook, value prop, CTA, hashtags |
| Facebook | 500 chars | Conversational, engaging | Hook, benefit, CTA |
| Email | Subject: 60 chars, Body: 300 words | Direct, personalized | Subject, preheader, body, CTA |
| Web | Headline: 10 words, Hero: 50 words | SEO-friendly, benefit-driven | Headline, subhead, hero paragraph |

#### 3. Critic Node

**Purpose:** Evaluate drafts against quality rubric and provide actionable feedback.

**Inputs:**
- All channel drafts
- Brand chunks (for comparison)

**Outputs:**
```python
class CriticFeedback(BaseModel):
    brand_voice_score: int  # 0-10
    cta_clarity_score: int  # 0-10
    length_ok: bool
    issues: list[str]  # Specific problems found
    fixes: list[str]  # Actionable suggestions
    passed: bool  # True if all scores >= 7 and length_ok
```

**Scoring Criteria:**

| Criterion | Score 0-3 | Score 4-6 | Score 7-10 |
|-----------|-----------|-----------|------------|
| Brand Voice | Generic/off-brand | Partially aligned | Matches examples |
| CTA Clarity | Missing/buried | Present but weak | Clear, compelling, prominent |

#### 4. Verifier Node

**Purpose:** Ensure every factual claim can be traced to a source document.

**Inputs:**
- All channel drafts with claim annotations
- Retrieved source chunks

**Process:**
1. Extract all claims from drafts
2. For each claim, attempt to match to a source chunk
3. If match found: mark as supported, record source_chunk_id
4. If no match: mark as unsupported

**Output:**
- Updated drafts with verification status on each claim
- List of unsupported claims

**HARD RULE:** Any claim without a source citation must be removed or rewritten as non-factual.

#### 5. Conditional Router

**Logic:**
```python
def should_continue(state):
    if state["iteration"] >= 3:
        return "export"  # Max iterations reached
    if not state["critic_feedback"]["passed"]:
        return "draft"  # Quality issues, try again
    if any(not c.is_supported for c in all_claims(state)):
        return "draft"  # Unsupported claims, try again
    return "export"  # All good
```

#### 6. Exporter Node

**Purpose:** Package final outputs with metadata and audit trail.

**Outputs:**
```python
{
    "content": {
        "linkedin": {...},
        "facebook": {...},
        "email": {...},
        "web": {...}
    },
    "scorecard": {
        "brand_voice_score": 8,
        "cta_clarity_score": 9,
        "iterations": 2
    },
    "claims_table": [
        {"claim": "...", "source": "chunk_id", "channel": "linkedin"},
        ...
    ],
    "audit_log": {
        "run_id": "...",
        "timestamp": "...",
        "input": {...},
        "sources_retrieved": [...],
        "iterations": [...]
    }
}
```

---

## State Schema

```python
from typing import TypedDict, Literal
from pydantic import BaseModel

class Claim(BaseModel):
    text: str
    source_chunk_id: str | None
    is_supported: bool

class ChannelDraft(BaseModel):
    channel: Literal["linkedin", "facebook", "email", "web"]
    headline: str | None
    body: str
    cta: str
    subject_line: str | None
    claims: list[Claim]

class CriticFeedback(BaseModel):
    brand_voice_score: int
    cta_clarity_score: int
    length_ok: bool
    issues: list[str]
    fixes: list[str]
    passed: bool

class ContentGeneratorState(TypedDict):
    # Input
    event_title: str
    event_description: str
    event_date: str | None
    target_audience: str
    key_messages: list[str]
    channels: list[str]
    
    # Retrieved context
    brand_chunks: list[dict]  # {"id": "...", "text": "..."}
    product_chunks: list[dict]
    
    # Drafts (evolve through iterations)
    drafts: dict[str, ChannelDraft]
    
    # Quality tracking
    critic_feedback: CriticFeedback | None
    iteration: int
    
    # Output
    final_output: dict | None
    audit_log: list[dict]
```

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Orchestration | LangGraph | Target team's stack; best for stateful workflows with loops |
| LLM | Claude 3.5 Sonnet | Best reasoning for critique/verification |
| Embeddings | OpenAI text-embedding-3-small | Simple, good quality, cheap |
| Vector Store | ChromaDB (local) | Zero config, free, sufficient for demo |
| UI | Streamlit | Fast to build, professional enough for demo |
| Tracing | LangSmith | Essential for interview demos, free tier |
| Deployment | Hugging Face Spaces | Free, shareable URL |

---

## UI Requirements

### Input Form

- Event Title (text input, required)
- Event Description (text area, required)
- Event Date (date picker, optional)
- Target Audience (text input, required)
- Key Messages (text area, one per line, required)
- Channels (multi-select checkboxes, at least one required)
- Generate button

### Output Display

- **Tabs:** One tab per channel
- **Content:** Final copy with copy-to-clipboard button
- **Scorecard:** Visual display of quality scores
- **Claims Table:** Expandable table showing claim → source mappings
- **Audit Log:** Expandable JSON view
- **Trace Link:** Link to LangSmith trace (if available)

---

## RAG Corpus Requirements

### Initial Corpus (for development)

Use personal project docs (SocialPlaybook.ai or Wane):
- 10-15 documents
- Mix of: brand voice guidelines, product descriptions, sample marketing copy
- Each document chunked into ~500 token segments
- Each chunk gets unique ID

### Interview Demo Corpus

Before interviews, add:
- 5-10 public Acme Identity blog posts
- Acme Identity brand voice examples (scrape from acme-identity.com)

---

## Success Criteria

### Functional
- [ ] Generates coherent content for all 4 channels
- [ ] Critique loop iterates when quality is low
- [ ] Unsupported claims are removed
- [ ] Audit log captures full trace

### Quality
- [ ] Brand voice score ≥ 7 on final output
- [ ] No hallucinated claims in final output
- [ ] Content is actually usable (not generic fluff)

### Demo
- [ ] Deployed to shareable URL
- [ ] LangSmith traces visible
- [ ] Can demo full flow in < 5 minutes

---

## Out of Scope (V1)

- Real-time social media trends
- Image generation prompts
- Human-in-the-loop approval gates
- Multiple brand profile switching
- Parallel drafter execution (sequential is fine)
- Authentication/multi-user

---

## Build Timeline (7 Days)

| Day | Focus | Deliverables |
|-----|-------|--------------|
| 1 | Foundation | Project structure, schemas, empty graph that compiles |
| 2 | Retriever | Chroma setup, corpus ingestion, retrieval working |
| 3 | Drafter | Content generation for all channels, structured output |
| 4 | Critic | Quality scoring, actionable feedback |
| 5 | Verifier | Claim extraction, source matching, loop wiring |
| 6 | UI | Streamlit interface, all outputs displayed |
| 7 | Deploy | HF Spaces deployment, LangSmith traces, demo prep |

---

## Interview Talking Points

> "After we talked, I built a working prototype of the event content generator. It's a LangGraph pipeline with three key features:
>
> First, RAG grounding for brand voice—it retrieves examples from a corpus and matches the company's tone.
>
> Second, a critique loop that scores output against a rubric and iterates until it passes.
>
> Third—and this is the important one—a claim verifier. Every factual statement in the output gets matched to a source. If it can't be cited, it gets removed. No hallucinations.
>
> Here's the demo. Here's the trace. Here's the audit log. This is what I'd build for our internal platform."

---

## Appendix: Example Input/Output

### Example Input

```json
{
  "event_title": "Zero Trust Security Webinar",
  "event_description": "Join us for a deep dive into implementing Zero Trust architecture. Learn from Acme Identity security experts about identity-first security approaches.",
  "event_date": "2025-01-15",
  "target_audience": "Enterprise Security Leaders and CTOs",
  "key_messages": [
    "Zero Trust is identity-centric",
    "Traditional perimeter security is obsolete",
    "Acme Identity provides the foundation for Zero Trust"
  ],
  "channels": ["linkedin", "email"]
}
```

### Example Output (LinkedIn)

```
🔐 The perimeter is dead. Identity is the new security boundary.

Join our upcoming webinar where Acme Identity security experts break down:
→ Why Zero Trust starts with identity [source: chunk_12]
→ How to move beyond legacy perimeter models [source: chunk_7]
→ Real implementation strategies that work [source: chunk_15]

📅 January 15, 2025

Perfect for security leaders ready to modernize their approach.

Register now: [link]

#ZeroTrust #IdentitySecurity #CISO
```

### Example Claims Table

| Claim | Source Chunk | Channel |
|-------|--------------|---------|
| "Zero Trust starts with identity" | chunk_12 | linkedin |
| "move beyond legacy perimeter models" | chunk_7 | linkedin |
| "Real implementation strategies" | chunk_15 | linkedin |