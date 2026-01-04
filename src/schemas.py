"""Pydantic models and state schema for the Event Content Generator."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A factual claim that must be traced to a source document."""

    text: str = Field(description="The factual statement made in the content")
    source_chunk_id: Optional[str] = Field(
        default=None, description="ID of the supporting source chunk"
    )
    is_supported: bool = Field(
        default=False, description="Whether this claim has a verified source"
    )


class ChannelDraft(BaseModel):
    """Generated content for a specific marketing channel."""

    channel: Literal["linkedin", "facebook", "email", "web"] = Field(
        description="The marketing channel this draft is for"
    )
    headline: Optional[str] = Field(
        default=None, description="Headline (not all channels need this)"
    )
    body: str = Field(description="Main content body")
    cta: str = Field(description="Call-to-action text")
    subject_line: Optional[str] = Field(
        default=None, description="Email subject line (email channel only)"
    )
    claims: List[Claim] = Field(
        default_factory=list, description="Factual claims made in this draft"
    )


class CriticFeedback(BaseModel):
    """Quality evaluation feedback from the critic node."""

    brand_voice_score: int = Field(
        ge=0, le=10, description="How well content matches brand voice (0-10)"
    )
    cta_clarity_score: int = Field(
        ge=0, le=10, description="How clear and compelling the CTA is (0-10)"
    )
    length_ok: bool = Field(
        description="Whether content meets length requirements for all channels"
    )
    issues: List[str] = Field(
        default_factory=list, description="Specific problems found in the content"
    )
    fixes: List[str] = Field(
        default_factory=list, description="Actionable suggestions for improvement"
    )
    passed: bool = Field(
        description="True if all scores >= 7 and length_ok is True"
    )


class SourceChunk(BaseModel):
    """A chunk of text from the source corpus."""

    id: str = Field(description="Unique identifier for this chunk")
    text: str = Field(description="The chunk content")
    source: Optional[str] = Field(
        default=None, description="Original document this chunk came from"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the chunk"
    )


class AuditLogEntry(BaseModel):
    """An entry in the audit log tracking pipeline execution."""

    node: str = Field(description="Name of the node that generated this entry")
    timestamp: str = Field(description="ISO format timestamp")
    action: str = Field(description="What action was taken")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the action"
    )


class ContentGeneratorState(TypedDict, total=False):
    """State schema for the LangGraph pipeline.

    This TypedDict defines all the data that flows through the pipeline,
    from initial input through retrieval, drafting, critique, and export.
    """

    # Input fields
    event_title: str
    event_description: str
    event_date: Optional[str]
    target_audience: str
    key_messages: List[str]
    channels: List[str]
    relevant_urls: List[Dict[str, str]]  # [{"label": "Register", "url": "https://..."}]

    # Retrieved context from RAG
    brand_chunks: List[Dict[str, Any]]  # List of {"id": str, "text": str, ...}
    product_chunks: List[Dict[str, Any]]

    # Draft content (evolves through iterations)
    drafts: Dict[str, ChannelDraft]

    # Quality tracking
    critic_feedback: Optional[CriticFeedback]
    iteration: int

    # Output
    final_output: Optional[Dict[str, Any]]
    audit_log: List[Dict[str, Any]]
    images: Dict[str, Any]  # {"linkedin": PIL.Image or bytes, ...}


# Channel configuration constants
CHANNEL_CONFIGS = {
    "linkedin": {
        "max_length": 3000,
        "tone": "Professional, thought-leadership",
        "required_elements": ["Hook", "value prop", "CTA", "hashtags"],
    },
    "facebook": {
        "max_length": 500,
        "tone": "Conversational, engaging",
        "required_elements": ["Hook", "benefit", "CTA"],
    },
    "email": {
        "subject_max_length": 60,
        "body_max_words": 300,
        "tone": "Direct, personalized",
        "required_elements": ["Subject", "preheader", "body", "CTA"],
    },
    "web": {
        "headline_max_words": 10,
        "hero_max_words": 50,
        "tone": "SEO-friendly, benefit-driven",
        "required_elements": ["Headline", "subhead", "hero paragraph"],
    },
}
