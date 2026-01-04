"""Exporter node - packages final outputs with metadata and audit trail."""

from __future__ import annotations

from datetime import datetime

from ..schemas import ContentGeneratorState


def export_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Package final content with metadata and audit trail.

    Creates the final output structure including:
    - All channel content
    - Quality scorecard
    - Claims table with source citations
    - Full audit log

    Args:
        state: Final pipeline state

    Returns:
        Updated state with final_output populated
    """
    # Build content output per channel
    content = {}
    for channel, draft in state.get("drafts", {}).items():
        content[channel] = {
            "headline": draft.headline,
            "body": draft.body,
            "cta": draft.cta,
            "subject_line": draft.subject_line,
        }

    # Build scorecard from critic feedback
    critic_feedback = state.get("critic_feedback")
    scorecard = {
        "brand_voice_score": (
            critic_feedback.brand_voice_score if critic_feedback else None
        ),
        "cta_clarity_score": (
            critic_feedback.cta_clarity_score if critic_feedback else None
        ),
        "iterations": state.get("iteration", 0),
        "passed": critic_feedback.passed if critic_feedback else False,
    }

    # Build claims table
    claims_table = []
    for channel, draft in state.get("drafts", {}).items():
        for claim in draft.claims:
            claims_table.append(
                {
                    "claim": claim.text,
                    "source": claim.source_chunk_id,
                    "is_supported": claim.is_supported,
                    "channel": channel,
                }
            )

    # Compile final output
    final_output = {
        "content": content,
        "images": state.get("images", {}),
        "relevant_urls": state.get("relevant_urls", []),
        "scorecard": scorecard,
        "claims_table": claims_table,
        "audit_log": {
            "run_id": _generate_run_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "input": {
                "event_title": state.get("event_title"),
                "event_description": state.get("event_description"),
                "event_date": state.get("event_date"),
                "target_audience": state.get("target_audience"),
                "key_messages": state.get("key_messages"),
                "channels": state.get("channels"),
            },
            "sources_retrieved": {
                "brand_chunks": len(state.get("brand_chunks", [])),
                "product_chunks": len(state.get("product_chunks", [])),
            },
            "iterations": state.get("audit_log", []),
        },
    }

    # Log the export action
    audit_entry = {
        "node": "export",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "exported_final_output",
        "details": {
            "channels_exported": list(content.keys()),
            "total_claims": len(claims_table),
        },
    }

    return {
        **state,
        "final_output": final_output,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _generate_run_id() -> str:
    """Generate a unique run ID for this pipeline execution."""
    import uuid

    return f"run_{uuid.uuid4().hex[:12]}"
