"""Critic node - evaluates drafts against quality rubric."""

from __future__ import annotations

from datetime import datetime

from openai import OpenAI

from ..prompts import get_critic_prompt
from ..schemas import ContentGeneratorState, CriticFeedback


def critic_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Evaluate all channel drafts against quality criteria.

    Scores each draft on:
    - Brand voice alignment (0-10)
    - CTA clarity (0-10)
    - Length compliance

    Provides actionable feedback for improvement if scores are below threshold.

    Args:
        state: Current pipeline state with drafts

    Returns:
        Updated state with critic_feedback populated
    """
    client = OpenAI()

    # Compile all drafts for evaluation
    drafts_text = ""
    for channel, draft in state.get("drafts", {}).items():
        drafts_text += f"\n\n=== {channel.upper()} ===\n"
        if draft.headline:
            drafts_text += f"Headline: {draft.headline}\n"
        if draft.subject_line:
            drafts_text += f"Subject: {draft.subject_line}\n"
        drafts_text += f"Body: {draft.body}\n"
        drafts_text += f"CTA: {draft.cta}\n"

    # Build brand context for comparison
    brand_context = "\n\n".join(
        f"[{chunk['id']}]: {chunk['text']}" for chunk in state.get("brand_chunks", [])
    )

    prompt = get_critic_prompt(
        drafts=drafts_text,
        brand_context=brand_context,
        channels=state["channels"],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse the response into CriticFeedback
    feedback = _parse_critic_response(response.choices[0].message.content)

    # Log the critique action
    audit_entry = {
        "node": "critic",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "evaluated_drafts",
        "details": {
            "brand_voice_score": feedback.brand_voice_score,
            "cta_clarity_score": feedback.cta_clarity_score,
            "passed": feedback.passed,
            "issues_count": len(feedback.issues),
        },
    }

    return {
        **state,
        "critic_feedback": feedback,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _parse_critic_response(response_text: str) -> CriticFeedback:
    """Parse the LLM response into structured CriticFeedback.

    TODO: Implement proper structured output parsing.
    For now, returns a placeholder that passes.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed CriticFeedback object
    """
    # Placeholder implementation - will be enhanced with structured output
    # Default to passing for now
    return CriticFeedback(
        brand_voice_score=8,
        cta_clarity_score=8,
        length_ok=True,
        issues=[],
        fixes=[],
        passed=True,
    )
