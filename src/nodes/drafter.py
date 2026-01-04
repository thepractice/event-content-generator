"""Drafter node - generates content for all selected channels."""

from __future__ import annotations

from datetime import datetime

from openai import OpenAI

from ..prompts import get_drafter_prompt
from ..schemas import ChannelDraft, Claim, ContentGeneratorState


def draft_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Generate content drafts for all selected channels.

    Uses OpenAI to generate channel-appropriate content, grounded in
    the retrieved brand voice and product documentation chunks.

    If this is a re-draft (iteration > 0), incorporates critic feedback.

    Args:
        state: Current pipeline state with retrieved chunks

    Returns:
        Updated state with drafts populated for each channel
    """
    client = OpenAI()
    iteration = state.get("iteration", 0) + 1

    # Build context from retrieved chunks
    brand_context = "\n\n".join(
        f"[{chunk['id']}]: {chunk['text']}" for chunk in state.get("brand_chunks", [])
    )
    product_context = "\n\n".join(
        f"[{chunk['id']}]: {chunk['text']}" for chunk in state.get("product_chunks", [])
    )

    # Get critic feedback if this is a re-draft
    critic_feedback = state.get("critic_feedback")
    feedback_text = ""
    if critic_feedback and iteration > 1:
        feedback_text = f"""
Previous feedback to address:
- Brand Voice Score: {critic_feedback.brand_voice_score}/10
- CTA Clarity Score: {critic_feedback.cta_clarity_score}/10
- Issues: {', '.join(critic_feedback.issues)}
- Suggested Fixes: {', '.join(critic_feedback.fixes)}
"""

    drafts: dict[str, ChannelDraft] = {}

    for channel in state["channels"]:
        prompt = get_drafter_prompt(
            channel=channel,
            event_title=state["event_title"],
            event_description=state["event_description"],
            event_date=state.get("event_date"),
            target_audience=state["target_audience"],
            key_messages=state["key_messages"],
            brand_context=brand_context,
            product_context=product_context,
            feedback=feedback_text,
            relevant_urls=state.get("relevant_urls", []),
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response into a ChannelDraft
        draft = _parse_draft_response(channel, response.choices[0].message.content)
        drafts[channel] = draft

    # Log the drafting action
    audit_entry = {
        "node": "draft",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "generated_drafts",
        "details": {
            "iteration": iteration,
            "channels": list(drafts.keys()),
            "had_feedback": bool(feedback_text),
        },
    }

    return {
        **state,
        "drafts": drafts,
        "iteration": iteration,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _parse_draft_response(channel: str, response_text: str) -> ChannelDraft:
    """Parse the LLM response into a structured ChannelDraft.

    Extracts HEADLINE, SUBJECT, BODY, CTA, and CLAIMS from the LLM response.

    Args:
        channel: The channel this draft is for
        response_text: Raw LLM response

    Returns:
        Parsed ChannelDraft object
    """
    import re

    headline = None
    subject_line = None
    body = ""
    cta = "Learn more"
    claims = []

    # Extract HEADLINE
    headline_match = re.search(r"HEADLINE:\s*(.+?)(?=\n(?:SUBJECT|BODY|CTA|CLAIMS)|$)", response_text, re.IGNORECASE | re.DOTALL)
    if headline_match:
        headline = headline_match.group(1).strip()
        # Clean up if it contains other sections
        if "\n" in headline:
            headline = headline.split("\n")[0].strip()

    # Extract SUBJECT (for email)
    subject_match = re.search(r"SUBJECT:\s*(.+?)(?=\n(?:BODY|CTA|CLAIMS)|$)", response_text, re.IGNORECASE | re.DOTALL)
    if subject_match:
        subject_line = subject_match.group(1).strip()
        if "\n" in subject_line:
            subject_line = subject_line.split("\n")[0].strip()

    # Extract BODY
    body_match = re.search(r"BODY:\s*(.+?)(?=\nCTA:|CLAIMS:|$)", response_text, re.IGNORECASE | re.DOTALL)
    if body_match:
        body = body_match.group(1).strip()
    else:
        # Fallback: use the whole response if no BODY marker found
        body = response_text

    # Extract CTA
    cta_match = re.search(r"CTA:\s*(.+?)(?=\nCLAIMS:|$)", response_text, re.IGNORECASE | re.DOTALL)
    if cta_match:
        cta = cta_match.group(1).strip()
        if "\n" in cta:
            cta = cta.split("\n")[0].strip()

    # Extract CLAIMS from dedicated section
    claims_match = re.search(r"CLAIMS:\s*(.+?)$", response_text, re.IGNORECASE | re.DOTALL)
    if claims_match:
        claims_text = claims_match.group(1).strip()
        # Parse individual claims
        claim_lines = [line.strip().lstrip("- ") for line in claims_text.split("\n") if line.strip()]
        for claim_line in claim_lines:
            # Extract source from claim line
            source_match = re.search(r"\[source:\s*(\w+)\]", claim_line)
            source_id = source_match.group(1) if source_match else None
            claim_text = re.sub(r"\[source:\s*\w+\]", "", claim_line).strip()
            if claim_text:
                claims.append(Claim(
                    text=claim_text,
                    source_chunk_id=source_id,
                    is_supported=source_id is not None,
                ))

    # Also extract inline citations from body text
    # Find sentences/phrases with [source: chunk_xxx] citations
    inline_citations = re.findall(r"([^.!?\n]*\[source:\s*(\w+)\][^.!?\n]*[.!?]?)", body)
    for citation_match in inline_citations:
        full_text, source_id = citation_match
        # Clean up the claim text
        claim_text = re.sub(r"\[source:\s*\w+\]", "", full_text).strip()
        claim_text = claim_text.strip(".,!? ")
        # Avoid duplicates
        existing_texts = [c.text.lower() for c in claims]
        if claim_text and claim_text.lower() not in existing_texts:
            claims.append(Claim(
                text=claim_text,
                source_chunk_id=source_id,
                is_supported=True,
            ))

    return ChannelDraft(
        channel=channel,
        headline=headline,
        body=body,
        cta=cta,
        subject_line=subject_line,
        claims=claims,
    )
