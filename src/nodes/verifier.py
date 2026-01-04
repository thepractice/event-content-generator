"""Verifier node - ensures every claim can be traced to a source."""

from __future__ import annotations

from datetime import datetime

from openai import OpenAI

from ..prompts import get_verifier_prompt
from ..schemas import Claim, ContentGeneratorState


def verify_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Verify that all factual claims in drafts have source citations.

    For each claim in each draft:
    1. Attempt to match the claim to a corpus source chunk
    2. If from corpus: mark as supported with source_chunk_id
    3. If from event form input: mark as supported with source "user_input"
    4. If neither: mark as unsupported (will trigger re-draft)

    Args:
        state: Current pipeline state with drafts and claims

    Returns:
        Updated state with verified claims
    """
    client = OpenAI()

    # Combine all source chunks for verification
    all_chunks = state.get("brand_chunks", []) + state.get("product_chunks", [])
    chunks_text = "\n\n".join(
        f"[{chunk['id']}]: {chunk['text']}" for chunk in all_chunks
    )

    # Build event context for user_input verification
    event_context = _build_event_context(state)

    # Track verification results
    verified_drafts = {}
    unsupported_claims = []

    for channel, draft in state.get("drafts", {}).items():
        # Extract claims from the draft content
        claims = _extract_claims(draft.body, chunks_text, event_context, client)

        # Update draft with verified claims
        verified_draft = draft.model_copy(update={"claims": claims})
        verified_drafts[channel] = verified_draft

        # Track unsupported claims
        for claim in claims:
            if not claim.is_supported:
                unsupported_claims.append(
                    {"channel": channel, "claim": claim.text}
                )

    # Log the verification action
    audit_entry = {
        "node": "verify",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "verified_claims",
        "details": {
            "total_claims": sum(
                len(d.claims) for d in verified_drafts.values()
            ),
            "unsupported_claims": len(unsupported_claims),
            "unsupported_details": unsupported_claims,
        },
    }

    return {
        **state,
        "drafts": verified_drafts,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }


def _build_event_context(state: ContentGeneratorState) -> str:
    """Build event form context for user_input verification."""
    parts = []

    if state.get("event_title"):
        parts.append(f"Event Title: {state['event_title']}")

    if state.get("event_description"):
        parts.append(f"Event Description: {state['event_description']}")

    if state.get("event_date"):
        parts.append(f"Event Date: {state['event_date']}")

    if state.get("target_audience"):
        parts.append(f"Target Audience: {state['target_audience']}")

    if state.get("key_messages"):
        messages = "\n".join(f"- {msg}" for msg in state["key_messages"])
        parts.append(f"Key Messages:\n{messages}")

    return "\n\n".join(parts)


def _extract_claims(
    content: str, chunks_text: str, event_context: str, client: OpenAI
) -> list[Claim]:
    """Extract factual claims from content and verify against sources.

    Args:
        content: The draft content to analyze
        chunks_text: Combined source chunks for verification
        event_context: Event form data provided by user
        client: OpenAI client for LLM calls

    Returns:
        List of Claim objects with verification status
    """
    import re

    prompt = get_verifier_prompt(
        content=content, sources=chunks_text, event_context=event_context
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.choices[0].message.content or ""
    claims = []

    # Parse claims from verifier response
    # Format: CLAIM: [text]\nSOURCE: [chunk_id or NONE]\nSUPPORTED: [true/false]
    claim_blocks = re.split(r'\n---\n|\n\n(?=CLAIM:)', response_text)

    for block in claim_blocks:
        claim_match = re.search(r'CLAIM:\s*(.+?)(?=\nSOURCE:|\n|$)', block, re.IGNORECASE | re.DOTALL)
        source_match = re.search(r'SOURCE:\s*(\S+)', block, re.IGNORECASE)
        supported_match = re.search(r'SUPPORTED:\s*(true|false)', block, re.IGNORECASE)

        if claim_match:
            claim_text = claim_match.group(1).strip()
            source_id = source_match.group(1).strip() if source_match else None
            is_supported = supported_match.group(1).lower() == 'true' if supported_match else False

            # Clean up source_id
            if source_id and source_id.upper() == 'NONE':
                source_id = None
                is_supported = False
            elif source_id and source_id.lower() == 'user_input':
                # Keep "user_input" as valid source
                is_supported = True

            if claim_text and len(claim_text) > 5:  # Filter out very short matches
                claims.append(Claim(
                    text=claim_text,
                    source_chunk_id=source_id,
                    is_supported=is_supported,
                ))

    return claims
