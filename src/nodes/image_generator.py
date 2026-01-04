"""Image generator node - generates marketing images using Gemini."""

from __future__ import annotations

import os
from datetime import datetime

from google import genai

from ..prompts import get_image_prompt
from ..schemas import ContentGeneratorState


def generate_images_node(state: ContentGeneratorState) -> ContentGeneratorState:
    """Generate marketing images for each channel using Gemini image generation.

    Creates visually appealing header/banner images tailored to each
    marketing channel based on the event details and generated content.

    Args:
        state: Current pipeline state with finalized drafts

    Returns:
        Updated state with generated images
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        # Skip image generation if no API key
        audit_entry = {
            "node": "generate_images",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "skipped",
            "details": {"reason": "No GEMINI_API_KEY found in environment"},
        }
        return {
            **state,
            "images": {},
            "audit_log": state.get("audit_log", []) + [audit_entry],
        }

    client = genai.Client(api_key=api_key)
    images = {}
    generated_count = 0
    errors = []

    for channel, draft in state.get("drafts", {}).items():
        try:
            # Build channel-specific prompt
            prompt = get_image_prompt(
                channel=channel,
                headline=draft.headline or state.get("event_title", ""),
                event_title=state.get("event_title", ""),
                target_audience=state.get("target_audience", ""),
            )

            # Generate image using Gemini 2.5 Flash
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
            )

            # Extract image from response parts
            for part in response.parts:
                if part.inline_data is not None:
                    # Store raw image bytes for easy display
                    images[channel] = part.inline_data.data
                    generated_count += 1
                    break

        except Exception as e:
            errors.append({"channel": channel, "error": str(e)})

    # Log the generation action
    audit_entry = {
        "node": "generate_images",
        "timestamp": datetime.utcnow().isoformat(),
        "action": "generated_images",
        "details": {
            "channels_requested": list(state.get("drafts", {}).keys()),
            "images_generated": generated_count,
            "errors": errors,
        },
    }

    return {
        **state,
        "images": images,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
