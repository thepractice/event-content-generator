"""All LLM prompts for the Event Content Generator pipeline."""

from typing import Dict, List, Optional

from .schemas import CHANNEL_CONFIGS


def get_drafter_prompt(
    channel: str,
    event_title: str,
    event_description: str,
    event_date: Optional[str],
    target_audience: str,
    key_messages: List[str],
    brand_context: str,
    product_context: str,
    feedback: str = "",
    relevant_urls: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generate the prompt for the drafter node.

    Args:
        channel: Marketing channel (linkedin, facebook, email, web)
        event_title: Name of the event
        event_description: What the event is about
        event_date: When the event occurs (optional)
        target_audience: Who this content is for
        key_messages: Key points to communicate
        brand_context: Retrieved brand voice examples
        product_context: Retrieved product documentation
        feedback: Critic feedback from previous iteration (if any)
        relevant_urls: Optional list of URLs to include in content

    Returns:
        Formatted prompt string
    """
    config = CHANNEL_CONFIGS.get(channel, {})

    channel_instructions = _get_channel_instructions(channel, config)

    messages_formatted = "\n".join(f"- {msg}" for msg in key_messages)

    prompt = f"""You are an expert marketing content writer. Generate {channel} content for an event promotion.

## Event Details
- **Title:** {event_title}
- **Description:** {event_description}
- **Date:** {event_date or "TBD"}
- **Target Audience:** {target_audience}

## Key Messages to Convey
{messages_formatted}

## Channel Requirements ({channel.upper()})
{channel_instructions}

## Brand Voice Examples (match this tone and style)
{brand_context or "No brand examples available - use professional marketing tone."}

## Product/Company Context (use for factual claims)
{product_context or "No product context available."}

## Relevant URLs (include naturally in CTAs and body where appropriate)
{format_urls_for_prompt(relevant_urls) if relevant_urls else "No URLs provided - use generic CTA language."}

{feedback}

## Instructions
1. Write compelling {channel} content that promotes this event
2. Match the brand voice from the examples above
3. Include a clear call-to-action
4. For ANY factual claim you make, note which source chunk it comes from using [source: chunk_id] format
5. If you cannot cite a source for a claim, do not make that claim
6. Stay within the character/word limits for this channel

## Output Format
Provide your response in this exact format:

HEADLINE: (if applicable for this channel)
SUBJECT: (for email only)
BODY:
[Your main content here with [source: chunk_id] citations for factual claims]

CTA: [Your call-to-action]

CLAIMS:
- Claim 1 [source: chunk_id]
- Claim 2 [source: chunk_id]
(List all factual claims with their sources)
"""
    return prompt


def _get_channel_instructions(channel: str, config: Dict) -> str:
    """Get channel-specific formatting instructions."""
    if channel == "linkedin":
        return f"""
- Maximum {config.get('max_length', 3000)} characters
- Tone: {config.get('tone', 'Professional, thought-leadership')}
- Required elements: {', '.join(config.get('required_elements', []))}
- Use 1-2 relevant hashtags at the end
- Open with a hook that grabs attention
- Include clear value proposition
"""
    elif channel == "facebook":
        return f"""
- Maximum {config.get('max_length', 500)} characters
- Tone: {config.get('tone', 'Conversational, engaging')}
- Required elements: {', '.join(config.get('required_elements', []))}
- Keep it short and punchy
- Use conversational language
- Include an emoji or two if appropriate
"""
    elif channel == "email":
        return f"""
- Subject line: Maximum {config.get('subject_max_length', 60)} characters
- Body: Maximum {config.get('body_max_words', 300)} words
- Tone: {config.get('tone', 'Direct, personalized')}
- Required elements: {', '.join(config.get('required_elements', []))}
- Write a compelling subject line
- Open with personalization if possible
- Keep paragraphs short (2-3 sentences)
"""
    elif channel == "web":
        return f"""
- Headline: Maximum {config.get('headline_max_words', 10)} words
- Hero paragraph: Maximum {config.get('hero_max_words', 50)} words
- Tone: {config.get('tone', 'SEO-friendly, benefit-driven')}
- Required elements: {', '.join(config.get('required_elements', []))}
- Focus on benefits, not features
- Use action-oriented language
- Optimize for scanning
"""
    return "Follow standard marketing best practices."


def get_critic_prompt(
    drafts: str,
    brand_context: str,
    channels: List[str],
) -> str:
    """Generate the prompt for the critic node.

    Args:
        drafts: All channel drafts formatted as text
        brand_context: Brand voice examples for comparison
        channels: List of channels being evaluated

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert marketing content critic. Evaluate the following drafts against quality criteria.

## Brand Voice Examples (the standard to match)
{brand_context or "No brand examples - evaluate for general marketing quality."}

## Drafts to Evaluate
{drafts}

## Evaluation Criteria

### Brand Voice Score (0-10)
- 0-3: Generic, off-brand, sounds like generic AI
- 4-6: Partially aligned with brand voice
- 7-10: Matches brand voice examples well

### CTA Clarity Score (0-10)
- 0-3: CTA missing, buried, or unclear
- 4-6: CTA present but weak or generic
- 7-10: CTA is clear, compelling, and prominent

### Length Compliance
- LinkedIn: Max 3000 characters
- Facebook: Max 500 characters
- Email subject: Max 60 characters
- Email body: Max 300 words
- Web headline: Max 10 words
- Web hero: Max 50 words

## Output Format
Provide your evaluation in this exact format:

BRAND_VOICE_SCORE: [0-10]
CTA_CLARITY_SCORE: [0-10]
LENGTH_OK: [true/false]

ISSUES:
- [List specific problems found]
- [One issue per line]

FIXES:
- [Actionable suggestion for each issue]
- [One fix per line]

PASSED: [true if brand_voice >= 7 AND cta_clarity >= 7 AND length_ok, else false]
"""


def get_verifier_prompt(content: str, sources: str, event_context: str = "") -> str:
    """Generate the prompt for the verifier node.

    Args:
        content: Draft content to verify
        sources: All available source chunks
        event_context: User-provided event form data

    Returns:
        Formatted prompt string
    """
    return f"""You are a fact-checker. Extract all factual claims from the content and verify each against the available sources.

## Content to Verify
{content}

## Source Type 1: Corpus Documents
IMPORTANT: Corpus sources are formatted as [chunk_id]: followed by the text content.
The chunk_id looks like "chunk_abc12345" (chunk_ followed by 8 characters).
Use the EXACT chunk_id when citing these sources.

{sources}

## Source Type 2: User-Provided Event Details
The following information was provided by the user in the event form.
Claims derived from this data should use SOURCE: user_input

{event_context}

## Instructions
1. Identify every factual claim in the content (statements that could be true or false)
2. For each claim, check BOTH source types:
   - If the claim matches corpus content: use the exact chunk_id (e.g., "chunk_abc12345")
   - If the claim matches user-provided event details: use "user_input"
   - If the claim matches neither: use "NONE"
3. A claim is SUPPORTED if it comes from either corpus OR user_input
4. A claim is UNSUPPORTED only if it matches neither source

Note: Opinions, calls-to-action, and general marketing statements are NOT factual claims.

## Example Output

CLAIM: The event is on January 15, 2026
SOURCE: user_input
SUPPORTED: true

---

CLAIM: Teams can reduce review time by 50%
SOURCE: chunk_ed531959
SUPPORTED: true

---

CLAIM: The platform was founded in 2019
SOURCE: NONE
SUPPORTED: false

---

## Your Output
Now analyze the content above. For each factual claim found:

CLAIM: [The factual statement]
SOURCE: [chunk_id, "user_input", or "NONE"]
SUPPORTED: [true/false]

---

(Repeat for each claim)

SUMMARY:
Total claims: [number]
Supported from corpus: [number]
Supported from user_input: [number]
Unsupported: [number]
"""


def get_image_prompt(
    channel: str,
    headline: str,
    event_title: str,
    target_audience: str,
) -> str:
    """Generate an optimal prompt for marketing image generation.

    Uses channel-specific styles and event context to create
    visually appealing header/banner images.

    Args:
        channel: Marketing channel (linkedin, facebook, email, web)
        headline: Content headline or event title
        event_title: Name of the event
        target_audience: Who the event is for

    Returns:
        Formatted image generation prompt
    """
    channel_styles = {
        "linkedin": "Professional corporate photography, clean modern design, business atmosphere",
        "facebook": "Vibrant engaging social media graphic, eye-catching colors, dynamic composition",
        "email": "Clean professional header image, minimalist design, elegant typography space",
        "web": "Hero banner photography, high-impact visual, cinematic quality",
    }

    style = channel_styles.get(channel, "Professional marketing visual")

    # Build a detailed, effective prompt
    return f"""{style}, representing a professional event titled "{event_title}" for {target_audience}, theme inspired by: {headline}, high-quality professional photography, no text or words in the image, 16:9 aspect ratio, modern aesthetic, soft professional lighting, corporate color palette"""


def format_urls_for_prompt(urls: List[Dict[str, str]]) -> str:
    """Format relevant URLs for inclusion in drafter prompt.

    Args:
        urls: List of {"label": str, "url": str} dicts

    Returns:
        Formatted string for prompt inclusion
    """
    if not urls:
        return "No URLs provided"

    lines = []
    for url_info in urls:
        label = url_info.get("label", "Link")
        url = url_info.get("url", "")
        lines.append(f"- {label}: {url}")

    return "\n".join(lines)
