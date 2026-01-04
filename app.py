"""Streamlit UI for the Event Content Generator."""

from __future__ import annotations

import glob
import json
import re
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


def strip_citations(text: str) -> str:
    """Remove citation markers from text for clean copy.

    Removes patterns like [source: chunk_xxx] from the text.
    """
    if not text:
        return text
    # Remove [source: chunk_xxx] patterns
    cleaned = re.sub(r'\s*\[source:\s*\w+\]\.?', '', text)
    # Clean up any double spaces left behind
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


def get_clean_copy_text(channel_content: dict, channel: str) -> str:
    """Generate clean, copy-ready text without citations."""
    parts = []

    if channel_content.get("headline"):
        parts.append(channel_content["headline"])
        parts.append("")  # Empty line

    if channel_content.get("subject_line"):
        parts.append(f"Subject: {channel_content['subject_line']}")
        parts.append("")

    if channel_content.get("body"):
        parts.append(strip_citations(channel_content["body"]))
        parts.append("")

    if channel_content.get("cta"):
        parts.append(strip_citations(channel_content["cta"]))

    return "\n".join(parts).strip()


def parse_urls(text: str) -> list:
    """Parse URLs from text input (format: Label | URL, one per line)."""
    if not text or not text.strip():
        return []

    urls = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            label, url = line.split("|", 1)
            urls.append({"label": label.strip(), "url": url.strip()})
        elif line.startswith("http"):
            urls.append({"label": "Link", "url": line.strip()})
    return urls

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Event Content Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .score-high { background: #d4edda; color: #155724; }
    .score-mid { background: #fff3cd; color: #856404; }
    .score-low { background: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""
    st.title("📝 Event Content Generator")
    st.markdown(
        "Transform event briefs into brand-safe, citation-backed marketing content."
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Corpus management
        st.subheader("📚 Corpus")
        st.caption("Documents are persisted. Only re-index if you've added new files to `/corpus`.")
        if st.button("🔄 Re-index Corpus", help="Re-process documents in the corpus folder"):
            with st.spinner("Re-indexing documents..."):
                try:
                    from src.rag import ingest_documents

                    result = ingest_documents(force_reingest=True)
                    st.success(f"Re-indexed: {result}")
                except Exception as e:
                    st.error(f"Re-indexing failed: {e}")

        # Corpus document viewer
        st.subheader("📄 Documents")
        corpus_files = sorted(glob.glob("corpus/*.md"))
        if corpus_files:
            for file_path in corpus_files:
                filename = Path(file_path).name
                with st.expander(filename, expanded=False):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.markdown(content)
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
        else:
            st.caption("No documents found in /corpus")

        st.divider()

        # LangSmith trace link placeholder
        st.subheader("🔍 Tracing")
        if "trace_url" in st.session_state:
            st.markdown(f"[View LangSmith Trace]({st.session_state.trace_url})")
        else:
            st.caption("Trace URL will appear after generation")

    # Sample event data for testing
    import random
    SAMPLE_EVENTS = [
        {
            "title": "Zero Trust Security Webinar",
            "description": "Join our security experts for a deep dive into Zero Trust architecture. Learn how identity-first security protects your organization from modern threats.",
            "audience": "CISOs, Security Architects, and IT Leaders",
            "messages": "• Identity is the new security perimeter\n• Zero Trust eliminates implicit trust\n• Protect against modern cyber threats",
            "urls": "Register | https://example.com/register/zero-trust\nLearn More | https://example.com/events/zero-trust",
        },
        {
            "title": "AI Innovation Summit 2026",
            "description": "Explore cutting-edge AI technologies shaping the future. Features keynote speakers, hands-on workshops, and networking with industry leaders.",
            "audience": "Tech leaders, CTOs, data scientists, and AI enthusiasts",
            "messages": "• Discover breakthrough AI technologies\n• Network with industry innovators\n• Gain practical skills through workshops",
            "urls": "Register | https://example.com/register/ai-summit\nAgenda | https://example.com/events/ai-summit/agenda",
        },
        {
            "title": "Cloud Migration Masterclass",
            "description": "Learn proven strategies for migrating enterprise workloads to the cloud. Our experts share best practices and common pitfalls to avoid.",
            "audience": "IT Directors, Cloud Architects, and DevOps Engineers",
            "messages": "• Reduce infrastructure costs by 40%\n• Improve scalability and reliability\n• Accelerate digital transformation",
            "urls": "Register | https://example.com/register/cloud-masterclass\nResources | https://example.com/cloud-migration-guide",
        },
        {
            "title": "Developer Experience Conference",
            "description": "A full-day conference focused on improving developer productivity and satisfaction. Learn about modern tooling, workflows, and team culture.",
            "audience": "Engineering managers, developers, and DevEx practitioners",
            "messages": "• Boost developer productivity\n• Reduce cognitive load and friction\n• Build a culture of engineering excellence",
            "urls": "Register | https://example.com/register/devex-conf\nSpeakers | https://example.com/devex-conf/speakers",
        },
    ]

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Event Details")

        # Test data button (outside form)
        if st.button("🎲 Fill Random Sample Data", help="Fill form with random test data"):
            sample = random.choice(SAMPLE_EVENTS)
            st.session_state["sample_title"] = sample["title"]
            st.session_state["sample_description"] = sample["description"]
            st.session_state["sample_audience"] = sample["audience"]
            st.session_state["sample_messages"] = sample["messages"]
            st.session_state["sample_urls"] = sample.get("urls", "")
            st.rerun()

        # Input form
        with st.form("event_form"):
            event_title = st.text_input(
                "Event Title *",
                value=st.session_state.get("sample_title", ""),
                placeholder="e.g., Zero Trust Security Webinar",
            )

            event_description = st.text_area(
                "Event Description *",
                value=st.session_state.get("sample_description", ""),
                placeholder="What is this event about? Can be bullet points or rough notes.",
                height=150,
            )

            event_date = st.date_input(
                "Event Date",
                value=None,
                min_value=date.today(),
            )

            target_audience = st.text_input(
                "Target Audience *",
                value=st.session_state.get("sample_audience", ""),
                placeholder="e.g., Enterprise Security Leaders and CTOs",
            )

            key_messages = st.text_area(
                "Key Messages * (one per line)",
                value=st.session_state.get("sample_messages", ""),
                placeholder="• First key message\n• Second key message\n• Third key message",
                height=100,
            )

            relevant_urls_text = st.text_area(
                "Relevant URLs (optional, one per line: Label | URL)",
                value=st.session_state.get("sample_urls", ""),
                placeholder="Registration | https://example.com/register\nLearn More | https://example.com/info",
                height=80,
            )

            st.markdown("**Select Channels** *")
            col_a, col_b = st.columns(2)
            with col_a:
                linkedin = st.checkbox("LinkedIn", value=True)
                facebook = st.checkbox("Facebook")
            with col_b:
                email = st.checkbox("Email", value=True)
                web = st.checkbox("Web / Landing Page")

            submitted = st.form_submit_button(
                "Generate Content",
                type="primary",
                use_container_width=True,
            )

    # Process form submission
    if submitted:
        # Validation
        errors = []
        if not event_title:
            errors.append("Event Title is required")
        if not event_description:
            errors.append("Event Description is required")
        if not target_audience:
            errors.append("Target Audience is required")
        if not key_messages:
            errors.append("Key Messages are required")
        if not any([linkedin, facebook, email, web]):
            errors.append("Select at least one channel")

        if errors:
            for error in errors:
                st.error(error)
        else:
            # Build channel list
            channels = []
            if linkedin:
                channels.append("linkedin")
            if facebook:
                channels.append("facebook")
            if email:
                channels.append("email")
            if web:
                channels.append("web")

            # Parse key messages
            messages = [
                m.strip().lstrip("•-*")
                for m in key_messages.strip().split("\n")
                if m.strip()
            ]

            # Generate content with progress indicator
            try:
                from src.graph import run_pipeline

                # Initialize step log for this run
                step_log = []

                # Create a status container for progress
                status_container = st.status("🚀 Starting content generation...", expanded=True)

                def update_progress(step_name: str, step_info: dict, iteration: int):
                    """Callback to update progress display with rich details."""
                    description = step_info.get("description", "Processing...")
                    details = step_info.get("details", [])
                    metrics = step_info.get("metrics", {})

                    # Determine if this is a completion callback
                    is_done = step_name.endswith("_done")
                    base_step = step_name.replace("_done", "")

                    # Format iteration text
                    iter_text = f" (iteration {iteration + 1})" if iteration > 0 else ""

                    # Update the status label
                    if is_done:
                        status_container.update(label=f"✓ {description}")
                    else:
                        status_container.update(label=f"⏳ {description}{iter_text}")

                    # Show step details
                    step_display = base_step.upper()
                    status_container.write(f"**{step_display}**{iter_text}: {description}")

                    # Show bullet points for details
                    for detail in details:
                        status_container.write(f"  • {detail}")

                    # Log this step for persistent display
                    step_log.append({
                        "step": base_step,
                        "iteration": iteration,
                        "description": description,
                        "details": details,
                        "metrics": metrics,
                        "is_done": is_done,
                    })

                result = run_pipeline(
                    event_title=event_title,
                    event_description=event_description,
                    target_audience=target_audience,
                    key_messages=messages,
                    channels=channels,
                    event_date=str(event_date) if event_date else None,
                    relevant_urls=parse_urls(relevant_urls_text),
                    on_step=update_progress,
                )

                status_container.update(label="✅ Content generated successfully!", state="complete")
                st.session_state.result = result
                st.session_state.step_log = step_log

            except Exception as e:
                status_container.update(label="❌ Generation failed", state="error")
                st.error(f"Generation failed: {e}")
                st.exception(e)

    # Display results
    with col2:
        st.header("Generated Content")

        if "result" in st.session_state and st.session_state.result:
            result = st.session_state.result
            content = result.get("content", {})
            scorecard = result.get("scorecard", {})
            claims_table = result.get("claims_table", [])
            audit_log = result.get("audit_log", {})
            images = result.get("images", {})

            # Scorecard
            st.subheader("Quality Scorecard")
            score_cols = st.columns(3)

            with score_cols[0]:
                brand_score = scorecard.get("brand_voice_score", 0)
                score_class = (
                    "score-high"
                    if brand_score >= 7
                    else "score-mid" if brand_score >= 4 else "score-low"
                )
                st.markdown(
                    f"**Brand Voice:** <span class='{score_class}'>{brand_score}/10</span>",
                    unsafe_allow_html=True,
                )

            with score_cols[1]:
                cta_score = scorecard.get("cta_clarity_score", 0)
                score_class = (
                    "score-high"
                    if cta_score >= 7
                    else "score-mid" if cta_score >= 4 else "score-low"
                )
                st.markdown(
                    f"**CTA Clarity:** <span class='{score_class}'>{cta_score}/10</span>",
                    unsafe_allow_html=True,
                )

            with score_cols[2]:
                iterations = scorecard.get("iterations", 0)
                st.markdown(f"**Iterations:** {iterations}")

            st.divider()

            # Content tabs
            if content:
                tabs = st.tabs([ch.upper() for ch in content.keys()])

                for tab, (channel, channel_content) in zip(tabs, content.items()):
                    with tab:
                        # Get clean content for display
                        clean_body = strip_citations(channel_content.get("body", ""))
                        clean_cta = strip_citations(channel_content.get("cta", ""))

                        if channel_content.get("headline"):
                            st.markdown(f"**Headline:** {channel_content['headline']}")

                        if channel_content.get("subject_line"):
                            st.markdown(
                                f"**Subject:** {channel_content['subject_line']}"
                            )

                        st.markdown("**Body:**")
                        st.text_area(
                            "Content",
                            value=clean_body,
                            height=200,
                            key=f"body_{channel}",
                            label_visibility="collapsed",
                        )

                        # Character count
                        char_count = len(clean_body)
                        max_chars = {"linkedin": 3000, "facebook": 500, "email": 1500, "web": 500}.get(channel, 3000)
                        count_color = "green" if char_count <= max_chars else "red"
                        st.caption(f":{count_color}[{char_count} / {max_chars} characters]")

                        st.markdown(f"**CTA:** {clean_cta}")

                        # Copy-ready content in a code block
                        clean_copy = get_clean_copy_text(channel_content, channel)
                        with st.expander("📋 Copy-Ready Content", expanded=False):
                            st.code(clean_copy, language=None)
                            st.caption("Select all and copy the text above")

                        # Display generated image for this channel
                        if channel in images and images[channel]:
                            st.divider()
                            st.markdown("**Generated Image:**")
                            try:
                                # Images are stored as raw bytes
                                st.image(
                                    images[channel],
                                    caption=f"{channel.upper()} Header Image",
                                    use_container_width=True,
                                )
                            except Exception as img_err:
                                st.warning(f"Could not display image: {img_err}")

            # Claims table
            with st.expander("📋 Claims Table", expanded=False):
                if claims_table:
                    st.dataframe(claims_table, use_container_width=True)
                else:
                    st.caption("No factual claims extracted")

            # Pipeline execution log
            with st.expander("🔄 Pipeline Steps", expanded=False):
                if "step_log" in st.session_state and st.session_state.step_log:
                    for entry in st.session_state.step_log:
                        if entry.get("is_done"):
                            step = entry["step"].upper()
                            iteration = entry["iteration"]
                            iter_text = f" (iteration {iteration + 1})" if iteration > 0 else ""
                            desc = entry["description"]
                            details = entry.get("details", [])
                            metrics = entry.get("metrics", {})

                            st.markdown(f"**✓ {step}**{iter_text}: {desc}")
                            for detail in details:
                                st.caption(f"  • {detail}")
                            if metrics:
                                cols = st.columns(len(metrics))
                                for i, (key, val) in enumerate(metrics.items()):
                                    cols[i].metric(key.replace("_", " ").title(), val)
                            st.divider()
                else:
                    st.caption("Pipeline steps will appear here after generation")

            # Audit log
            with st.expander("🔍 Audit Log", expanded=False):
                st.json(audit_log)

        else:
            st.caption(
                "Generated content will appear here after you submit the form."
            )


if __name__ == "__main__":
    main()
