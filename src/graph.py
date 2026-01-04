"""Main LangGraph pipeline for the Event Content Generator."""

from __future__ import annotations

from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from .nodes import (
    critic_node,
    draft_node,
    export_node,
    generate_images_node,
    retrieve_node,
    verify_node,
)
from .schemas import ContentGeneratorState

# Pipeline step descriptions for UI (base descriptions)
PIPELINE_STEPS = {
    "retrieve": "Searching corpus for relevant brand voice and product docs...",
    "draft": "Writing content for selected channels...",
    "critic": "Evaluating brand voice alignment and CTA clarity...",
    "verify": "Verifying factual claims against source documents...",
    "generate_images": "Generating marketing images with Imagen 3...",
    "export": "Preparing final output...",
}


def get_step_details(step: str, state: ContentGeneratorState, after: bool = False) -> dict:
    """Get detailed step information for UI display.

    Args:
        step: Current step name
        state: Current pipeline state
        after: If True, show results after step ran; if False, show what's about to happen

    Returns:
        Dict with 'description', 'details' list, and optional 'metrics'
    """
    iteration = state.get("iteration", 0)
    channels = state.get("channels", [])

    if step == "retrieve":
        if after:
            brand_count = len(state.get("brand_chunks", []))
            product_count = len(state.get("product_chunks", []))
            return {
                "description": "Retrieved relevant documents from corpus",
                "details": [
                    f"Found {brand_count} brand voice chunks",
                    f"Found {product_count} product documentation chunks",
                ],
                "metrics": {"brand_chunks": brand_count, "product_chunks": product_count}
            }
        return {
            "description": "Searching corpus for relevant brand voice and product docs...",
            "details": [
                "Querying ChromaDB vector store",
                "Matching event description to indexed documents",
            ]
        }

    elif step == "draft":
        channel_list = ", ".join(ch.upper() for ch in channels)
        if after:
            return {
                "description": f"Drafted content for {len(channels)} channel(s)",
                "details": [f"Generated: {channel_list}"],
            }
        if iteration == 0:
            return {
                "description": f"Writing initial drafts for {channel_list}...",
                "details": [
                    "Using retrieved brand voice as style guide",
                    "Citing product docs for factual claims",
                ]
            }
        else:
            feedback = state.get("critic_feedback")
            issues = feedback.issues if feedback else []
            return {
                "description": f"Revising drafts based on critic feedback (iteration {iteration + 1})...",
                "details": [
                    f"Addressing {len(issues)} issue(s) from critic",
                    "Improving brand voice alignment" if feedback and feedback.brand_voice_score < 7 else "Strengthening call-to-action",
                ]
            }

    elif step == "critic":
        if after:
            feedback = state.get("critic_feedback")
            if feedback:
                status = "✅ PASSED" if feedback.passed else "🔄 NEEDS REVISION"
                return {
                    "description": f"Evaluation complete: {status}",
                    "details": [
                        f"Brand Voice: {feedback.brand_voice_score}/10",
                        f"CTA Clarity: {feedback.cta_clarity_score}/10",
                        f"Length OK: {'Yes' if feedback.length_ok else 'No'}",
                    ],
                    "metrics": {
                        "brand_voice": feedback.brand_voice_score,
                        "cta_clarity": feedback.cta_clarity_score,
                        "passed": feedback.passed,
                    }
                }
        return {
            "description": "Evaluating drafts against quality criteria...",
            "details": [
                "Checking brand voice alignment (target: 7+/10)",
                "Checking CTA clarity (target: 7+/10)",
                "Verifying length constraints",
            ]
        }

    elif step == "verify":
        if after:
            drafts = state.get("drafts", {})
            total_claims = sum(len(d.claims) for d in drafts.values())
            supported = sum(
                sum(1 for c in d.claims if c.is_supported)
                for d in drafts.values()
            )
            unsupported = total_claims - supported
            status = "✅ All claims verified" if unsupported == 0 else f"⚠️ {unsupported} unsupported claim(s)"
            return {
                "description": f"Verification complete: {status}",
                "details": [
                    f"Total claims extracted: {total_claims}",
                    f"Supported by corpus: {supported}",
                    f"Unsupported (will trigger revision): {unsupported}",
                ],
                "metrics": {"total_claims": total_claims, "supported": supported, "unsupported": unsupported}
            }
        return {
            "description": "Extracting and verifying factual claims...",
            "details": [
                "Identifying factual statements in drafts",
                "Matching claims to corpus chunk IDs",
                "Marking user-provided info as 'user_input'",
            ]
        }

    elif step == "generate_images":
        if after:
            images = state.get("images", {})
            count = len(images)
            channels_with_images = list(images.keys())
            return {
                "description": f"Generated {count} marketing image(s)",
                "details": [
                    f"Channels: {', '.join(ch.upper() for ch in channels_with_images)}" if channels_with_images else "No images generated",
                ],
                "metrics": {"images_generated": count}
            }
        return {
            "description": "Generating marketing images with Imagen 3...",
            "details": [
                "Creating channel-specific visuals",
                "Using event theme and audience context",
            ]
        }

    elif step == "export":
        if after:
            return {
                "description": "Content generation complete!",
                "details": [
                    f"Generated {len(channels)} channel(s) of content",
                    f"Completed in {iteration + 1} iteration(s)",
                ]
            }
        return {
            "description": "Preparing final output...",
            "details": [
                "Formatting content for each channel",
                "Building claims table with sources",
                "Compiling audit log",
            ]
        }

    return {"description": PIPELINE_STEPS.get(step, "Processing..."), "details": []}


def should_continue(state: ContentGeneratorState) -> str:
    """Determine whether to continue iterating or export final content.

    Decision logic:
    1. If max iterations (3) reached -> export
    2. If critic feedback says not passed -> draft again
    3. If any claims are unsupported -> draft again
    4. Otherwise -> export

    Args:
        state: Current pipeline state

    Returns:
        "draft" to loop back, "export" to finish
    """
    # Check iteration limit
    if state.get("iteration", 0) >= 3:
        return "export"

    # Check critic feedback
    critic_feedback = state.get("critic_feedback")
    if critic_feedback and not critic_feedback.passed:
        return "draft"

    # Check for unsupported claims
    drafts = state.get("drafts", {})
    for draft in drafts.values():
        for claim in draft.claims:
            if not claim.is_supported:
                return "draft"

    return "export"


def create_graph() -> StateGraph:
    """Create and compile the content generation pipeline.

    Pipeline flow:
    INPUT -> RETRIEVER -> DRAFTER -> CRITIC -> VERIFIER -> [LOOP?] -> GENERATE_IMAGES -> EXPORTER

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(ContentGeneratorState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("generate_images", generate_images_node)
    workflow.add_node("export", export_node)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "draft")
    workflow.add_edge("draft", "critic")
    workflow.add_edge("critic", "verify")

    # Conditional edge for the quality loop
    workflow.add_conditional_edges(
        "verify",
        should_continue,
        {
            "draft": "draft",  # Loop back for improvements
            "export": "generate_images",  # Generate images before export
        },
    )

    # Image generation leads to export
    workflow.add_edge("generate_images", "export")

    # Final edge to end
    workflow.add_edge("export", END)

    return workflow.compile()


# Create a singleton graph instance
graph = create_graph()


def run_pipeline(
    event_title: str,
    event_description: str,
    target_audience: str,
    key_messages: list[str],
    channels: list[str],
    event_date: str | None = None,
    relevant_urls: list[dict] | None = None,
    on_step: Optional[Callable[[str, str, int], None]] = None,
) -> dict:
    """Run the content generation pipeline with the given inputs.

    Args:
        event_title: Name of the event
        event_description: What the event is about
        target_audience: Who this content is for
        key_messages: 2-5 bullet points of what to communicate
        channels: List of channels to generate content for
        event_date: Optional event date
        relevant_urls: Optional list of {"label": str, "url": str} dicts
        on_step: Optional callback called before each step with (step_name, description, iteration)

    Returns:
        Final output dictionary with generated content and audit log
    """
    initial_state: ContentGeneratorState = {
        "event_title": event_title,
        "event_description": event_description,
        "event_date": event_date,
        "target_audience": target_audience,
        "key_messages": key_messages,
        "channels": channels,
        "relevant_urls": relevant_urls or [],
        "brand_chunks": [],
        "product_chunks": [],
        "drafts": {},
        "critic_feedback": None,
        "iteration": 0,
        "final_output": None,
        "audit_log": [],
        "images": {},
    }

    # If callback provided, run step-by-step for progress updates
    if on_step:
        return _run_pipeline_with_callbacks(initial_state, on_step)

    # Otherwise, run the graph normally
    result = graph.invoke(initial_state)
    return result.get("final_output", {})


def _run_pipeline_with_callbacks(
    state: ContentGeneratorState,
    on_step: Callable[[str, dict, int], None],
) -> dict:
    """Run pipeline step-by-step with progress callbacks.

    This allows the UI to show real-time progress updates.

    The callback receives:
        - step_name: Name of the step (retrieve, draft, critic, verify, export)
        - step_info: Dict with 'description', 'details' list, and optional 'metrics'
        - iteration: Current iteration number (0-indexed)
    """
    max_iterations = 3

    # Step 1: Retrieve
    on_step("retrieve", get_step_details("retrieve", state, after=False), 0)
    state = {**state, **retrieve_node(state)}
    on_step("retrieve_done", get_step_details("retrieve", state, after=True), 0)

    while True:
        iteration = state.get("iteration", 0)

        # Step 2: Draft
        on_step("draft", get_step_details("draft", state, after=False), iteration)
        state = {**state, **draft_node(state)}

        # Step 3: Critic
        on_step("critic", get_step_details("critic", state, after=False), iteration)
        state = {**state, **critic_node(state)}
        on_step("critic_done", get_step_details("critic", state, after=True), iteration)

        # Step 4: Verify
        on_step("verify", get_step_details("verify", state, after=False), iteration)
        state = {**state, **verify_node(state)}
        on_step("verify_done", get_step_details("verify", state, after=True), iteration)

        # Check if we should continue
        next_step = should_continue(state)
        if next_step == "export" or state.get("iteration", 0) >= max_iterations:
            break

    # Step 5: Generate Images
    iteration = state.get("iteration", 0)
    on_step("generate_images", get_step_details("generate_images", state, after=False), iteration)
    state = {**state, **generate_images_node(state)}
    on_step("generate_images_done", get_step_details("generate_images", state, after=True), iteration)

    # Step 6: Export
    on_step("export", get_step_details("export", state, after=False), iteration)
    state = {**state, **export_node(state)}
    on_step("export_done", get_step_details("export", state, after=True), iteration)

    return state.get("final_output", {})
