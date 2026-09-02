"""Run the tutor directly through ADK's Runner for local inspection."""

import argparse
import asyncio
from typing import Any

from dotenv import load_dotenv
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_4_gcp_learning_studio.agent import root_agent

load_dotenv()

DEFAULT_PROMPT = "What is the difference between a Cloud Run service and a job?"
DEFAULT_FOLLOW_UP = "Can you summarise that in one sentence?"
USER_ID = "local-learner"


def _print_event(event: Event) -> None:
    """Print the useful parts of one ADK event without dumping large payloads."""
    print(f"\nEVENT author={event.author}")

    # Show model text, tool calls, and compact tool responses in one place.
    if event.content:
        for part in event.content.parts or []:
            if part.text:
                print(f"  text: {part.text}")
            elif part.function_call is not None:
                print(
                    f"  function_call: {part.function_call.name} "
                    f"args={dict(part.function_call.args or {})}"
                )
            elif part.function_response is not None:
                response = part.function_response.response
                if isinstance(response, dict):
                    _print_function_response(part.function_response.name, response)
                else:
                    print(f"  function_response: {part.function_response.name}")

    # State changes are carried by events; printing the keys makes that visible.
    state_delta = event.actions.state_delta if event.actions else {}
    if state_delta:
        print(f"  state_delta keys: {sorted(state_delta)}")


def _print_function_response(name: str | None, response: dict[str, Any]) -> None:
    """Print a compact summary of a tool response."""
    print(f"  function_response: {name} keys={sorted(response)}")

    # Source text and grading guidance can be very large, so show their metadata.
    sources = response.get("sources")
    if isinstance(sources, list):
        print(f"  sources returned: {len(sources)}")
        for source in sources:
            if isinstance(source, dict):
                print(
                    "    - "
                    f"{source.get('title')} | {source.get('url')} | "
                    f"chunk={source.get('chunk_id')}"
                )


def _print_session_state(session: Any) -> None:
    """Print the final state snapshot produced by the invocation."""
    print("\nFINAL SESSION STATE")
    for key, value in sorted(session.state.items()):
        if key == "active_quiz" and isinstance(value, dict):
            print(
                f"  {key}: topic={value.get('topic')!r}, "
                f"status={value.get('status')!r}, "
                f"answered_count={value.get('answered_count')}"
            )
        else:
            print(f"  {key}: {value!r}")


async def run(prompt: str, follow_up: str) -> None:
    """Run two prompts in one Session and print the ADK execution trace.

    Args:
        prompt: The first user message to send to the tutor.
        follow_up: The second user message sent in the same Session.
    """
    # SessionService stores sessions, events, and state for the Runner.
    session_service = InMemorySessionService()

    # A Session is one conversation and its current state for this user.
    session = await session_service.create_session(
        app_name=root_agent.name,
        user_id=USER_ID,
    )

    # Runner orchestrates the agent, tools, events, and session service.
    runner = Runner(
        app_name=root_agent.name,
        agent=root_agent,
        session_service=session_service,
    )

    print(f"SESSION id={session.id}")
    # Reuse the same Runner and Session for both conversational turns.
    for turn_number, turn_prompt in enumerate((prompt, follow_up), start=1):
        # Content is the user message that starts this turn.
        message = types.Content(
            role="user",
            parts=[types.Part(text=turn_prompt)],
        )

        print(f"\nTURN {turn_number}")
        print(f"PROMPT: {turn_prompt}")
        # Runner streams each event as the model and tools make progress.
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=message,
        ):
            _print_event(event)

    # Reload the session to demonstrate where the final state can be read from.
    final_session = await session_service.get_session(
        app_name=root_agent.name,
        user_id=USER_ID,
        session_id=session.id,
    )
    if final_session is not None:
        _print_session_state(final_session)


def main() -> None:
    """Parse two prompts and run them in one local agent session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("follow_up", nargs="?", default=DEFAULT_FOLLOW_UP)
    args = parser.parse_args()
    asyncio.run(run(args.prompt, args.follow_up))


if __name__ == "__main__":
    main()
