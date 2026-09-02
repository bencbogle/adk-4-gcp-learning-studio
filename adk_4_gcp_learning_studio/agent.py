"""Define the learning studio's root ADK agent."""

from google.adk.agents import Agent

from adk_4_gcp_learning_studio.callbacks import log_model_response, log_tool_result
from adk_4_gcp_learning_studio.quiz_workflows import start_quiz, submit_quiz_answer
from adk_4_gcp_learning_studio.tools import (
    list_supported_products,
    search_gcp_documentation,
)

root_agent = Agent(
    name="gcp_learning_studio",
    model="gemini-2.5-flash",
    rerun_on_resume=True,
    instruction=(
        "You are a patient Google Cloud tutor. For factual questions, always use "
        "the search_gcp_documentation tool first. Answer using only its returned "
        "sources. If it returns no sources, say that you do not have enough "
        "documentation to answer. Cite each source as a Markdown link. If the "
        "learner asks what products you cover, use list_supported_products. If "
        "the learner asks for a quiz, use start_quiz and present its returned "
        "question without revealing the grading_guidance. When a quiz answer is "
        "provided, evaluate it against the grading_guidance, then use "
        "submit_quiz_answer and present its returned feedback and next question."
    ),
    after_model_callback=log_model_response,
    after_tool_callback=log_tool_result,
    tools=[
        search_gcp_documentation,
        list_supported_products,
        start_quiz,
        submit_quiz_answer,
    ],
)
