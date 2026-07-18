"""State schema for the Document Analyst graph (Task 1.1).

TODO: Define `AnalystState` as a TypedDict with the fields from the spec table:
  messages, plan, current_step_index, step_results, next_agent, final_answer.
Use `Annotated[list, add_messages]` for `messages`.
"""

from __future__ import annotations

from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    # TODO: plan, current_step_index, step_results, next_agent, final_answer
    """
    messages - add_messages reducer, conversation history 
    plan - ordered list of steps to complete the task
    current_step_index - index of the current step in the plan
    step_results - results of completed steps
    next_agent - the next agent to execute the next step
    final_answer - the final answer to the task, if available
  
    """
    messages: Annotated[list, add_messages]
    plan: List[str]
    current_step_index: int
    step_results: List[str]
    next_agent: str
    final_answer: str