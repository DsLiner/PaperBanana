from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from paperbanana.agents import PaperBananaAgents
from paperbanana.schema import (
    PipelineResult,
    PaperBananaState,
    ReferenceExample,
    PaperBananaTask,
)


class PaperBananaGraph:
    def __init__(self, agents: PaperBananaAgents):
        self.agents = agents
        self._compiled = self._build()

    def run(
        self,
        task: PaperBananaTask,
        references: list[ReferenceExample],
        style_guide: str,
    ) -> PipelineResult:
        state: PaperBananaState = {
            "task": task,
            "reference_pool": references,
            "style_guide": style_guide,
            "top_k": self.agents.config.top_k,
            "max_iterations": self.agents.config.max_iterations,
            "retrieved_ids": [],
            "retrieved_examples": [],
            "planner_description": "",
            "styled_description": "",
            "current_description": "",
            "latest_artifact": "",
            "artifact_history": [],
            "critic_feedback": [],
            "iteration": 0,
            "stop_refinement": False,
            "final_artifact": "",
        }

        final_state = self._compiled.invoke(state)
        return PipelineResult(
            final_artifact=final_state["final_artifact"],
            artifact_history=final_state["artifact_history"],
            retrieved_ids=final_state["retrieved_ids"],
            planner_description=final_state["planner_description"],
            styled_description=final_state["styled_description"],
            critic_feedback=final_state["critic_feedback"],
        )

    def _build(self):
        graph = StateGraph(PaperBananaState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("plan", self._plan)
        graph.add_node("style", self._style)
        graph.add_node("visualize", self._visualize)
        graph.add_node("critic", self._critic)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("plan", "style")
        graph.add_edge("style", "visualize")
        graph.add_edge("visualize", "critic")
        graph.add_conditional_edges(
            "critic",
            self._should_continue,
            {"loop": "visualize", "finish": "finalize"},
        )
        graph.add_edge("finalize", END)

        return graph.compile()

    def _retrieve(self, state: PaperBananaState) -> dict[str, object]:
        top_ids = self.agents.retrieve(
            task=state["task"],
            references=state["reference_pool"],
            top_k=state["top_k"],
        )
        ordered_examples: list[ReferenceExample] = []
        by_id = {ref.ref_id: ref for ref in state["reference_pool"]}
        for ref_id in top_ids:
            item = by_id.get(ref_id)
            if item is not None:
                ordered_examples.append(item)

        return {
            "retrieved_ids": top_ids,
            "retrieved_examples": ordered_examples,
        }

    def _plan(self, state: PaperBananaState) -> dict[str, object]:
        description = self.agents.plan(
            task=state["task"],
            retrieved_examples=state["retrieved_examples"],
        )
        return {
            "planner_description": description,
            "current_description": description,
        }

    def _style(self, state: PaperBananaState) -> dict[str, object]:
        styled = self.agents.style(
            task=state["task"],
            planner_description=state["planner_description"],
            style_guide=state["style_guide"],
        )
        return {
            "styled_description": styled,
            "current_description": styled,
        }

    def _visualize(self, state: PaperBananaState) -> dict[str, object]:
        artifact = self.agents.visualize(
            task=state["task"],
            description=state["current_description"],
            output_dir=self.agents.config.output_dir,
            iteration=state["iteration"],
        )
        return {
            "latest_artifact": artifact,
            "artifact_history": [*state["artifact_history"], artifact],
        }

    def _critic(self, state: PaperBananaState) -> dict[str, object]:
        suggestion, revised = self.agents.critic(
            task=state["task"],
            current_description=state["current_description"],
            image_path=state["latest_artifact"],
            iteration=state["iteration"],
        )

        should_stop = self._critic_requests_stop(
            suggestion=suggestion,
            revised_description=revised,
            current_description=state["current_description"],
        )

        next_description = state["current_description"] if should_stop else revised
        return {
            "critic_feedback": [*state["critic_feedback"], suggestion],
            "current_description": next_description,
            "iteration": state["iteration"] + 1,
            "stop_refinement": should_stop,
        }

    def _should_continue(self, state: PaperBananaState) -> str:
        if state["stop_refinement"]:
            return "finish"
        if state["iteration"] >= state["max_iterations"]:
            return "finish"
        return "loop"

    @staticmethod
    def _critic_requests_stop(
        suggestion: str,
        revised_description: str,
        current_description: str,
    ) -> bool:
        suggestion_lower = suggestion.strip().lower()
        revised_lower = revised_description.strip().lower()

        stop_markers = (
            "no changes needed",
            "no further changes",
            "looks good",
            "sufficient",
            "good as is",
        )
        if any(marker in suggestion_lower for marker in stop_markers):
            return True
        if any(marker in revised_lower for marker in stop_markers):
            return True

        if not revised_description.strip():
            return True
        if revised_description.strip() == current_description.strip():
            return True

        return False

    @staticmethod
    def _finalize(state: PaperBananaState) -> dict[str, object]:
        return {"final_artifact": state["latest_artifact"]}
