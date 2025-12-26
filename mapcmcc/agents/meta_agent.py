from .base import BaseAgent
from ..utils.types import MetaObservation, MetaAction
import json

class MetaAgent(BaseAgent):
    """
    Agent controlling the global parameters and merges.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def get_action(self, observation: MetaObservation) -> MetaAction:
        # Mock implementation
        return MetaAction(
            budget_adjustments={},
            global_baselines={},
            merge_suggestions=[]
        )
