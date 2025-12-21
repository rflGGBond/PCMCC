import json
import os
from typing import Dict, Any, Optional

class LLMClient:
    """
    A simple client to interact with an LLM provider (e.g., OpenAI, Anthropic, or Local).
    Currently implements a mock/simulator for demonstration, but structured to be easily swapped.
    """
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def get_completion(self, system_prompt: str, user_prompt: str, response_format: str = "json") -> str:
        """
        Sends a prompt to the LLM and returns the response content.
        """
        if self.provider == "mock":
            return self._mock_response(system_prompt, user_prompt)
        
        elif self.provider == "openai":
            # Real implementation example (commented out to avoid dependency errors if not installed)
            # import openai
            # client = openai.OpenAI(api_key=self.api_key)
            # response = client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     response_format={"type": "json_object"} if response_format == "json" else None
            # )
            # return response.choices[0].message.content
            pass
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a fake JSON response based on keywords in the prompt.
        This allows testing the flow without paying for tokens.
        """
        # Detect if this is a Community Agent or Meta Agent request
        if "Community Agent" in system_prompt:
            # Simulate a decision to adjust parameters slightly
            return json.dumps({
                "reasoning": "Performance is stable, increasing exploration slightly.",
                "action_type": "adjust_parameters",
                "parameters": {
                    "cr1": 0.4,
                    "cr2": 0.4,
                    "beta": 2.5,
                    "alpha": 10.0
                },
                "candidate_seed_set": None
            })
        
        elif "Meta Agent" in system_prompt:
            # Simulate a decision to keep baselines
            return json.dumps({
                "reasoning": "Global convergence is proceeding normally. No merges needed yet.",
                "global_baselines": {
                    "cr1": 0.3, 
                    "cr2": 0.3,
                    "beta": 2.0,
                    "alpha": 12.0
                },
                "budget_adjustments": {},
                "merge_suggestions": []
            })
            
        return "{}"
