import os
from typing import Dict, Any, List
from groq import Groq
from dotenv import load_dotenv
import json
from utils.helpers import extract_json

load_dotenv()

class GroqWorldModelAgent:
    """
    Sirennet Agent powered by Groq LPU.
    Focuses on Theme #3: World Modeling and Causal Reasoning.
    """
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
            
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        
        # Theme #3: Persistent Internal World Model
        self.world_belief = {
            "zones": {
                "Industrial": {"estimated_spread": 0.1, "priority": "high"},
                "Residential": {"estimated_spread": 0.08, "priority": "medium"},
                "Commercial": {"estimated_spread": 0.05, "priority": "low"},
            },
            "resource_status": {} 
        }
        self.history = []

    def _generate_system_prompt(self) -> str:
        return f"""You are the SirenWorld Dispatch AI, capable of complex World Modeling.
Your task is to orchestrate emergency responses while maintaining an accurate internal belief of the city's state.

WORLD PRINCIPLES:
1. LATENT GROWTH: Fires and disasters grow based on hidden zone-specific spread rates. 
2. RESOURCE FATIGUE: Units lose energy with every task. Low energy ( < 30%) drastically slows them down.
3. CAUSAL IMPACT: Your decisions today impact resource availability for future, potentially more lethal, events.

YOUR CURRENT MENTAL MAP (WORLD BELIEF):
{json.dumps(self.world_belief, indent=2)}

You must respond in VALID JSON format with these exact keys:
- "reasoning": A detailed Chain-of-Thought explanation of why you are taking this action based on your world model.
- "latent_variable_updates": Any changes you want to make to your internal belief about zone spread rates or resource fatigue based on recent events.
- "action": your dispatch command. Format: {{"event_id": "SOS_1", "dispatch": ["AMBULANCE_0"], "classification": "medical"}}. Use empty values if no action is needed.
"""

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent decision loop using Groq for high-speed inference.
        """
        prompt = f"CURRENT OBSERVATION:\n{json.dumps(observation, indent=2)}\n\nUpdated your mental map and decide on the next dispatch strategy."
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._generate_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            result = extract_json(response_text)
            
            # Update internal belief based on Agent's own reasoning/updates
            if "latent_variable_updates" in result:
                self._apply_belief_updates(result["latent_variable_updates"])
                
            return result
        except Exception as e:
            print(f"Groq API Error: {e}")
            return {"reasoning": "Error occurred", "action": {}}

    def _apply_belief_updates(self, updates: Dict[str, Any]):
        """Theme #3: Persist the agent's learning into its world model."""
        if not isinstance(updates, dict): return
        
        if "zones" in updates:
            for zone, data in updates["zones"].items():
                if zone in self.world_belief["zones"]:
                    self.world_belief["zones"][zone].update(data)
        
        if "resource_status" in updates:
            self.world_belief["resource_status"].update(updates["resource_status"])

    def get_mental_map(self) -> Dict[str, Any]:
        return self.world_belief
