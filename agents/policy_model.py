import json
from typing import Dict, Any

class SirenAgentPolicy:
    """
    Handles prompt generation and action parsing for the Mistral agent.
    """
    
    SYSTEM_PROMPT = """You are the SirenWorld Dispatch AI. Your goal is to optimize emergency response.
You will receive the current world state including SOS requests and resource availability.
Analyze the situation, reason about the priority and resource allocation, and then output your decision in JSON format.

Your output MUST be a JSON object with the following structure:
{
    "thought": "Briefly explain your reasoning and priorities",
    "event_id": "ID of the SOS event you are responding to",
    "classification": "fire | medical | disaster | false_alarm",
    "dispatch": ["LIST_OF_RESOURCE_IDs"],
    "routing": "shortest | safest | efficient"
}
"""

    def generate_prompt(self, obs: Dict[str, Any]) -> str:
        """
        Convert observation dict to text prompt.
        """
        prompt = f"{self.SYSTEM_PROMPT}\n\n### Current State:\n"
        
        # Format SOS Events
        prompt += "Incoming SOS Requests:\n"
        for sos in obs.get("incoming_sos", []):
            prompt += f"- ID: {sos['id']}, Description: {sos['description']}, Severity Est: {sos['severity_estimate']}, Location: {sos['location']}\n"
        
        # Format Resources
        prompt += "\nAvailable Resources:\n"
        for res in obs.get("resources", []):
            if res["status"] == "free":
                prompt += f"- ID: {res['id']}, Type: {res['type']}, Location: {res['location']}\n"
        
        # Format Env
        env = obs.get("environment", {})
        prompt += f"\nEnvironment: Weather: {env.get('weather')}, Time: {env.get('time')} mins\n"
        
        prompt += "\n### Your Decision (JSON):"
        return prompt

    def parse_action(self, response: str) -> str:
        """
        Extract the JSON part from the model's response.
        """
        try:
            # Look for JSON block
            if "{" in response:
                json_part = response[response.find("{"):response.rfind("}")+1]
                # Validate it's reachable JSON
                json.loads(json_part)
                return json_part
            return "{}"
        except:
            return "{}"
