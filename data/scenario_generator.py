import json
import random
import os
from typing import List, Dict

class ScenarioGenerator:
    """
    Generates diverse disaster response scenarios for SirenWorldEnv training.
    """
    
    def __init__(self, output_dir: str = "data/scenarios"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.categories = ["fire", "medical", "disaster", "false_alarm"]
        self.templates = {
            "fire": [
                "Residential fire at {location}. Multiple families trapped.",
                "Industrial smoke detected in {sector} sector.",
                "Wildfire spreading near the {boundary} park."
            ],
            "medical": [
                "Elderly person unconscious at {location}.",
                "Multi-vehicle collision on {highway}. Severe injuries.",
                "Allergic reaction reported at {public_place}."
            ],
            "disaster": [
                "Flash flood in {location}. Roads submerged.",
                "Gas leak reported near {subway} station.",
                "Building collapse in {downtown} area."
            ],
            "false_alarm": [
                "Silent alarm triggered at {bank}.",
                "Suspect package at {station} - likely empty trash.",
                "Report of 'fire' turned out to be a legal campfire at {park}."
            ]
        }

    def generate_batch(self, count: int = 100, filename: str = "train_scenarios.json"):
        scenarios = []
        for i in range(count):
            cat = random.choice(self.categories)
            severity = random.randint(1, 10)
            loc = [random.uniform(0, 100), random.uniform(0, 100)]
            
            # Simple template filling
            desc = random.choice(self.templates[cat]).format(
                location="Main St", sector="Industrial", boundary="North", 
                highway="I-95", public_place="Mall", subway="Central", 
                downtown="Financial", bank="City Bank", station="Metro", park="East"
            )
            
            scenarios.append({
                "id": f"SCN_{i}",
                "category": cat,
                "description": desc,
                "true_severity": severity,
                "location": loc
            })
            
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(scenarios, f, indent=4)
        print(f"Generated {count} scenarios in {filepath}")

if __name__ == "__main__":
    gen = ScenarioGenerator()
    gen.generate_batch(count=50, filename="val_scenarios.json")
    gen.generate_batch(count=200, filename="train_scenarios.json")
