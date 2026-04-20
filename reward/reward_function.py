from typing import Dict, Any, List

class RewardManager:
    """
    Modular reward function for SirenWorldEnv.
    Calculates multi-component rewards based on agent actions and world outcomes.
    """
    
    def __init__(self):
        self.weights = {
            "accuracy": {"correct": 20.0, "wrong": -15.0},
            "effectiveness": {"correct_team": 25.0, "wrong_team": -20.0, "insufficient": -10.0},
            "time": {"fast": 10.0, "delayed": -10.0},
            "optimization": {"efficient": 5.0, "waste": -8.0},
            "safety": {"success": 30.0, "failure": -30.0},
            "exploration": 3.0,
            "penalty_ignored": -25.0
        }

    def calculate_reward(self, 
                         action: Dict[str, Any], 
                         event: Any, 
                         outcome: str, 
                         time_taken: int) -> Dict[str, float]:
        """
        Calculate breakdown of rewards for a specific step.
        """
        rewards = {
            "accuracy": 0.0,
            "effectiveness": 0.0,
            "time": 0.0,
            "optimization": 0.0,
            "safety": 0.0,
            "exploration": 0.0,
            "penalty": 0.0
        }

        # 1. Decision Accuracy
        if action.get("classification") == event.category:
            rewards["accuracy"] = self.weights["accuracy"]["correct"]
        else:
            rewards["accuracy"] = self.weights["accuracy"]["wrong"]

        # 2. Response Effectiveness (Simplified check here, real check in env)
        # Assuming environment passed 'outcome'
        if outcome == "effective":
            rewards["effectiveness"] = self.weights["effectiveness"]["correct_team"]
        elif outcome == "ineffective":
            rewards["effectiveness"] = self.weights["effectiveness"]["wrong_team"]

        # 3. Time Efficiency
        if time_taken < 15: # < 15 mins is fast
            rewards["time"] = self.weights["time"]["fast"]
        else:
            rewards["time"] = self.weights["time"]["delayed"]

        # 4. Resource Optimization
        dispatch_count = len(action.get("dispatch", []))
        if dispatch_count == 1:
            rewards["optimization"] = self.weights["optimization"]["efficient"]
        elif dispatch_count > 2:
            rewards["optimization"] = self.weights["optimization"]["waste"]

        # 5. Safety Outcome (Delayed reward usually, but computed per step if resolution occurs)
        if outcome == "resolved":
            rewards["safety"] = self.weights["safety"]["success"]
        elif outcome == "escalated":
            rewards["safety"] = self.weights["safety"]["failure"]

        # 6. Exploration Bonus
        if action.get("action_type") == "info_request":
            rewards["exploration"] = self.weights["exploration"]

        return rewards

    def total_reward(self, reward_breakdown: Dict[str, float]) -> float:
        return sum(reward_breakdown.values())
