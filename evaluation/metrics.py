import numpy as np
from typing import List, Dict

class MetricsTracker:
    """
    Tracks and computes performance metrics for the SirenWorld agent.
    """
    
    def __init__(self):
        self.rewards = []
        self.successes = []
        self.response_times = []
        self.resource_utilization = []

    def log_step(self, reward: float, info: Dict):
        self.rewards.append(reward)
        
        # Extract metadata from info
        true_events = info.get("true_events", [])
        resolved = [e for e in true_events if e["status"] == "resolved"]
        pending = [e for e in true_events if e["status"] == "pending"]
        
        # Success Rate: resolved / total
        total = len(true_events)
        if total > 0:
            self.successes.append(len(resolved) / total)

    def get_summary(self) -> Dict:
        return {
            "avg_reward": np.mean(self.rewards) if self.rewards else 0,
            "max_reward": np.max(self.rewards) if self.rewards else 0,
            "success_rate": np.mean(self.successes) if self.successes else 0,
            "total_steps": len(self.rewards)
        }

    def print_report(self):
        summary = self.get_summary()
        print("\n" + "="*30)
        print("SIRENWORLD PERFORMANCE REPORT")
        print("="*30)
        print(f"Average Reward:  {summary['avg_reward']:.2f}")
        print(f"Success Rate:    {summary['success_rate']*100:.1f}%")
        print(f"Total Decisions: {summary['total_steps']}")
        print("="*30 + "\n")
