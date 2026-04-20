import time
import json
import random
from env.siren_env import SirenWorldEnv
from evaluation.metrics import MetricsTracker

def run_live_simulation(steps=10):
    print("🔥 Starting SirenWorld Live Simulation Demo 🔥")
    print("-" * 50)
    
    env = SirenWorldEnv()
    tracker = MetricsTracker()
    
    obs, info = env.reset()
    
    for i in range(steps):
        print(f"\n[STEP {i+1}]")
        env.render()
        
        # Simple Logic-based Agent (Mocking the LLM behavior)
        sos_requests = obs.get("incoming_sos", [])
        if sos_requests:
            target = sos_requests[0]
            # Match resource
            free_res = [r for r in obs["resources"] if r["status"] == "free"]
            dispatch = [free_res[0]["id"]] if free_res else []
            
            action = {
                "thought": f"Responding to {target['id']} with available units.",
                "event_id": target["id"],
                "classification": "fire" if "fire" in target["description"].lower() else "medical",
                "dispatch": dispatch,
                "routing": "shortest"
            }
        else:
            action = {"thought": "No active SOS. Monitoring station.", "event_id": None}

        # Convert to JSON as required by env
        action_str = json.dumps(action)
        print(f"Agent Action: {action['thought']}")
        
        # Step
        obs, reward, terminated, _, info = env.step(action_str)
        tracker.log_step(reward, info)
        
        print(f"Step Reward: {reward}")
        time.sleep(1)
        
        if terminated:
            break

    tracker.print_report()

if __name__ == "__main__":
    run_live_simulation()
