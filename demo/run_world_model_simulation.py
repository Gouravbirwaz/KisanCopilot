import os
import sys
import json
import time
from env.siren_env import SirenWorldEnv
from agents.groq_agent import GroqWorldModelAgent
from utils.helpers import format_location

def run_simulation():
    print("\n" + "="*60)
    print(" SIRENWORLD FINALE: THEME #3 - WORLD MODELING ".center(60, "="))
    print("="*60 + "\n")

    env = SirenWorldEnv()
    agent = GroqWorldModelAgent()
    
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    print(f"[*] Initial World State: Weather={obs['env_conditions']['weather']}")
    print(f"[*] Initial Mental Map Spread Rates: {json.dumps(agent.get_mental_map()['zones'], indent=2)}")
    
    while not done and step_count < 10:
        step_count += 1
        print(f"\n--- STEP {step_count} ---")
        
        # 1. Agent Logic (Groq Powered)
        result = agent.act(obs)
        reasoning = result.get("reasoning", "No reasoning provided.")
        action_dict = result.get("action", {})
        
        print(f"\n[REASONING]: {reasoning}")
        
        if action_dict:
            print(f"[ACTION]: Dispatching to {action_dict.get('event_id')} | Classification: {action_dict.get('classification')}")
        else:
            print("[ACTION]: No immediate action decided.")
            
        # 2. Env Step
        # The env expects action as JSON string
        action_str = json.dumps(action_dict)
        next_obs, reward, done, _, info = env.step(action_str)
        total_reward += reward
        
        # 3. Visualization of Hidden Variables (Storytelling 30%)
        real_state = info.get("full_state", {})
        print(f"\n[SYSTEM FEEDBACK]: Reward={reward:.2f} | Total={total_reward:.2f}")
        
        # Compare mental map vs Reality for specific event
        if action_dict and action_dict.get("event_id"):
            eid = action_dict["event_id"]
            real_ev = next((e for e in real_state["events"] if e["id"] == eid), None)
            if real_ev:
                print(f"[WORLD MODEL CHECK]: Event {eid} Spread Rate: {real_ev['latent_spread_rate']:.3f}")
        
        obs = next_obs
        time.sleep(1) # For demo pacing

    print("\n" + "="*60)
    print(" SIMULATION COMPLETE ".center(60, "="))
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Final Mental Map Beliefs: {json.dumps(agent.get_mental_map()['zones'], indent=2)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"\n[!] Critical Error during simulation: {e}")
        import traceback
        traceback.print_exc()
