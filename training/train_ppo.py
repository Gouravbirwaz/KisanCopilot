try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
except Exception:
    HAS_UNSLOTH = False

import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import numpy as np
import json
from env.siren_env import SirenWorldEnv
from agents.policy_model import SirenAgentPolicy
from reward.reward_function import compute_reward

# 1. Configuration
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=8,
    mini_batch_size=4,
    gradient_accumulation_steps=2,
    # optimize_cuda_cache=True,
)

# 2. Setup Unsloth Model (Only if GPU/Unsloth available)
if HAS_UNSLOTH and torch.cuda.is_available():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # Enable PEFT (LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
    )
else:
    print("Warning: GPU or Unsloth not detected. Skipping model loading. Training will run in MOCK mode.")
    model, tokenizer = None, None

# Wrap for TRL
# Note: In production, use ValueHead enabled models
# model_ppo = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# tokenizer.pad_token = tokenizer.eos_token

# 3. Training Logic
def run_training():
    env = SirenWorldEnv()
    policy = SirenAgentPolicy()
    # ppo_trainer = PPOTrainer(config, model_ppo, None, tokenizer)

    print("--- Starting SirenWorld RL Training Loop ---")
    
    for epoch in range(10):
        obs, _ = env.reset()
        done = False
        epoch_rewards = []
        
        while not done:
            # Generate Prompt
            prompt = policy.generate_full_prompt(obs)
            
            # Agent Forward (Simulated during training setup)
            # In real training: response = ppo_trainer.generate(prompt)
            # For hackathon demo structure:
            print(f"Epoch {epoch} | Generating decision for Time {obs['env_conditions']['time']}...")
            
            # Placeholder for actual generation
            mock_action = '{"event_id": "SOS_1", "dispatch": ["AMBULANCE_0"], "classification": "medical"}'
            
            # Step Env
            next_obs, reward, done, _, info = env.step(mock_action)
            epoch_rewards.append(reward)
            
            obs = next_obs
            
        print(f"Epoch {epoch} Completed. Avg Reward: {np.mean(epoch_rewards)}")

if __name__ == "__main__":
    if HAS_UNSLOTH and torch.cuda.is_available():
        # run_training()
        print("Optimization Script Ready. GPU environment detected.")
    else:
        print("\n" + "!"*50)
        print("GPU ACCELERATOR NOT FOUND")
        print("SirenWorld Training requires a CUDA-enabled GPU.")
        print("Running mock verification of environment logic...")
        print("!"*50 + "\n")
        run_training()
