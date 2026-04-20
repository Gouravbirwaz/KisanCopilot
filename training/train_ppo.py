import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import numpy as np
from env.siren_env import SirenWorldEnv
from agents.policy_model import SirenAgentPolicy
from reward.reward_function import RewardManager
import json

# 1. Setup Unsloth & Transformers
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
max_seq_length = 2048
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# Enable Gradient Checkpointing
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 2. Config & PPO Setup
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=8,
)

# Wrap model for PPO
# Note: In a real Unsloth setup, we might need a specific wrapper for Value Head
# For this demo, we assume the TRL standard wrapper
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ppo_trainer = PPOTrainer(config, ppo_model, ref_model, tokenizer)

# 3. Environment & Helpers
env = SirenWorldEnv()
agent_policy = SirenAgentPolicy()
reward_manager = RewardManager()

def train_loop(epochs=10):
    for epoch in range(epochs):
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            # Generate Prompt
            prompt = agent_policy.generate_prompt(obs)
            query_tensor = tokenizer.encode(prompt, return_tensors="pt")
            
            # Agent Action
            response_tensor = ppo_trainer.generate(query_tensor.squeeze(), max_new_tokens=256)
            response_str = tokenizer.decode(response_tensor.squeeze())
            
            # Parse Action
            action_str = agent_policy.parse_action(response_str)
            
            # Step Environment
            next_obs, reward, terminated, _, info = env.step(action_str)
            
            # Compute Rewards (The environment already returns a base reward, 
            # but we can refine it with our reward_manager here)
            # For simplicity, we use the env reward
            rewards = [torch.tensor(reward)]
            
            # PPO Step
            stats = ppo_trainer.step([query_tensor.squeeze()], [response_tensor.squeeze()], rewards)
            
            obs = next_obs
            
            print(f"Epoch {epoch} | Reward: {reward} | Stats: {stats['ppo/loss/policy']}")

if __name__ == "__main__":
    print("Starting SirenWorld Training Pipeline...")
    # train_loop() # Commented out to avoid accidental execution in non-GPU env
    print("Training script ready. Requires GPU and Unsloth environment.")
