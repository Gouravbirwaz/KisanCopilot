# SirenWorldEnv: World Modeling RL Environment

A research-grade reinforcement learning environment for training agentic models in disaster response, compatible with **OpenEnv** and **TRL**.

## 🚀 Overview

SirenWorldEnv simulates a multi-step disaster response scenario where an LLM agent must:
1.  **Observe**: Process partially observable SOS requests and resource states.
2.  **Reason**: Use Chain of Thought to prioritize calls.
3.  **Act**: Dispatch appropriate resources via JSON actions.
4.  **Adapt**: Respond to real-time world changes (weather, road blocks, new emergencies).

## 🛠 Project Structure

```text
sirennet_world_model/
├── env/
│   ├── siren_env.py    # Core Gymnasium Environment
│   └── server.py       # FastAPI OpenEnv Wrapper
├── agents/
│   └── policy_model.py  # Mistral Prompt & Parsing Logic
├── training/
│   └── train_ppo.py     # TRL/Unsloth Training Pipeline
├── reward/
│   └── reward_function.py # Multi-component Reward logic
├── data/
│   └── scenario_generator.py # Scenario dataset builder
├── evaluation/
│   └── metrics.py      # Performance tracking
├── demo/
│   └── run_simulation.py # Live simulation demo
└── Dockerfile          # Containerization for OpenEnv
```

## 🚥 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python -m demo.run_simulation
```

### 3. Start the OpenEnv Server (FastAPI)
```bash
python -m env.server
```

### 4. Training (Requires GPU)
```bash
python -m training.train_ppo
```

## 🧠 Reward System

The environment implements a high-quality multi-component reward function:
- **Decision Accuracy**: +20 / -15 (Correct classification)
- **Response Effectiveness**: +25 / -20 (Correct resource type)
- **Safety Outcome**: +30 / -30 (Resolution success)
- **Time/Optimization**: Efficiency bonuses

## 🐳 Docker Support

To run the environment as a containerized OpenEnv service:
```bash
docker build -t sirenworld-env .
docker run -p 8000:8000 sirenworld-env
```
