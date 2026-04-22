# 🌾 KisanAgent: Hackathon Pitch & Presentation Notes

**Theme**: World Modeling & Reinforcement Learning (Meta-PyTorch OpenEnv Hackathon)

## 1. The Elevator Pitch (What this project does)
"KisanAgent is a production-grade, reinforcement learning environment that trains Large Language Models to act as expert agricultural advisors for Indian smallholder farmers. We built a highly realistic, 90-day simulation of a 2-acre tomato farm in Kolar, Karnataka. Instead of just fine-tuning an LLM on static text, we force the model to 'live' through simulated farming seasons where it must use APIs (like weather forecasts, soil sensors, and mandi prices) to make daily decisions. It learns through trial and error using **Group Relative Policy Optimization (GRPO)**, where the ultimate reward is the actual net profit (in Rupees) generated for the farmer at harvest."

## 2. The Environment (The "World Model")
To teach the AI to be a good advisor, we had to build a causal, partially observable world.

* **The Setup**: The agent plays on behalf of "Harish," starting with ₹15,000 capital. The season lasts exactly 90 days (from seedling to harvest).
* **Partial Observability**: The agent doesn't magically know the state of the world. It is given a budget of 3 tool calls per day. It must proactively call:
  * `weather` (IMD forecast)
  * `soil` (IoT sensors)
  * `mandi_price` (Market rates)
  * `pest_alert` (Govt agriculture dept)
  * `govt_scheme` (Subsidy deadlines)
* **The Actions**: It must choose exactly one action per day: `irrigate`, `fertilize`, `spray_pesticide`, `sell_now`, `hold_crop`, `apply_scheme`, `take_loan`, or `do_nothing`.
* **Causality & Consequences**: 
  * If it forgets to irrigate during fruiting, yield drops permanently.
  * If it sprays pesticide during the flowering stage without a verified pest threat, it kills pollinators and ruins the crop.
  * If it misses a government scheme deadline, it loses free capital.

## 3. How the Reward System Enforces Learning
We do **not** use an "LLM-as-a-judge" because it's subjective and expensive. Instead, we built the **KisanGrader**, a programmatic, multi-dimensional reward function that generates a strict score between `0.0` and `1.0`.

During GRPO training, the LLM generates 4 different "trajectories" (timelines) for the same state. The grader evaluates them based on 5 dimensions:

1. **Net Income (40%)**: The ultimate ground truth. Did the agent bankrupt Harish, or achieve the theoretical maximum of ₹40,000?
2. **Tool-Use Quality (20%)**: Did the agent call the *right* tools before acting? (e.g., You get a 0.0 if you `irrigate` without checking `soil` or `weather` first. You can't just guess).
3. **Pest Response (20%)**: Pests escalate from LOW to CRITICAL over 6 days. The agent is rewarded for treating them precisely within a tight 4-day window. If it sprays randomly (wasting money and killing bees), it is heavily penalized.
4. **Scheme Capture (10%)**: Did the agent check for and apply to government subsidies before the strict deadlines expired?
5. **Sustainability (10%)**: The agent is penalized if it uses more than 180,000 liters of water or applies chemicals more than 4 times a season.

**Why this forces learning:**
Because GRPO reinforces the trajectory with the highest relative reward, the agent quickly learns the causal rules of the world. It learns that "blindly spraying" yields a bad score. It learns that "checking the mandi price before selling" yields a higher score. Over 500 episodes (using a curriculum of easy → medium → hard seasons), the LLM develops a deep internal representation of agricultural economics and agronomy.

---

## 🛡️ Defending Against Judges' Questions

**Q: Why not just use standard prompt engineering or RAG?**
**A:** "RAG only retrieves text; it doesn't teach an LLM *consequences*. Farming is highly dynamic and temporal. If it rains tomorrow, you shouldn't irrigate today. RAG can't simulate temporal cause-and-effect. By placing the LLM in an RL environment and using GRPO, the model learns the *delayed consequences* of its actions, building a true internal 'world model' of a farm ecosystem."

**Q: How do you prevent the agent from just 'hacking' the reward function?**
**A:** "We use a multi-dimensional grader. If an agent tries to hack the 'Income' score by never spending money on pesticides, the crop dies, and the income drops. If it tries to ensure crop health by spraying pesticide every single day, the 'Sustainability' score tanks, and the 'Pest Response' score drops to 0.0 for unnecessary chemical use. The weights balance each other out perfectly."

**Q: Why GRPO instead of standard PPO?**
**A:** "GRPO (Group Relative Policy Optimization) is vastly superior for LLMs because it completely eliminates the need for a separate 'Critic' or Value model. PPO requires an entirely separate neural network to estimate the value of a state, doubling memory requirements. GRPO simply generates a group of answers (e.g., 4 rollouts), grades them programmatically, and reinforces the one that performed better than the group average. This allowed us to train a 7B model locally on a single GPU/Colab instance!"

**Q: Where did the agricultural data come from?**
**A:** "The simulation constraints are modeled after real-world Kolar district tomato farming parameters—factoring in actual IMD weather profiles, Agmarknet price volatility, and Karnataka Raitha Seva Kendra scheme deadlines. The OpenEnv server accurately mimics the unreliability and noise of real-world APIs."
