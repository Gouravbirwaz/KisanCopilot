from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .siren_env import SirenWorldEnv
import uvicorn

app = FastAPI(title="SirenWorld OpenEnv Wrapper")

# Initialize the global environment instance
env = SirenWorldEnv()

class ActionRequest(BaseModel):
    action: str  # JSON string action

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    return {"observation": obs, "info": info}

@app.post("/step", response_model=StepResponse)
async def step(request: ActionRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def get_state():
    return env._get_obs()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
