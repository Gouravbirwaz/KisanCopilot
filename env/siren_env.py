import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class Resource(BaseModel):
    id: str
    type: str  # 'ambulance', 'fire_truck', 'police_car', 'rescue_team'
    location: List[float]
    status: str  # 'free', 'busy', 'returning'
    fuel: float = 100.0
    capacity: int = 1

class SOSEvent(BaseModel):
    id: str
    description: str
    category: str  # 'fire', 'medical', 'disaster', 'false_alarm'
    true_severity: int  # 1-10
    location: List[float]
    status: str  # 'pending', 'active', 'resolved'
    timestamp: int
    assigned_resources: List[str] = []

class SirenWorldEnv(gym.Env):
    """
    SirenWorldEnv: A world-modeling environment for disaster response.
    Inspired by Sirennet concept.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Grid Size: 100x100
        self.grid_size = self.config.get("grid_size", 100.0)
        
        # Define Action Space: Dict-based for internal logic, but we accept JSON strings
        # In Gymnasium, we'd normally use MultiDiscrete, but for LLMs, we use text.
        # We'll define a dummy action space for standard compatibility.
        self.action_space = spaces.Text(min_length=0, max_length=1000)
        
        # Define Observation Space: Flat string for LLMs or Dict for structured agents
        self.observation_space = spaces.Dict({
            "incoming_sos": spaces.List(spaces.Dict({
                "id": spaces.Text(min_length=0, max_length=10),
                "description": spaces.Text(min_length=0, max_length=500),
                "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
                "severity_estimate": spaces.Discrete(3), # 0: Low, 1: Medium, 2: High
            })),
            "resources": spaces.List(spaces.Dict({
                "id": spaces.Text(min_length=0, max_length=10),
                "type": spaces.Text(min_length=0, max_length=20),
                "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
                "status": spaces.Text(min_length=0, max_length=15),
            })),
            "environment": spaces.Dict({
                "weather": spaces.Text(min_length=0, max_length=20),
                "time": spaces.Discrete(1440), # Minutes in a day
            })
        })

        self.state = {}
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        
        # Initialize Resources
        self.resources = [
            Resource(id=f"AMB_{i}", type="ambulance", location=[40.0, 40.0], status="free") for i in range(3)
        ] + [
            Resource(id=f"FIRE_{i}", type="fire_truck", location=[60.0, 60.0], status="free") for i in range(2)
        ] + [
            Resource(id=f"POL_{i}", type="police_car", location=[50.0, 50.0], status="free") for i in range(3)
        ]
        
        # Initialize Events
        self.events = []
        self._spawn_event(count=2)
        
        self.current_time = 0
        self.weather = random.choice(["clear", "rainy", "stormy", "foggy"])
        
        return self._get_obs(), self._get_info()

    def _spawn_event(self, count=1):
        for _ in range(count):
            eid = f"SOS_{len(self.events) + 1}"
            category = random.choice(["fire", "medical", "disaster", "false_alarm"])
            severity = random.randint(1, 10)
            location = [random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)]
            
            descriptions = {
                "fire": ["Smoke sighted at building A", "Apartment fire reported", "Kitchen fire out of control"],
                "medical": ["Person collapsed on sidewalk", "Car accident with injuries", "Suspected heart attack"],
                "disaster": ["Flood blocking main road", "Gas leak detected in subway", "Massive structural collapse"],
                "false_alarm": ["Loud bang heard", "Accidental alarm trigger", "Smoke from BBQ mistaken for fire"]
            }
            
            self.events.append(SOSEvent(
                id=eid,
                description=random.choice(descriptions[category]),
                category=category,
                true_severity=severity,
                location=location,
                status="pending",
                timestamp=self.current_time
            ))

    def _get_obs(self):
        """
        Produce partially observable state.
        - Add GPS noise.
        - Obfuscate true severity.
        - Include current resource status.
        """
        noise_level = 2.0 if self.weather == "clear" else 5.0
        
        obs_sos = []
        for e in self.events:
            if e.status != "resolved":
                noisy_loc = [
                    e.location[0] + random.gauss(0, noise_level),
                    e.location[1] + random.gauss(0, noise_level)
                ]
                # Severity estimate: 0 (Low: 1-3), 1 (Med: 4-7), 2 (High: 8-10)
                sev_est = 0 if e.true_severity <= 3 else (1 if e.true_severity <= 7 else 2)
                
                obs_sos.append({
                    "id": e.id,
                    "description": e.description,
                    "location": noisy_loc,
                    "severity_estimate": sev_est
                })
        
        obs_resources = [r.dict() for r in self.resources]
        
        return {
            "incoming_sos": obs_sos,
            "resources": obs_resources,
            "environment": {
                "weather": self.weather,
                "time": self.current_time
            }
        }

    def _get_info(self):
        return {"true_events": [e.dict() for e in self.events]}

    def step(self, action_str: str):
        """
        Action Format Expected: JSON string
        {
            "event_id": "SOS_1",
            "classification": "fire",
            "dispatch": ["FIRE_0"],
            "routing": "shortest"
        }
        """
        try:
            action = json.loads(action_str)
        except:
            # Penalty for malformed JSON handled in reward
            return self._get_obs(), -50.0, False, False, {"error": "Invalid JSON"}

        # Logic for processing action
        target_event_id = action.get("event_id")
        dispatch_ids = action.get("dispatch", [])
        classification = action.get("classification")
        
        reward = 0.0
        
        # Find target event
        event = next((e for e in self.events if e.id == target_event_id), None)
        if not event:
            return self._get_obs(), -10.0, False, False, {"error": "Event not found"}

        # 1. Decision Accuracy (+20/-15)
        if classification == event.category:
            reward += 20.0
        else:
            reward -= 15.0

        # 2. Response Effectiveness
        # Check if dispatched resources match category needs
        match_reward = 0
        for rid in dispatch_ids:
            res = next((r for r in self.resources if r.id == rid), None)
            if res and res.status == "free":
                # Check category compatibility
                compatible = (
                    (event.category == "fire" and res.type == "fire_truck") or
                    (event.category == "medical" and res.type == "ambulance") or
                    (event.category == "disaster" and res.type in ["fire_truck", "rescue_team", "police_car"]) or
                    (event.category == "false_alarm" and res.type == "police_car")
                )
                if compatible:
                    match_reward += 25
                    res.status = "busy"
                    event.assigned_resources.append(rid)
                    event.status = "active"
                else:
                    match_reward -= 20
            else:
                match_reward -= 10 # Busy or missing resource
        reward += match_reward

        # Transition Dynamics: Advance time
        self.current_time += 5 # Advance by 5 minutes
        
        # Update Event Progress
        for e in self.events:
            if e.status == "active":
                # Resolution logic: if enough resources, resolve
                # Scale by severity
                active_res_count = len(e.assigned_resources)
                if active_res_count >= (e.true_severity / 3):
                    e.status = "resolved"
                    reward += 30.0 # Safety Outcome reward
                    # Free up resources
                    for rid in e.assigned_resources:
                        res = next((r for r in self.resources if r.id == rid), None)
                        if res: res.status = "free"
        
        # Spawn new events randomly
        if random.random() < 0.2:
            self._spawn_event()

        # Termination?
        # Fixed horizon of 100 steps or similar
        terminated = self.current_time > 500 
        
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        print(f"Time: {self.current_time} | Weather: {self.weather}")
        print(f"Active Events: {[e.id for e in self.events if e.status != 'resolved']}")
        print(f"Busy Resources: {[r.id for r in self.resources if r.status == 'busy']}")
