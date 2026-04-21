import random
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from utils.helpers import calculate_distance

class Resource(BaseModel):
    id: str
    type: str  # 'ambulance', 'fire_truck', 'police_car', 'rescue_team'
    location: List[float]
    status: str  # 'free', 'busy', 'returning'
    speed: float = 5.0
    energy: float = 100.0  # Theme #3: Resource Fatigue
    target_event_id: Optional[str] = None

class SOSEvent(BaseModel):
    id: str
    description: str
    category: str
    true_severity: float
    latent_spread_rate: float = 0.05  # Theme #3: Latent Variable
    location: List[float]
    status: str  # 'pending', 'active', 'resolved', 'failed'
    creation_time: int
    resolution_progress: float = 0.0

class WorldSimulator:
    """
    Upgraded Simulator for Theme #3: World Modeling.
    Includes persistent state and causal dependencies.
    """
    
    def __init__(self, grid_size: float = 100.0):
        self.grid_size = grid_size
        self.city_zones = self._init_city_zones()
        self.reset()

    def _init_city_zones(self):
        """Hidden world variables for Theme #3"""
        return {
            "Industrial": {"spread_multiplier": 1.5, "risk": "high"},
            "Residential": {"spread_multiplier": 1.0, "risk": "medium"},
            "Commercial": {"spread_multiplier": 0.8, "risk": "low"},
        }

    def reset(self):
        self.current_time = 0
        self.weather = random.choice(["clear", "rainy", "stormy", "foggy"])
        self.road_conditions = self._generate_road_conditions()
        
        self.resources = self._init_resources()
        self.events = []
        self._spawn_events(count=2)
        
        return self.get_state()

    def _init_resources(self) -> List[Resource]:
        res = []
        counts = {"ambulance": 3, "fire_truck": 2, "police_car": 3, "rescue_team": 2}
        for r_type, count in counts.items():
            for i in range(count):
                res.append(Resource(
                    id=f"{r_type.upper()}_{i}",
                    type=r_type,
                    location=[random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)],
                    status="free",
                    energy=100.0
                ))
        return res

    def _generate_road_conditions(self):
        return {"main_road": 1.0, "highways": 1.0 if self.weather == "clear" else 0.7}

    def _spawn_events(self, count=1):
        for _ in range(count):
            eid = f"SOS_{len(self.events) + 1}"
            category = random.choice(["fire", "medical", "disaster", "false_alarm"])
            severity = random.randint(1, 10)
            location = [random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)]
            
            # Determine Zone
            zone = "Industrial" if location[0] > 60 else ("Residential" if location[1] < 50 else "Commercial")
            spread = self.city_zones[zone]["spread_multiplier"] * 0.1
            
            desc_map = {
                "fire": f"Building fire in {zone} zone.",
                "medical": "Emergency medical request.",
                "disaster": f"Disaster event at {zone} junction.",
                "false_alarm": "Sensor alert (Potential false alarm)."
            }
            
            self.events.append(SOSEvent(
                id=eid,
                description=desc_map[category],
                category=category,
                true_severity=severity,
                latent_spread_rate=spread,
                location=location,
                status="pending",
                creation_time=self.current_time
            ))

    def update(self, action: Dict[str, Any]):
        # 1. Dispatch
        target_eid = action.get("event_id")
        dispatch_ids = action.get("dispatch", [])
        
        event = next((e for e in self.events if e.id == target_eid), None)
        if event and event.status in ["pending", "active"]:
            for rid in dispatch_ids:
                res = next((r for r in self.resources if r.id == rid), None)
                if res and res.status == "free" and res.energy > 10:
                    res.status = "busy"
                    res.target_event_id = target_eid
                    event.status = "active"

        # 2. Physics & Fatigue
        self.current_time += 1
        weather_penalty = 0.5 if self.weather == "stormy" else 1.0
        
        for res in self.resources:
            if res.status == "busy" and res.target_event_id:
                target_ev = next((e for e in self.events if e.id == res.target_event_id), None)
                if target_ev:
                    dist = calculate_distance(res.location, target_ev.location)
                    if dist > 0.5:
                        move_dist = res.speed * weather_penalty * (res.energy / 100.0)
                        direction = [(target_ev.location[0]-res.location[0])/dist, (target_ev.location[1]-res.location[1])/dist]
                        res.location[0] += direction[0] * min(move_dist, dist)
                        res.location[1] += direction[1] * min(move_dist, dist)
                        res.energy -= 0.5 # Fatigue from travel
                    else:
                        target_ev.resolution_progress += (0.2 * (res.energy / 100.0))
                        res.energy -= 1.0 # Fatigue from work
            elif res.status == "free" and res.energy < 100:
                res.energy = min(100.0, res.energy + 2.0) # Recharge

        # 3. Resolution & Event Persistence (Causal growth)
        for e in self.events:
            if e.status == "active":
                if e.resolution_progress >= (e.true_severity * 0.1):
                    e.status = "resolved"
                    for res in self.resources:
                        if res.target_event_id == e.id:
                            res.status = "free"
                            res.target_event_id = None
                else:
                    # Causal: If not solved, it grows
                    e.resolution_progress -= e.latent_spread_rate 
            elif e.status == "pending":
                # Growth while ignored
                e.true_severity += 0.05 
                if (self.current_time - e.creation_time) > 100:
                    e.status = "failed"

        # 4. Stochastic
        if random.random() < 0.15:
            self._spawn_events()
        
        if self.current_time % 20 == 0:
            self.weather = random.choice(["clear", "rainy", "stormy", "foggy"])

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.current_time,
            "weather": self.weather,
            "resources": [r.dict() for r in self.resources],
            "events": [e.dict() for e in self.events],
            "road_conditions": self.road_conditions
        }
