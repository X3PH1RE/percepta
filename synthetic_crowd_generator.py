"""
Synthetic Crowd Data Generator
Uses Social Force Model to generate realistic crowd movement data for training.
Supports both normal and panic/high-density scenarios.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class BehaviorMode(Enum):
    NORMAL = "normal"
    HIGH_DENSITY = "high_density"
    PANIC = "panic"
    EVACUATION = "evacuation"


@dataclass
class Agent:
    """Individual agent in the simulation"""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    desired_speed: float = 1.4  # m/s typical walking speed
    mass: float = 80.0  # kg
    radius: float = 0.3  # m (personal space)
    goal_x: float = 0.0
    goal_y: float = 0.0
    
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


class SocialForceModel:
    """
    Social Force Model for pedestrian dynamics.
    Based on Helbing & Molnar (1995) with extensions for panic behavior.
    """
    
    def __init__(self, 
                 width: float = 100.0,  # meters
                 height: float = 100.0,
                 dt: float = 0.1,  # time step in seconds
                 tau: float = 0.5,  # relaxation time
                 A: float = 2000.0,  # repulsion strength
                 B: float = 0.08,  # repulsion range
                 k: float = 120000.0,  # body force coefficient
                 kappa: float = 240000.0):  # sliding friction
        
        self.width = width
        self.height = height
        self.dt = dt
        self.tau = tau
        self.A = A
        self.B = B
        self.k = k
        self.kappa = kappa
        
        # Panic parameters
        self.panic_level = 0.0  # 0 = calm, 1 = full panic
        
        # Obstacles (walls)
        self.walls = [
            # Format: (x1, y1, x2, y2) - line segments
            (0, 0, width, 0),  # bottom
            (0, 0, 0, height),  # left
            (width, 0, width, height),  # right
            (0, height, width, height),  # top
        ]
    
    def set_panic_level(self, level: float):
        """Set panic level (0 = calm, 1 = full panic)"""
        self.panic_level = np.clip(level, 0.0, 1.0)
    
    def desired_force(self, agent: Agent) -> np.ndarray:
        """Force towards desired velocity/goal"""
        goal_dir = np.array([agent.goal_x - agent.x, agent.goal_y - agent.y])
        dist_to_goal = np.linalg.norm(goal_dir)
        
        if dist_to_goal < 0.1:
            return np.array([0.0, 0.0])
        
        goal_dir = goal_dir / dist_to_goal
        
        # In panic, desired speed increases
        desired_speed = agent.desired_speed * (1 + 0.5 * self.panic_level)
        desired_velocity = goal_dir * desired_speed
        
        current_velocity = agent.velocity()
        
        return agent.mass * (desired_velocity - current_velocity) / self.tau
    
    def social_force(self, agent: Agent, other: Agent) -> np.ndarray:
        """Repulsion force from another agent"""
        r_ij = agent.radius + other.radius
        d_ij = np.array([agent.x - other.x, agent.y - other.y])
        dist = np.linalg.norm(d_ij)
        
        if dist < 0.01:  # Avoid division by zero
            return np.array([0.0, 0.0])
        
        n_ij = d_ij / dist  # normal direction
        
        # Psychological repulsion
        f_rep = self.A * np.exp((r_ij - dist) / self.B) * n_ij
        
        # Physical contact forces (if overlapping)
        if dist < r_ij:
            # Body force
            f_body = self.k * (r_ij - dist) * n_ij
            
            # Tangential (friction) force
            t_ij = np.array([-n_ij[1], n_ij[0]])  # perpendicular
            delta_v = np.dot(other.velocity() - agent.velocity(), t_ij)
            f_friction = self.kappa * (r_ij - dist) * delta_v * t_ij
            
            f_rep = f_rep + f_body + f_friction
        
        # In panic, social forces are reduced (people ignore personal space)
        f_rep = f_rep * (1 - 0.5 * self.panic_level)
        
        return f_rep
    
    def wall_force(self, agent: Agent) -> np.ndarray:
        """Repulsion force from walls/boundaries"""
        total_force = np.array([0.0, 0.0])
        
        for wall in self.walls:
            x1, y1, x2, y2 = wall
            
            # Find closest point on wall
            wall_vec = np.array([x2 - x1, y2 - y1])
            wall_len = np.linalg.norm(wall_vec)
            if wall_len < 0.01:
                continue
            wall_unit = wall_vec / wall_len
            
            agent_to_wall = np.array([x1 - agent.x, y1 - agent.y])
            proj_length = -np.dot(agent_to_wall, wall_unit)
            proj_length = np.clip(proj_length, 0, wall_len)
            
            closest_point = np.array([x1, y1]) + wall_unit * proj_length
            
            # Direction away from wall
            d_iw = agent.position() - closest_point
            dist = np.linalg.norm(d_iw)
            
            if dist < 0.01:
                continue
            
            n_iw = d_iw / dist
            
            # Repulsion
            f_rep = self.A * np.exp((agent.radius - dist) / self.B) * n_iw
            
            # Physical contact
            if dist < agent.radius:
                f_rep = f_rep + self.k * (agent.radius - dist) * n_iw
            
            total_force = total_force + f_rep
        
        return total_force
    
    def random_fluctuation(self, agent: Agent) -> np.ndarray:
        """Random force for realistic variation"""
        magnitude = 0.1 * agent.mass * (1 + self.panic_level)
        return np.random.randn(2) * magnitude
    
    def compute_total_force(self, agent: Agent, all_agents: List[Agent]) -> np.ndarray:
        """Compute total force on an agent"""
        f_total = np.array([0.0, 0.0])
        
        # Desired direction force
        f_total = f_total + self.desired_force(agent)
        
        # Social forces from other agents
        for other in all_agents:
            if other.id != agent.id:
                f_total = f_total + self.social_force(agent, other)
        
        # Wall forces
        f_total = f_total + self.wall_force(agent)
        
        # Random fluctuation
        f_total = f_total + self.random_fluctuation(agent)
        
        return f_total
    
    def update_agent(self, agent: Agent, force: np.ndarray):
        """Update agent position and velocity"""
        # Acceleration
        a = force / agent.mass
        
        # Update velocity
        agent.vx += a[0] * self.dt
        agent.vy += a[1] * self.dt
        
        # Limit velocity
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        max_speed = agent.desired_speed * 2 * (1 + self.panic_level)
        if speed > max_speed:
            agent.vx = agent.vx / speed * max_speed
            agent.vy = agent.vy / speed * max_speed
        
        # Update position
        agent.x += agent.vx * self.dt
        agent.y += agent.vy * self.dt
        
        # Boundary constraints
        agent.x = np.clip(agent.x, agent.radius, self.width - agent.radius)
        agent.y = np.clip(agent.y, agent.radius, self.height - agent.radius)


class SyntheticCrowdGenerator:
    """
    Generates synthetic crowd movement data for training.
    """
    
    def __init__(self, 
                 width: float = 1920.0,  # pixels (matching video resolution)
                 height: float = 1080.0,
                 scale: float = 0.05):  # pixels to meters conversion
        
        self.width = width
        self.height = height
        self.scale = scale  # Convert simulation to pixel space
        
        # Simulation in meters
        self.sim_width = width * scale
        self.sim_height = height * scale
        
        self.model = SocialForceModel(
            width=self.sim_width,
            height=self.sim_height,
            dt=0.033  # ~30 fps
        )
        
        self.agents: List[Agent] = []
        self.time = 0.0
        self.history: Dict[int, List[Dict]] = {}
    
    def to_pixels(self, x: float, y: float) -> Tuple[float, float]:
        """Convert simulation coordinates to pixels"""
        return x / self.scale, y / self.scale
    
    def from_pixels(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to simulation"""
        return px * self.scale, py * self.scale
    
    def spawn_agents(self, num_agents: int, 
                     spawn_pattern: str = "random",
                     goal_pattern: str = "random"):
        """
        Spawn agents in the simulation.
        
        spawn_pattern: "random", "edges", "cluster", "grid"
        goal_pattern: "random", "exit", "center", "opposite"
        """
        self.agents = []
        self.history = {}
        
        for i in range(num_agents):
            # Spawn position
            if spawn_pattern == "random":
                x = np.random.uniform(1, self.sim_width - 1)
                y = np.random.uniform(1, self.sim_height - 1)
            elif spawn_pattern == "edges":
                side = np.random.randint(4)
                if side == 0:  # top
                    x = np.random.uniform(1, self.sim_width - 1)
                    y = self.sim_height - 1
                elif side == 1:  # bottom
                    x = np.random.uniform(1, self.sim_width - 1)
                    y = 1
                elif side == 2:  # left
                    x = 1
                    y = np.random.uniform(1, self.sim_height - 1)
                else:  # right
                    x = self.sim_width - 1
                    y = np.random.uniform(1, self.sim_height - 1)
            elif spawn_pattern == "cluster":
                center_x = np.random.uniform(self.sim_width * 0.3, self.sim_width * 0.7)
                center_y = np.random.uniform(self.sim_height * 0.3, self.sim_height * 0.7)
                x = np.clip(np.random.normal(center_x, self.sim_width * 0.1), 
                           1, self.sim_width - 1)
                y = np.clip(np.random.normal(center_y, self.sim_height * 0.1), 
                           1, self.sim_height - 1)
            elif spawn_pattern == "grid":
                cols = int(np.sqrt(num_agents * self.sim_width / self.sim_height))
                rows = int(np.ceil(num_agents / cols))
                idx_x = i % cols
                idx_y = i // cols
                x = (idx_x + 0.5) * self.sim_width / cols
                y = (idx_y + 0.5) * self.sim_height / rows
            else:
                x = np.random.uniform(1, self.sim_width - 1)
                y = np.random.uniform(1, self.sim_height - 1)
            
            # Goal position
            if goal_pattern == "random":
                goal_x = np.random.uniform(0, self.sim_width)
                goal_y = np.random.uniform(0, self.sim_height)
            elif goal_pattern == "exit":
                # Random exit point on edge
                side = np.random.randint(4)
                if side == 0:
                    goal_x, goal_y = np.random.uniform(0, self.sim_width), 0
                elif side == 1:
                    goal_x, goal_y = np.random.uniform(0, self.sim_width), self.sim_height
                elif side == 2:
                    goal_x, goal_y = 0, np.random.uniform(0, self.sim_height)
                else:
                    goal_x, goal_y = self.sim_width, np.random.uniform(0, self.sim_height)
            elif goal_pattern == "center":
                goal_x = self.sim_width / 2 + np.random.normal(0, self.sim_width * 0.05)
                goal_y = self.sim_height / 2 + np.random.normal(0, self.sim_height * 0.05)
            elif goal_pattern == "opposite":
                goal_x = self.sim_width - x
                goal_y = self.sim_height - y
            else:
                goal_x = np.random.uniform(0, self.sim_width)
                goal_y = np.random.uniform(0, self.sim_height)
            
            # Randomize agent properties
            agent = Agent(
                id=i,
                x=x,
                y=y,
                desired_speed=np.random.uniform(1.0, 2.0),  # Walking speed variation
                mass=np.random.uniform(60, 100),
                radius=np.random.uniform(0.25, 0.4),
                goal_x=goal_x,
                goal_y=goal_y
            )
            
            self.agents.append(agent)
            self.history[i] = []
    
    def add_obstacle(self, x1: float, y1: float, x2: float, y2: float):
        """Add an obstacle (wall segment) in pixel coordinates"""
        sx1, sy1 = self.from_pixels(x1, y1)
        sx2, sy2 = self.from_pixels(x2, y2)
        self.model.walls.append((sx1, sy1, sx2, sy2))
    
    def step(self):
        """Advance simulation by one time step"""
        # Compute forces
        forces = []
        for agent in self.agents:
            f = self.model.compute_total_force(agent, self.agents)
            forces.append(f)
        
        # Update agents
        for agent, force in zip(self.agents, forces):
            self.model.update_agent(agent, force)
            
            # Record history
            px, py = self.to_pixels(agent.x, agent.y)
            pvx, pvy = agent.vx / self.scale, agent.vy / self.scale
            
            self.history[agent.id].append({
                "x": px,
                "y": py,
                "vx": pvx,
                "vy": pvy,
                "t": self.time
            })
        
        self.time += self.model.dt
    
    def run_simulation(self, duration: float, 
                       behavior_mode: BehaviorMode = BehaviorMode.NORMAL,
                       panic_trigger_time: Optional[float] = None) -> Dict:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation time in seconds
            behavior_mode: Type of crowd behavior to simulate
            panic_trigger_time: When panic starts (if applicable)
        
        Returns:
            Dictionary containing all trajectory data
        """
        steps = int(duration / self.model.dt)
        
        # Set initial behavior
        if behavior_mode == BehaviorMode.NORMAL:
            self.model.set_panic_level(0.0)
        elif behavior_mode == BehaviorMode.HIGH_DENSITY:
            self.model.set_panic_level(0.2)
        elif behavior_mode == BehaviorMode.PANIC:
            self.model.set_panic_level(0.0)  # Will increase at trigger
        elif behavior_mode == BehaviorMode.EVACUATION:
            # Set all goals to exits
            for agent in self.agents:
                side = np.random.randint(4)
                if side == 0:
                    agent.goal_x, agent.goal_y = agent.x, 0
                elif side == 1:
                    agent.goal_x, agent.goal_y = agent.x, self.sim_height
                elif side == 2:
                    agent.goal_x, agent.goal_y = 0, agent.y
                else:
                    agent.goal_x, agent.goal_y = self.sim_width, agent.y
            self.model.set_panic_level(0.3)
        
        for step in range(steps):
            # Show progress every 10%
            if step % (steps // 10 + 1) == 0:
                pct = int(step / steps * 100)
                print(f"      Simulating... {pct}%", end="\r", flush=True)
            
            # Handle panic trigger
            if behavior_mode == BehaviorMode.PANIC and panic_trigger_time:
                if self.time >= panic_trigger_time:
                    # Gradual panic increase
                    panic = min(1.0, (self.time - panic_trigger_time) / 5.0)
                    self.model.set_panic_level(panic)
            
            self.step()
            
            # Reassign random goals occasionally for normal behavior
            if behavior_mode == BehaviorMode.NORMAL and step % 100 == 0:
                for agent in self.agents:
                    if np.random.random() < 0.1:  # 10% chance
                        dist_to_goal = np.sqrt((agent.goal_x - agent.x)**2 + 
                                              (agent.goal_y - agent.y)**2)
                        if dist_to_goal < 2.0:  # Reached goal
                            agent.goal_x = np.random.uniform(0, self.sim_width)
                            agent.goal_y = np.random.uniform(0, self.sim_height)
        
        print("      Simulating... 100%", flush=True)
        
        return self.history
    
    def generate_training_data(self, 
                               num_simulations: int = 10,
                               agents_per_sim: int = 50,
                               duration: float = 60.0,
                               behavior_mix: Dict[BehaviorMode, float] = None) -> Dict:
        """
        Generate complete training dataset with multiple simulations.
        
        Args:
            num_simulations: Number of separate simulations
            agents_per_sim: Agents in each simulation
            duration: Duration of each simulation in seconds
            behavior_mix: Distribution of behavior types
        
        Returns:
            Complete dataset with labeled trajectories
        """
        if behavior_mix is None:
            behavior_mix = {
                BehaviorMode.NORMAL: 0.5,
                BehaviorMode.HIGH_DENSITY: 0.2,
                BehaviorMode.PANIC: 0.15,
                BehaviorMode.EVACUATION: 0.15
            }
        
        dataset = {
            "metadata": {
                "num_simulations": num_simulations,
                "agents_per_sim": agents_per_sim,
                "duration": duration,
                "width": self.width,
                "height": self.height
            },
            "simulations": []
        }
        
        # Create simulations based on behavior mix
        behaviors = list(behavior_mix.keys())
        probs = list(behavior_mix.values())
        
        for sim_idx in range(num_simulations):
            # Choose behavior mode
            mode = np.random.choice(behaviors, p=probs)
            
            # Choose spawn pattern based on mode
            if mode == BehaviorMode.HIGH_DENSITY:
                spawn = "cluster"
                goal = "random"
            elif mode == BehaviorMode.EVACUATION:
                spawn = "random"
                goal = "exit"
            elif mode == BehaviorMode.PANIC:
                spawn = "random"
                goal = "exit"
            else:
                spawn = np.random.choice(["random", "edges", "grid"])
                goal = np.random.choice(["random", "opposite", "center"])
            
            # Run simulation
            self.spawn_agents(agents_per_sim, spawn_pattern=spawn, goal_pattern=goal)
            
            panic_trigger = None
            if mode == BehaviorMode.PANIC:
                panic_trigger = np.random.uniform(5.0, duration / 2)
            
            trajectories = self.run_simulation(
                duration=duration,
                behavior_mode=mode,
                panic_trigger_time=panic_trigger
            )
            
            dataset["simulations"].append({
                "simulation_id": sim_idx,
                "behavior_mode": mode.value,
                "spawn_pattern": spawn,
                "goal_pattern": goal,
                "panic_trigger_time": panic_trigger,
                "trajectories": trajectories
            })
            
            print(f"   Generated simulation {sim_idx + 1}/{num_simulations} "
                  f"({mode.value}, {len(trajectories)} agents)", flush=True)
            
            # Reset for next simulation
            self.time = 0.0
        
        return dataset
    
    def save_dataset(self, dataset: Dict, filepath: str):
        """Save dataset to JSON file"""
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(dataset), f)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict:
        """Load dataset from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_current_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get current positions in pixel coordinates (for visualization)"""
        positions = {}
        for agent in self.agents:
            px, py = self.to_pixels(agent.x, agent.y)
            positions[agent.id] = (px, py)
        return positions


# Utility function to prepare data for PyTorch
def prepare_pytorch_data(dataset: Dict, 
                         seq_length: int = 30,
                         future_steps: int = 60,  # ~2 seconds at 30fps
                         normalize: bool = True,
                         max_samples_per_trajectory: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for PyTorch training.
    
    Returns:
        X: Input sequences [num_samples, seq_length, features]
        Y: Target positions [num_samples, future_steps, 2]
        labels: Behavior mode labels [num_samples]
    """
    width = dataset["metadata"]["width"]
    height = dataset["metadata"]["height"]
    
    X_list = []
    Y_list = []
    labels_list = []
    
    label_map = {
        "normal": 0,
        "high_density": 1,
        "panic": 2,
        "evacuation": 3
    }
    
    num_sims = len(dataset["simulations"])
    
    for sim_idx, sim in enumerate(dataset["simulations"]):
        print(f"   Processing simulation {sim_idx + 1}/{num_sims}...", end="\r", flush=True)
        
        label = label_map.get(sim["behavior_mode"], 0)
        
        for agent_id, trajectory in sim["trajectories"].items():
            if len(trajectory) < seq_length + future_steps:
                continue
            
            # Limit samples per trajectory to avoid memory issues
            total_windows = len(trajectory) - seq_length - future_steps + 1
            step = max(1, total_windows // max_samples_per_trajectory)
            
            # Sliding window over trajectory (with step)
            for i in range(0, total_windows, step):
                # Input sequence
                x_seq = []
                for j in range(seq_length):
                    point = trajectory[i + j]
                    x = point["x"] / width if normalize else point["x"]
                    y = point["y"] / height if normalize else point["y"]
                    vx = point.get("vx", 0) / width if normalize else point.get("vx", 0)
                    vy = point.get("vy", 0) / height if normalize else point.get("vy", 0)
                    speed = np.sqrt(vx**2 + vy**2)
                    direction = np.arctan2(vy, vx)
                    x_seq.append([x, y, vx, vy, speed, direction])
                
                # Target sequence
                y_seq = []
                for j in range(future_steps):
                    point = trajectory[i + seq_length + j]
                    x = point["x"] / width if normalize else point["x"]
                    y = point["y"] / height if normalize else point["y"]
                    y_seq.append([x, y])
                
                X_list.append(x_seq)
                Y_list.append(y_seq)
                labels_list.append(label)
    
    print(f"   Processed {num_sims}/{num_sims} simulations.    ")
    
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(X_list, dtype=np.float32), \
           np.array(Y_list, dtype=np.float32), \
           np.array(labels_list, dtype=np.int64)


if __name__ == "__main__":
    # Example: Generate training data
    print("Generating synthetic crowd data...")
    
    generator = SyntheticCrowdGenerator(width=1920, height=1080)
    
    # Generate dataset
    dataset = generator.generate_training_data(
        num_simulations=5,
        agents_per_sim=30,
        duration=30.0
    )
    
    # Save dataset
    generator.save_dataset(dataset, "synthetic_crowd_data.json")
    
    # Prepare for PyTorch
    X, Y, labels = prepare_pytorch_data(dataset, seq_length=30, future_steps=60)
    print(f"\nPrepared training data:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
