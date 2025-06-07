from Qlearning import config
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal
from gymnasium.spaces import Text

class UnlockPickupEnv(MiniGridEnv):
    def __init__(self, width=4, height=4, max_steps=50):
        self.width = width
        self.height = height
        self.mission = "pick up key, open door, reach goal"
        
        # Define action space (0: left, 1: right, 2: forward, 3: pickup, 4: toggle)
        self.action_space = gym.spaces.Discrete(5)
        self._actions = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}  # Map to MiniGrid actions
        
        # Set mission space for newer MiniGrid versions
        mission_space = Text(max_length=100)
        
        # Observation space includes image
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,  # Full observability as in paper
            agent_view_size=7,       # Agent sees a 7x7 square around itself
            mission_space=mission_space
        )
        
        # Track sub-task completion
        self.key_picked = False
        self.door_opened = False
        
        # Track previous positions for reward shaping
        self.prev_key_distance = None
        self.prev_door_distance = None
        self.prev_goal_distance = None

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place agent in the top-left
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Right
        
        # Key position - one corner
        self.key_pos = (width - 2, 1)
        self.grid.set(*self.key_pos, Key("yellow"))
        
        # Door position - opposite corner
        self.door_pos = (width - 2, height - 2)
        self.grid.set(*self.door_pos, Door("yellow", is_locked=True))
        
        # Goal - behind the door
        self.goal_pos = (width - 1, height - 2)
        self.grid.set(*self.goal_pos, Goal())
        
        # Initialize distances
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)

    def step(self, action):
        action = min(max(action, 0), 4)
        # Track previous completion state to detect new completions
        prev_key_picked = self.key_picked
        prev_door_opened = self.door_opened
        was_at_goal = self.agent_pos == self.goal_pos
        
        # Check if pickup/open action is valid BEFORE taking the action
        valid_pickup = False
        valid_open = False
        if action == 3:  # Pickup
            valid_pickup = self._valid_pickup()
        elif action == 4:  # Open door
            valid_open = self._valid_open()
        
        # Map action to MiniGrid's action
        minigrid_action = self._actions[action]
        obs, _, terminated, truncated, info = super().step(minigrid_action)
        
        # Track sub-tasks after taking the action
        if self.carrying and self.carrying.type == "key":
            self.key_picked = True
        
        door_cell = self.grid.get(*self.door_pos)
        if door_cell is None or (isinstance(door_cell, Door) and not door_cell.is_locked):
            self.door_opened = True
        
        # Initialize reward
        reward = 0
        
        # Invalid action penalty - only apply if the action was invalid
        if action == 3 and not valid_pickup:  # Invalid pickup
            reward += config.INVALID_ACTION_PENALTY
            print(f"Invalid pickup attempt! {config.INVALID_ACTION_PENALTY}")
        elif action == 4 and not valid_open:  # Invalid door open
            reward += config.INVALID_ACTION_PENALTY
            print(f"Invalid door open attempt! {config.INVALID_ACTION_PENALTY}")
        
        # Mission rewards - only award when newly completed
        # First mission: Key pickup reward - only if pickup was valid
        if not prev_key_picked and self.key_picked:
            reward += config.KEY_REWARD
            print(f"First mission completed: Key picked up! +{config.KEY_REWARD}")
        
        # Second mission: Door opening reward - only if door open was valid
        if not prev_door_opened and self.door_opened:
            reward += config.DOOR_REWARD
            print(f"Second mission completed: Door opened! +{config.DOOR_REWARD}")
        
        # Third mission: Reaching the goal
        if self.agent_pos == self.goal_pos and self.door_opened and not was_at_goal:
            goal_reward = config.GOAL_REWARD
            time_bonus = (1 - self.step_count / self.max_steps)
            reward += goal_reward + time_bonus
            print(f"Final mission completed: Goal reached! Base: +{goal_reward}, Time bonus: +{time_bonus:.4f}")
            terminated = True
        
        done = terminated or truncated
        return obs, reward, done, truncated, info
    
    def _valid_pickup(self):
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "key"
    
    def _valid_open(self):
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "door"

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)[0]  # Take first element (observation)
        self.key_picked = False
        self.door_opened = False
        
        # Reset distance tracking
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)
        
        return obs