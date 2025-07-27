#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
import py_trees
import numpy as np

from py_trees.trees import BehaviourTree

from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from minigrid_bt.utils import update_tree_obs, ReconstructObsWrapper, current_action
# from minigrid_bt.env_initialization import initialize_env
from minigrid_bt.main import \
    create_BabyAI_bt, \
    create_BlockedUnlockPickup_bt, \
    create_DoorKey_bt, \
    create_Empty_bt, \
    create_KeyCorridor_bt, \
    create_MultiRoom_bt, \
    create_ObstructedMaze_bt, \
    create_RedBlueDoors_bt, \
    create_Unlock_bt, \
    create_UnlockPickup_bt


from minigrid_bt.utils import ExtendedFlatObsWrapper, ReconstructObsWrapper
from minigrid.wrappers import FullyObsWrapper


class ManualControlBT:
    def __init__(
        self,
        env: Env,
        image_shape: tuple[int,...] | None,
        seed=None,
    ) -> None:
        self.env = env
        self.image_shape = image_shape
        self.seed = seed
        self.closed = False

    def initialize_obs_and_tree(self, initial_obs, tree_creation_func,
                                  reconstruct_obs_wrapper_class=ReconstructObsWrapper,
                                ):
        _observation = initial_obs
        if isinstance(_observation, tuple):
            _observation = _observation[0]
        if _observation.ndim > 1:
            _observation = np.squeeze(_observation)

        self.reconstruct_obs_wrapper = reconstruct_obs_wrapper_class(np.array(self.image_shape))
        self.obs = self.reconstruct_obs_wrapper.reconstruct_observation(_observation)
        self.tree: BehaviourTree = tree_creation_func(self.env, self.obs)
        py_trees.display.render_dot_tree(self.tree.root, target_directory="/tmp")

    def _predict(self, observation, deterministic=False):
        if isinstance(observation, tuple):
            observation = observation[0]
        if observation.ndim > 1:
            observation = np.squeeze(observation)
        update_tree_obs(self.tree, self.reconstruct_obs_wrapper.reconstruct_observation(observation))
        self.tree.tick()
        action = current_action["action"]
        return action


    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.unwrapped.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.predict_next_action()
            self.env.render()

    def predict_next_action(self):
        self._next_action_idx = self._predict(self.obs)
        if self._next_action_idx is not None:
            self.next_action = Actions(self._next_action_idx)
            print(f"BT NEXT ACTION: {repr(self.next_action)}")
        else:
            self.next_action = None
            print(f"BT NEXT ACTION: {self._next_action_idx}")
    
    def reset(self, seed=None):
        self.obs, _ = self.env.reset(seed=seed)
        self.predict_next_action()
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        elif key == "right shift":
            if self.next_action is not None:
                print(f"execute BT ACTION: {repr(self.next_action)}")
                self.step(self.next_action)
            else:
                print("NO NEXT ACTION -> NOOP")
        else:
            print(key)

# List of environment IDs
env_policies = {
    "MiniGrid-ObstructedMaze-1Dlhb-v0": create_ObstructedMaze_bt,    
    "MiniGrid-ObstructedMaze-2Dlhb-v0": create_ObstructedMaze_bt,    
    "MiniGrid-Empty-6x6-v0": create_Empty_bt,
    "MiniGrid-Empty-8x8-v0": create_Empty_bt,
    "MiniGrid-Empty-Random-5x5-v0": create_Empty_bt,
    "MiniGrid-Empty-Random-6x6-v0": create_Empty_bt,
    "BabyAI-GoToRedBallNoDists-v0": create_BabyAI_bt,
    "MiniGrid-DistShift2-v0": None,
    "MiniGrid-LavaGapS7-v0": None,
    "MiniGrid-FourRooms-v0": create_MultiRoom_bt,
    "MiniGrid-MultiRoom-N6-v0": create_MultiRoom_bt,
    "MiniGrid-SimpleCrossingS11N5-v0": None,
    "MiniGrid-LavaCrossingS11N5-v0": None,
    "MiniGrid-Unlock-v0": create_Unlock_bt,
    "MiniGrid-DoorKey-8x8-v0": create_DoorKey_bt,
    "MiniGrid-UnlockPickup-v0": create_UnlockPickup_bt
}



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    ) # type: ignore

 
    # Initialize the environment and capture the image shape
    # env, image_shape = initialize_env(args.env_id) # type: ignore
    env = FullyObsWrapper(env) # type: ignore
    image_shape = env.observation_space['image'].shape if isinstance(env.observation_space, gym.spaces.Dict) else env.observation_space.shape
    env = ExtendedFlatObsWrapper(env) # type: ignore
 
    tree_creation_func = env_policies.get(args.env_id, None)
    if tree_creation_func is None:
        print(f"WARNING: BT policy UNKNOWN for {args.env_id}")
        tree_creation_func = create_Empty_bt  # ?maybe: create_BabyAI_bt
 
    # Get observation space and action space by initializing the environment once
    observation_space = env.observation_space
    action_space = env.action_space

    # Create the behavior tree policy
    # policy = BehaviorTreePolicy(
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     env=env,
    #     image_shape=image_shape,
    #     tree_creation_func=tree_creation_func,
    #     reconstruct_obs_wrapper_class=ReconstructObsWrapper
    # )

    # Reset the environment and get the initial observation
    initial_obs = env.reset()
    manual_control = ManualControlBT(env, image_shape, seed=args.seed)
    manual_control.initialize_obs_and_tree(initial_obs, tree_creation_func)
    manual_control.start()
