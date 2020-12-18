from typing import Tuple

import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from dqn.HunterEnv import HunterEnv
from dqn.PreyEnv import PreyEnv


class MultiAgentSimEnv(MultiAgentEnv):
    def __init__(self, config):
        self.config = {
                'hunters': {
                    'start_amount': 20,
                    'energy_to_reproduce': 30,
                    'energy_per_prey_eaten': 10,
                    'max_age': 20, },
                'preys': {
                    'start_amount': 100,
                    'birth_rate': 17,
                    'max_age': 20},
                'sim': {
                    'width': 200,
                    'height': 200}
        }
        # Hunters
        hunter_config = self.config['hunters']
        self.hunter_count = hunter_config['start_amount']
        self.agents = {}
        for i in range(hunter_config['start_amount']):
            self.agents['hunter_' + str(i)] = HunterEnv(self.config)
        self.dones = []
        if self.hunter_count > 0:
            self.observation_space_hunter = self.agents['hunter_0'].observation_space
            self.observation_space = self.observation_space_hunter
            self.action_space_hunter = self.agents['hunter_0'].action_space
            self.action_space = self.action_space_hunter
            self.action_shape = self.agents['hunter_0'].action_space.n
        self.alive = 0

        prey_config = self.config['preys']
        self.prey_count = prey_config['start_amount']

        for i in range(prey_config['start_amount']):
            self.agents['prey_' + str(i)] = PreyEnv(self.config)
        self.dones = []
        if self.prey_count>0:
            self.observation_space_prey = self.agents['prey_0'].observation_space
            self.action_space_prey = self.agents['prey_0'].action_space
            self.action_shape = self.agents['prey_0'].action_space.n
        self.alive = 0

    def reset(self) -> MultiAgentDict:
        self.dones = []
        obs_batch = {}
        print(self.agents)
        for i, a in self.agents.items():
            obs_batch[i] = a.reset()
        return obs_batch

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observation, reward, done, reproduce = {}, {}, {}, {}
        alive = []
        for id, action in action_dict.items():
            # i = int(id.split('_')[1])

            if not id in self.dones:
                observation[id], reward[id], done[id], reproduce[id] = self.agents[id].step(action)
                if done[id]:
                    self.dones.append(id)
                alive.append(id)
                # else:
                #     observation[id], reward[id], done[id], reproduce[id] = self.prey_agents[id].step(action)
                #     if done[id]:
                #         self.dones.append(id)
                #     alive.append(id)

        for id in alive:
            # print("len", observation, action_dict[0], reward)
            if not id in self.dones:
                if reproduce[id]:
                    if "hunter" in id:
                        self.hunter_count += 1
                        new_agent = HunterEnv(self.config)
                        new_id = "hunter_" + str(self.hunter_count)
                    else:
                        self.prey_count += 1
                        new_agent = PreyEnv(self.config)
                        new_id = "prey_" + str(self.prey_count)

                    observation[new_id] = new_agent.reset()
                    reward[new_id] = 0
                    done[new_id] = False
                    reproduce[new_id] = False
                    self.agents[new_id] = new_agent
        done["__all__"] = len(self.dones) == len(self.agents)
        # print(observation)
        self.alive = len(observation)
        return observation, reward, done, reproduce
