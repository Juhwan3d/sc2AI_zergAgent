from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import random
import numpy as np
import pandas as pd
import os


class ZergAgent(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "harvest_gas",
               "build_spawning_pool",
               "train_overload",
               "train_drone",
               "train_zergling",
               "attack")

    def __init__(self):
        super(ZergAgent, self).__init__()

        self.attack_coordinates = None

    # 원하는 unit_type이 선택되어 있는가?
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    # 원하는 unit_type을 선택함
    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def built_by_type(self, obs, unit_type, built_only=True):
        if built_only and len(self.get_my_units_by_type(obs, unit_type)) != 0:
            return actions.FUNCTIONS.no_op()

        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        built_xy = 

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    # action을 할 수 있는가?
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ZergAgent, self).step(obs)
        # 공격 지점 확인
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        # 공격
        zerglings = self.get_my_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) >= 30:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        # 스포닝 풀 건설
        spawning_pools = self.get_my_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

            drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)

                return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
                                                                          drone.y))
        # 라바 선택 시 동작 (모든 생산)
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            # 오버로드 생산 (인구수 부족 시)
            free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

            if free_supply <= 2:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

            # 저글링 생산
            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")

        # 모든 라바 선택 (화면 상)
        larvae = self.get_my_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)

            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))

        return actions.FUNCTIONS.no_op()

    def do_nothing(self, obs):
        return actions.FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
        idle_drones = [drone for drone in drones if drone.order_length == 0]
        if len(idle_drones) > 0:
            mineral_patches = [unit for unit in obs.observation.feature_units
                               if unit.unit_type in [units.Neutral.BattleStationMineralField
                                                     units.Neutral.BattleStationMineralField750,
                                                     units.Neutral.LabMineralField,
                                                     units.Neutral.LabMineralField750,
                                                     units.Neutral.MineralField,
                                                     units.Neutral.MineralField750,
                                                     units.Neutral.PurifierMineralField,
                                                     units.Neutral.PurifierMineralField750,
                                                     units.Neutral.PurifierRichMineralField,
                                                     units.Neutral.PurifierRichMineralField750,
                                                     units.Neutral.RichMineralField,
                                                     units.Neutral.RichMineralField750
                                                     ]]
            drone = random.choice(idle_drones)
            distances = self.get_distances(obs, mineral_patches, (drone.x, drone.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.FUNCTIONS.Harvest_Gather_unit("now", drone.tag, mineral_patch.tag)
        return actions.FUNCTIONS.no_op()

    def

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))


def main(unused_argv):
    agent1 = ZergAgent()
    agent2 = ZergAgent()

    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="AbyssalReef",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
