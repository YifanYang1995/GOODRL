import gym


class GymEnv:
    def __init__(self, name, args):
        self.env = gym.make(name)

        self.name = name
        self.step_curr = 0
        self.step_max = args['step_max']

        if args['pomdp']:
            if "LunarLander" in name:
                self.env = LunarLanderPOMDP(self.env)
            elif "CartPole" in name:
                self.env = CartPolePOMDP(self.env)
            else:
                raise AssertionError(f"{name} doesn't support POMDP.")

    def reset(self, seed=None):
        self.env.action_space.seed(seed)
        self.env.seed(seed)
        self.step_curr = 0
        state_dict_list = {}
        state_dict = {}
        s = self.env.reset()
        self.env._max_episode_steps = self.step_max
        state_dict["state"] = s
        state_dict_list["0"] = state_dict
        return state_dict_list

    def step(self, action):
        self.step_curr += 1

        state_dict_list = {}
        state_dict = {}
        s, r, d, info = self.env.step(action["0"])
        if self.step_max != "None":
            if self.step_curr >= self.step_max or d:
                d = True
        state_dict["state"] = s
        state_dict["reward"] = r
        state_dict["done"] = d
        state_dict["info"] = info
        state_dict_list["0"] = state_dict
        return state_dict_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class LunarLanderPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # obs[2] = 0
        # obs[3] = 0
        # obs[5] = 0

        # return  modified observation
        return obs


class CartPolePOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        #obs[1] = 0
        #obs[3] = 0
        return obs
