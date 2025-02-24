from gym.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from env.gym_openAI.simulator_gym import GymEnv

"""
Current supported space class:
Box: gym.spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)  -> Box(3,4)
     gym.spaces.Box(low=np.array([-1.0, -2.0]).astype(np.float32), high=np.array([2.0, 4.0]).astype(np.float32), dtype=np.float32)  -> Box(2,)
Dict: self.observation_space = gym.spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
Discrete: a = gym.spaces.Discrete(n=3, start=-1), a.sample() -> one output
MultiBinary: An n-shape binary space. 
             self.observation_space = spaces.MultiBinary(n=5)
             self.observation_space.sample() -> array([0, 1, 0, 1, 0], dtype=int8)
MultiDiscrete: The multi-discrete action space consists of a series of discrete action spaces with different number of actions in each
             gym.spaces.MultiDiscrete(nvec=[ 5, 2, 2 ], dtype=np.int64)
"""

# todo: consider 2d observation input, isDiscreteAction: continous +discrete action

def get_space_shape(space):
    """
    Get the size of a given space.

    :param space: a class instance from gym.spaces
    """
    assert(space)
    if isinstance(space, Box):
        shape = space.shape
        if len(shape) == 0:
            return None
        elif len(shape) == 1:
            return shape[0]
        else:
            return shape
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiDiscrete):
        if len(space.nvec) == 0:
            return None
        else:
            return len(space.nvec)
    elif isinstance(space, MultiBinary):
        return space.n
    elif isinstance(space, Dict):
        temp = None
        for i in space.keys():
            item = space[i]
            if temp is None:
                temp = get_space_shape(item)
            elif (isinstance(get_space_shape(item), int)):
                temp += get_space_shape(item)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError


def get_state_num(env) -> int:
    """
    Get the number of state inputs for the policy.
    Used by 'builder.py' to pass the number of input nodes to policy initialization

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        observation_space = env.env.observation_space
    else:
        observation_space = env.observation_space
    return get_space_shape(observation_space)


def get_action_num(env) -> int:
    """
    Get the number of action inputs for the policy.
    Used by 'builder.py' to pass the number of output nodes to policy initialization

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        action_space = env.env.action_space
    else:
        action_space = env.action_space
    return get_space_shape(action_space)


def is_discrete_action(env) -> bool:
    """
    Check if the action is discrete
    Used by 'builder.py' for policy initialization
    Box: np.float32

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        space = env.env.action_space
    else:
        space = env.action_space

    if isinstance(space, Box):
        return False
    elif isinstance(space, Discrete) or isinstance(space, MultiBinary) or isinstance(space, MultiDiscrete):
        return True
    else:
        raise NotImplementedError


def get_nn_output_num(env) -> int:

    if env.name == "WorkflowScheduling-v2":
        from env.workflow_scheduling_v2.simulator_wf import WFEnv
    elif env.name == "WorkflowScheduling-v3":
        from env.workflow_scheduling_v3.simulator_wf import WFEnv

    if isinstance(env, GymEnv):
        space = env.env.action_space
    else:
        space = env.action_space
    if isinstance(env, WFEnv):
        return 1
    elif not is_discrete_action(env):
        return get_space_shape(space)
    else:
        assert (space)
        if isinstance(space, Discrete):
            return space.n
        else:
            raise NotImplementedError

