from copy import deepcopy


def agent_policy(agent_ids, policy):
    group = {}
    for agent_id in agent_ids:
        group[agent_id] = deepcopy(policy)
    return group
