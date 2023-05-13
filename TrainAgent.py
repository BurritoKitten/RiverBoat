

from src.Environment import Environment

if __name__ == '__main__':

    # case 0 - DQN for direct propeller control
    # case 1 - DDPG for direct propeller control
    # case 2 - DQN for path control
    # case 3 - DDPG for path control
    case = 1

    file_name = ''
    if case == 0:
        file_name = 'scenario_parameters_discrete_direct_control_DQN.yaml'
    elif case == 1:
        file_name = 'scenario_parameters_cont_direct_control_DDPG.yaml'
    elif case == 2:
        file_name = 'scenario_parameters_discrete_path_DQN.yaml'
    elif case == 3:
        file_name = 'scenario_parameters_cont_path_DDPG.yaml'
    elif case == 4:
        # not implemented
        file_name = 'scenario_parameters_hierarchical_DQN.yaml'
    elif case == 5:
        file_name = 'scenario_parameters_hierarchical_DDPG.yaml'
    else:
        raise ValueError('Unsupported case number')

    # create environment and launch training
    env = Environment.create_environment(file_name)
    env.launch_training()