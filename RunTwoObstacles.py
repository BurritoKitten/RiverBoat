

from src.Environment import Environment

if __name__ == '__main__':

    env = Environment.create_environment("scenario_parameters.yaml")

    env.launch_training()