import numpy as np

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from configparser import ConfigParser

from hlrl.torch.algos import SAC
from hlrl.core.logger import TensorboardLogger
from hlrl.torch.policies import LinearSAPolicy, TanhGaussianPolicy
from hlrl.torch.experience_replay import TorchPER

# Convert the game packet to a state for model input
def packet_to_state(packet, curr):
    return [1, 0]

class RLAgent(BaseAgent):
    def initialize_agent(self):
        import torch
        self.torch = torch

        self.out = [1, 0]

        self.controller = SimpleControllerState()

        config = ConfigParser()
        config.read("dragonbot/rlbot/config/algo.cfg")

        self.config = config
        self.state_space = [2]
        self.action_space = 8
        self.device = torch.device(config["General"]["device"])

        # The logger
        logs_path = config["General"]["logs_path"]

        self.logger = None if len(logs_path) == 0 else TensorboardLogger(logs_path)

        # Initialize SAC
        activation_fn = nn.ReLU
        qfunc = LinearSAPolicy(*self.state_space, self.action_space, 1,
                               int(config["Model"]["hidden_size"]),
                               int(config["Model"]["num_hidden"]),
                               activation_fn)
        policy = TanhGaussianPolicy(*self.state_space, self.action_space,
                                    int(config["Model"]["hidden_size"]),
                                    int(config["Model"]["num_hidden"]),
                                    activation_fn)

        optim = lambda params: torch.optim.Adam(params,
                                                lr=float(config["Algorithm"]["lr"]))
        self.algo = SAC(self.action_space, qfunc, policy,
                        config["Algorithm"]["discount"],
                        config["Algorithm"]["polyak"],
                        config["Algorithm"]["target_update_interval"], optim,
                        optim, optim, config["Algorithm"]["twin"],
                        self.logger).to(self.device)

        if (len(config["General"]["load_path"]) > 0):
            self.algo.load(config["General"]["load_path"])

        # Initialize replay buffer
        self.experience_replay = TorchPER(int(config["Replay"]["capacity"]),
                                          float(config["Replay"]["alpha"]),
                                          float(config["Replay"]["beta"]),
                                          float(config["Replay"]["beta_increment"]),
                                          float(config["Replay"]["epsilon"]))

    def make_controller_state_from_output(self, model_out):
        return np.concatenate((model_out[:5], (model_out[5:] >= 0)), axis = 0)

    def get_model_prediction(self, state):
        # Get the prediction of the current state
        state = self.torch.FloatTensor([state]).to(self.device)
        action, q_val = self.algo.step(state)

        return action[0].detach().cpu().numpy(), q_val.item()

    def train_step(self):
        experience_replay.add(*experience)

        save_path = None
        if len(self.config["Training"]["save_path"]) > 0:
            save_path = self.config["Training"]["save_path"]

        algo.train_from_buffer(experience_replay,
                               int(self.config["Training"]["batch_size"]),
                               int(self.config["Training"]["start_size"]),
                               save_path,
                               int(self.config["Training"]["save_interval"]))

    def get_output(self, packet) -> SimpleControllerState:
        state = packet_to_state(packet, self.out)
        self.out = self.out[::-1]

        my_car = packet.game_cars[0]
        action, q_val = self.get_model_prediction(state)
        new_controller_state = self.make_controller_state_from_output(action)

        self.controller.throttle = new_controller_state[0]
        self.controller.steer = new_controller_state[1]
        self.controller.pitch = new_controller_state[2]
        self.controller.yaw = new_controller_state[3]
        self.controller.roll = new_controller_state[4]
        self.controller.jump = new_controller_state[5]
        self.controller.boost = new_controller_state[6]
        self.controller.handbrake = new_controller_state[7]

        draw_debug(self.renderer, my_car, packet.game_ball, "Go forward")
        return self.controller

def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()