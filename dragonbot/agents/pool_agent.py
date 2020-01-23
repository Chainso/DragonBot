import torch
import numpy as np

from torch.multiprocessing import Pipe
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.botmanager.helper_process_request import HelperProcessRequest
from collections import deque
from time import time, sleep

from dragonbot.game.input import RecurrentInputFormatter
from dragonbot.game.output import RecurrentOutputFormatter

class PoolAgent(BaseAgent):
    """
    An agent that exists in a pool with other agents.
    """
    FPS = 30.0

    def initialize_agent(self):
        (self.config, self.device, self.model, self.experience_buffer,
         env) = self.pipe.recv()

        self.input_formatter = RecurrentInputFormatter(self.team, self.index,
                                                       self.device)
        self.output_formatter = RecurrentOutputFormatter()

        #self.env = env(self.team, self.index)

        self.empty_action = torch.FloatTensor([[[0] * self.output_formatter.action_space()]])
        self.empty_action = self.empty_action.to(self.device)

        self.hidden_state = self.model.reset_hidden_state()
        self.action = self.empty_action

        # Training values
        self.train = self.config["Training"].getboolean("train")

        if self.train:
            self.decay = float(self.config["Algorithm"]["discount"])
            self.sequence_length = int(self.config["Algorithm"]["sequence_length"])
            self.burn_in_length = int(self.config["Algorithm"]["burn_in_length"])

            self.experiences = deque(maxlen=int(self.config["Training"]["n_steps"]))
            self.ready_experiences = []
            self.q_vals = []
            self.target_qs = []
            self.hidden_states = deque(maxlen=(1 + int(np.ceil(self.burn_in_length
                                                               / self.sequence_length))))

            self.state = None
            self.reward = 0
            self.terminal = torch.FloatTensor([[[False]]]).to(self.device)
            self.last_action = self.empty_action

        #self.reset()

    def reset(self):
        self.set_game_state(self.env.reset())

    def train_batch(self, experience, q_val, next_q):
        batch_size = int(self.config["Training"]["batch_size"])
        start_size = int(self.config["Training"]["start_size"])

        save_path = self.config["General"]["save_path"]
        save_path = None if len(save_path) == 0 else save_path

        save_interval = int(self.config["Training"]["save_interval"])

        self.experience_buffer.add(experience, q_val, next_q)

        self.model.train_from_buffer(self.experience_buffer, batch_size,
                                        start_size, save_path, save_interval)

    def retire(self):
        # Currently doesn't really do anything
        print("Retired!")
        self.terminal = 1 - self.terminal

        # Add remaining experiences
        for i in range(len(self.experiences)):
            self._get_buffer_experience()

            if (len(self.ready_experiences) == self.sequence_length
                                               + self.burn_in_length):
                self.add_experience()

        if (len(self.ready_experiences) > self.burn_in_length):
            self.add_experience()

    def _n_step_decay(self):
        """
        Perform n-step decay on experiences.
        """
        reward = 0
        for experience in list(self.experiences)[::-1]:
            reward = experience[0][2] + self.decay * reward

        return reward

    def _get_buffer_experience(self):
        """
        Perpares the experience to add to the buffer.
        """
        reward = self._n_step_decay()

        experience = self.experiences.pop()

        experience, q_val, next_q = experience
        experience, hidden_state = experience[:-1], experience[-1]

        experience[2] = reward

        target_q_val = reward + self.decay * next_q

        if len(self.ready_experiences) % self.sequence_length == 0:
            self.hidden_states.append(hidden_state)

        self.ready_experiences.append(experience)
        self.q_vals.append(q_val)
        self.target_qs.append(target_q_val)

    def add_to_buffer(self, experience, q_val, target_q_val):
        """
        Adds the experience to the replay buffer.
        """
        self.experiences.append((experience, q_val, target_q_val))

        if len(self.experiences) == self.experiences.maxlen:
            self._get_buffer_experience()

        if (len(self.ready_experiences) == self.sequence_length
                                           + self.burn_in_length):
            self.add_experience()

    def add_experience(self):
        """
        Adds an experience to the real replay buffer.
        """
        q_vals = torch.cat(self.q_vals, dim=1)
        target_qs = torch.cat(self.target_qs, dim=1)

        hidden_state = np.array(self.hidden_states.pop(), dtype=object)
        ready_experiences = np.array(self.ready_experiences, dtype=object)

        self.train_batch((ready_experiences, hidden_state), q_vals,
                            target_qs)

        self.ready_experiences = self.ready_experiences[-self.burn_in_length:]
        self.q_vals = self.q_vals[-self.burn_in_length:]
        self.target_qs = self.target_qs[-self.burn_in_length:]

    def get_helper_process_request(self):
        manager_file = "dragonbot/managers/bot_pool_manager.py"
        key = "dragonbot_pool"
        request = HelperProcessRequest(manager_file, key)

        self.pipe, request.pipe = Pipe()
        return request

    def get_output(self, packet):
        if not packet.game_info.is_round_active:
            return SimpleControllerState()

        # Game is seriously going too fast, scaling down fps
        start_time = time()

        model_inp = self.input_formatter.transform_batch([[packet]])

        model_out = self.model.step(model_inp, self.action, self.hidden_state)
        next_hidden = model_out[-1]

        action, next_q = self.output_formatter.transform_output(model_out[:-1])

        if self.train:
            #next_state, reward, terminal, is_start = self.env.step(packet, model_inp)
            #self.terminal = torch.FloatTensor([[[terminal]]]).to(self.device)

            self.reward = self.get_reward(packet)
            if not is_start:
                self.add_to_buffer([self.state, self.action, reward,
                                    next_state, self.terminal, self.last_action,
                                    self.hidden_state], self.q_val, next_q)

            self.state = next_state
            self.action = model_out[0]
            self.q_val = next_q

        self.last_action = self.action
        self.hidden_state = next_hidden

        # Scaling down FPS
        end_time = time()
        if end_time - start_time < 1 / self.FPS:
            sleep(1 / self.FPS - (end_time - start_time))
        #print(time() - start_time)
        return action

    def get_reward(self, next_packet):
        """
        Returns the reward for the action.
        """
        # Reward is the squared difference between the locations of the car and
        # the ball
        car_info = self.input_formatter.get_obj_info(next_packet.game_cars[self.index])
        ball_info = self.input_formatter.get_obj_info(next_packet.game_ball)

        mse = torch.mean(0.5 * (car_info[:2] - ball_info[:2]) ** 2).mean(-1)
        mse = mse.view(1, 1, 1).to(self.device)

        return mse
