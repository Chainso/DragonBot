import torch

from torch.multiprocessing import Pipe
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.botmanager.helper_process_request import HelperProcessRequest

from dragonbot.game.input import InputFormatter
from dragonbot.game.output import OutputFormatter

class PoolAgent(BaseAgent):
    """
    An agent that exists in a pool with other agents.
    """
    def initialize_agent(self):
        (self.device, self.model) = self.pipe.recv()
        self.pipe.send(None)

        self.input_formatter = InputFormatter(self.team, self.index,
                                              self.device)
        self.output_formatter = OutputFormatter()

    def get_helper_process_request(self):
        manager_file = "dragonbot/managers/bot_pool_manager.py"
        key = "dragonbot_pool"
        request = HelperProcessRequest(manager_file, key)

        self.pipe, request.pipe = Pipe()
        return request

    def get_output(self, packet):
        model_inp = self.input_formatter.transform_packet(packet)

        action = torch.FloatTensor([[1, -0.1, 0.2, 0, 0, 0, 0, 0]])
        val = 1

        controller, val = self.output_formatter.transform_output((action, val))
        return controller
