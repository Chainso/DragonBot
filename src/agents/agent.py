from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from models.model.py import Model

class RLAgent(BaseAgent):
    def initialize_agent(self):
        
        # This runs once before the bot starts up
        self.controller = SimpleControllerState()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Make the bot go forwards by setting throttle to 1
        self.controller.throttle = -1
        return self.controller