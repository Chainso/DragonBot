import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from dragonbot.models.model import Model

# Convert policy to agent inputs
def model_out_to_agent_in(policy):
    # First 5 are -1 to 1, last 3 are bools (not using item)
    return policy[:5] + [True if inp >= 0 else False for inp in policy[3:]]

# Convert the game packet to a state for model input
def packet_to_state(packet):
    return [1, 0]

class RLAgent(BaseAgent):
    def initialize_agent(self):
        # This runs once before the bot starts up
        self.controller = SimpleControllerState()
        self.model = Model(2, 8)

    def get_model_prediction(self, state):
        # Get the prediction of the current state
        state = torch.FloatTensor([state])
        pred, val = self.model(state)
        return pred[0].numpy(), val.item()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        state = packet_to_state(packet)
        pred, val = self.get_model_prediction(state)
        new_controller_state = model_out_to_agent_in(pred)

        self.controller.throttle = new_controller_state[0]
        self.controller.steer = new_controller_state[1]
        self.controller.pitch = new_controller_state[2]
        self.controller.yaw = new_controller_state[3]
        self.controller.roll = new_controller_state[4]
        self.controller.jump = new_controller_state[5]
        self.controller.boost = new_controller_state[6]
        self.controller.handbrake = new_controller_state[7]

        return self.controller
