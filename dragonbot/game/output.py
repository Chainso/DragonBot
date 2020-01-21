import torch
import numpy as np

from rlbot.agents.base_agent import SimpleControllerState, BaseAgent

class OutputFormatter():
    """
    A class to format model output 
    """
    def transform_action(self, action):
        """
        Transforms the action into a controller state.
        """
        action = action[0].detach().cpu().numpy()

        # Convert the last 3 actions to their boolean values
        action = np.concatenate((action[:5], (action[5:] >= 0)), axis = 0)

        controller_out = BaseAgent.convert_output_to_v4(self, action)

        return controller_out

    def transform_output(self, model_output):
        """
        Transforms the output to the new controller state and the action or
        state value.
        """
        action, val = model_output

        action = self.transform_action(action)
        val = val.detach()[0]

        return action, val

    @staticmethod
    def action_space():
        """
        Returns the number of output actions.
        """
        return 8

class RecurrentOutputFormatter(OutputFormatter):
    def transform_action(self, action):
        return OutputFormatter.transform_action(self, action[0])