import torch

from rlbot.agents.base_agent import SimpleControllerState, BaseAgent

class OutputFormatter():
    """
    A class to format model output 
    """
    def transform_output(self, model_output):
        """
        Transforms the output to the new controller state and the action or
        state value.
        """
        action, val = model_output
        action = action[0].cpu().numpy()

        controller_out = BaseAgent.convert_output_to_v4(self, action)

        return controller_out, val

    @staticmethod
    def action_space():
        """
        Returns the number of output actions.
        """
        return 8