import torch

class InputFormatter():
    """
    A class to format game input to feed into the model.
    """
    def __init__(self, device):
        self.device = torch.device(device)

    def transform_state(self, state):
        """
        Transforms the state to feed into the model.
        """
        return torch.FloatTensor([state]).to(device)

    @staticmethod
    def state_space():
        """
        Returns the shape of the input state (excluding batch size and sequence
        length).
        """
        return (2,)

    @staticmethod
    def input_space():
        """
        Returns the shape of the formatted input.
        """
        return (1, *InputFormatter.state_space())