import torch

class InputFormatter():
    """
    A class to format game input to feed into the model.
    """
    def __init__(self, team, index, device="cpu"):
        self.team = team
        self.index = index
        self.device = torch.device(device)

    def _get_car_info(self, car):
        """
        Gets the relevant information for a game car.
        """
        physics = car.physics

    def transform_packet(self, packet):
        """
        Transforms the packet into a state to feed into the model.
        """
        cars_info = []
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