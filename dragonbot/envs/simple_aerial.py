import torch

from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from .env import RLEnv

class SimpleAerial(RLEnv):
    """
    A simple aerial from goal to midfield.
    """
    def __init__(self, team, index, max_steps=0):
        super().__init__(team, index, max_steps)

        self.device = "cuda"

    def state_space(self):
        return (25,)

    def reset(self):
        """
        Resets the game state.
        from https://github.com/SaltieRL/Saltie/blob/master/agents/self_evolving_car/train.py
        """
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(0, 0, 1022),
                                       angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0, 0, 0), rotation=Rotator(45, 90, 0),
                                             location=Vector3(0.0, -4608, 500), angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)

        return game_state

    def get_reward(self, next_packet, next_state):
        """
        Returns the reward for the action.
        """
        # Reward is the squared difference between the locations of the car and
        # the ball
        car_loc = next_state[:2]
        ball_loc = next_state[self.state_space[0] // 2 + 1:self.state_space[0] // 2 + 3]

        mse = torch.mean(0.5 * (car_info[:2] - ball_info[:2]) ** 2).mean(-1)
        mse = mse.view(1, 1, 1).to(self.device)

        return mse

    def get_terminal(next_packet, next_state):
        return next_packet.game_ball.latest_touch.player_index == self.index