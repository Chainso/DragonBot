import os

from pathlib import Path
from dataclasses import dataclass, field
from math import pi
from typing import Optional

from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
from rlbot.matchconfig.match_config import PlayerConfig, MatchConfig, Team
from rlbot.training.training import Grade, Pass, Fail

from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.common_graders.timeout import FailOnTimeout
from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_exercises.common_base_exercises import GoalieExercise
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.common_exercises.silver_striker import make_default_playlist as mdp
from rlbottraining.match_configs import make_empty_match_config
from rlbottraining.training_exercise import Playlist
from rlbottraining.grading.grader import Grader

@dataclass
class SaveGoalGrader(Grader):
    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        car = tick.game_tick_packet.game_cars[0].physics
        jumped = tick.game_tick_packet.game_cars[0].jumped
        ball = tick.game_tick_packet.game_ball.physics
        return Pass() if jumped else None

class GoldBallRollingToGoalieGrader(CompoundGrader):
    def __init__(self, timeout_seconds=4.0):
        super().__init__([
            SaveGoalGrader(),
            FailOnTimeout(timeout_seconds)
        ])

### FROM https://github.com/GodGamer029/YangBot/
@dataclass
class GoldBallRollingToGoalie(GoalieExercise):
    # The grader is mine
    grader: Grader = field(default_factory=GoldBallRollingToGoalieGrader)

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        return GameState(
            ball=BallState(physics=Physics(
                location=Vector3(-2500, -2500, 100),
                velocity=Vector3(1000, -1000, 0), # 
                angular_velocity=Vector3(0, 0, 0))),
            cars={
                0: CarState(
                    physics=Physics(
                        location=Vector3(0, -1500, 17),
                        rotation=Rotator(0, pi * -0.5, 0),
                        velocity=Vector3(0, 0, 0),
                        angular_velocity=Vector3(0, 0, 0)),
                    boost_amount=30)
            },
        )

def make_match_config() -> MatchConfig:
    file_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.realpath(file_path + "/../config/bot.cfg")

    match_config = make_empty_match_config()
    match_config.player_configs = [
        PlayerConfig.bot_config(Path(config_path), Team.BLUE)
    ]

    return match_config

def make_default_playlist() -> Playlist:
    #exercises = mdp()
    exercises = [
        GoldBallRollingToGoalie('GoldBallRollingToGoalie'),
    ]

    for exercise in exercises:
        exercise.match_config = make_match_config()

    return exercises