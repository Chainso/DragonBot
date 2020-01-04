from pathlib import Path

from rlbottraining.common_exercises.bronze_goalie import BallRollingToGoalie
from rlbot.matchconfig.match_config import PlayerConfig, Team

def make_default_playlist():
    exercises = [
        BallRollingToGoalie("Hello Exercise World")
    ]

    for exercise in exercises:
        exercise.match_config.player_configs = [
            PlayerConfig.bot_config(Path("../src/bot.cfg"), Team.BLUE)
        ]

    return exercises