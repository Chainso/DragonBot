from pathlib import Path

from rlbottraining.common_exercises.ball_prediction import make_default_playlist as mdp
from rlbot.matchconfig.match_config import PlayerConfig, Team

def make_default_playlist():
    for exercise in mdp():
        exercise.match_config.player_configs = [
            PlayerConfig.bot_config(Path("src/bot.cfg"), Team.BLUE)
        ]