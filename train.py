import os

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))
    exercises_path = file_path + "/src/rlbot/exercises/training_exercises.py"

    os.execvp(
        "rlbottraining",
        [
            "rlbottraining",
            "run_module",
            exercises_path
        ]
    )
