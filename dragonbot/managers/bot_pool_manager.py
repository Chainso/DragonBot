import torch

from rlbot.botmanager.bot_helper_process import BotHelperProcess

class BotPoolManager(BotHelperProcess):
    """
    Creates a manager for a pool of bots.
    """
    def __init__(self, metadata_queue, quit_event, options={}):
        # Imported here because the runner changes the system path
        import torch.nn as nn

        from configparser import ConfigParser
        from hlrl.torch.algos import SACRecurrent
        from hlrl.torch.policies import LSTMSAPolicy, LSTMGaussianPolicy
        from hlrl.torch.experience_replay import TorchR2D2
        from hlrl.core.logger import TensorboardLogger

        from dragonbot.game.input import RecurrentInputFormatter
        from dragonbot.game.output import RecurrentOutputFormatter
        from dragonbot.envs import SimpleAerial

        super().__init__(metadata_queue, quit_event, options)

        config = ConfigParser()
        config.read("dragonbot/rlbot/config/algo.cfg")

        self.env = SimpleAerial
        self.config = config
        self.device = torch.device(config["General"]["device"])

        # The logger
        logs_path = config["General"]["logs_path"]

        self.logger = None if len(logs_path) == 0 else TensorboardLogger(logs_path)

        # Initialize SAC
        state_space = RecurrentInputFormatter.state_space()
        action_space = RecurrentOutputFormatter.action_space()

        activation_fn = nn.ReLU
        qfunc = LSTMSAPolicy(*state_space, action_space, 1,
                             int(config["Model"]["hidden_size"]), 0,
                             int(config["Model"]["hidden_size"]), 1,
                             int(config["Model"]["hidden_size"]),
                             int(config["Model"]["num_hidden"]) - 1,
                             activation_fn)

        policy = LSTMGaussianPolicy(*state_space, action_space, action_space,
                                    int(config["Model"]["hidden_size"]),
                                    0, int(config["Model"]["hidden_size"]), 1,
                                    int(config["Model"]["hidden_size"]),
                                    int(config["Model"]["num_hidden"]) - 1,
                                    activation_fn, squished=True)

        optim = lambda params: torch.optim.Adam(params,
                                                lr=float(config["Algorithm"]["lr"]))
        self.model = SACRecurrent([action_space], qfunc, policy,
                                  float(config["Algorithm"]["discount"]),
                                  float(config["Algorithm"]["polyak"]),
                                  float(config["Algorithm"]["target_update_interval"]),
                                  optim, optim, optim,
                                  config["Algorithm"].getboolean("twin"),
                                  int(config["Algorithm"]["burn_in_length"]),
                                  self.logger)

        if (len(config["General"]["load_path"]) > 0):
            self.model.load(config["General"]["load_path"])

        self.model = self.model.to(self.device)

        self.model.share_memory()

        self.experience_buffer = TorchR2D2(int(config["Replay"]["capacity"]),
                                           float(config["Replay"]["alpha"]),
                                           float(config["Replay"]["beta"]),
                                           float(config["Replay"]["beta_increment"]),
                                           float(config["Replay"]["epsilon"]),
                                           float(config["Replay"]["max_factor"]))

        self.pipes = []

    def _get_pipe_input(self):
        """
        Returns the items that need to be sent to each agent.
        """
        return (self.config, self.device, self.model, self.experience_buffer,
                self.logger, sself.env)

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe
            pipe.send(self._get_pipe_input())
            self.pipes.append(pipe)

        if self.config["Training"].getboolean("train"):
            #self.train()
            self.play()
        else:
            self.play()

    def train(self):
        """
        Trains the model for as long as the game is running.
        """
        batch_size = int(self.config["Training"]["batch_size"])
        start_size = int(self.config["Training"]["start_size"])

        save_path = self.config["General"]["save_path"]
        save_path = None if len(save_path) == 0 else save_path

        save_interval = int(self.config["Training"]["save_interval"])

        while not self.quit_event.is_set():
            for pipe in self.pipes:
                print("Waiting to RECV")
                experience, q_val, next_q = pipe.recv()
                print("Post RECV")
                self.experience_buffer.add(experience, q_val, next_q)
            print("Post add", len(self.experience_buffer))
            self.model.train_from_buffer(self.experience_buffer, batch_size,
                                         start_size, save_path, save_interval)

    def play(self):
        """
        Plays using the model for as long as the game is running.
        """
        while not self.quit_event.is_set():
            pass