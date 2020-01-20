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
        from hlrl.torch.algos import SAC
        from hlrl.torch.policies import LinearSAPolicy, TanhGaussianPolicy
        from hlrl.torch.experience_replay import TorchPER

        from dragonbot.game.input import InputFormatter
        from dragonbot.game.output import OutputFormatter

        super().__init__(metadata_queue, quit_event, options)

        config = ConfigParser()
        config.read("dragonbot/rlbot/config/algo.cfg")

        self.config = config
        self.device = torch.device(config["General"]["device"])

        # The logger
        logs_path = config["General"]["logs_path"]

        self.logger = None if len(logs_path) == 0 else TensorboardLogger(logs_path)

        # Initialize SAC
        state_space = InputFormatter.state_space()
        action_space = OutputFormatter.action_space()

        activation_fn = nn.ReLU
        qfunc = LinearSAPolicy(*state_space, action_space, 1,
                               int(config["Model"]["hidden_size"]),
                               int(config["Model"]["num_hidden"]),
                               activation_fn)
        policy = TanhGaussianPolicy(*state_space, action_space,
                                    int(config["Model"]["hidden_size"]),
                                    int(config["Model"]["num_hidden"]),
                                    activation_fn)

        optim = lambda params: torch.optim.Adam(params,
                                                lr=float(config["Algorithm"]["lr"]))
        self.model = SAC(action_space, qfunc, policy,
                         config["Algorithm"]["discount"],
                         config["Algorithm"]["polyak"],
                         config["Algorithm"]["target_update_interval"], optim,
                         optim, optim, config["Algorithm"]["twin"],
                         self.logger).to(self.device)

        if (len(config["General"]["load_path"]) > 0):
            self.model.load(config["General"]["load_path"])

        self.model.share_memory()

        self.experience_buffer = TorchPER(int(config["Replay"]["capacity"]),
                                          float(config["Replay"]["alpha"]),
                                          float(config["Replay"]["beta"]),
                                          float(config["Replay"]["beta_increment"]),
                                          float(config["Replay"]["epsilon"]))

        self.pipes = []

    def _get_pipe_input(self):
        """
        Returns the items that need to be sent to each agent.
        """
        return (self.device, self.model)

    def start(self):
        while not self.metadata_queue.empty():
            metadata = self.metadata_queue.get()
            pipe = metadata.helper_process_request.pipe
            pipe.send(self._get_pipe_input())
            self.pipes.append(pipe)

        self.train()

    def train(self):
        """
        Trains the model for as long as the game is running.
        """
        while not self.quit_event.is_set():
            pass