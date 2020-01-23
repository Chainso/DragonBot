from hlrl.core.envs import Env

class RLEnv(Env):
    """
    An environment (game configuration) for the bot.
    """
    def __init__(self, team, index, max_steps=0):
        super().__init__()
        self.team = team
        self.index = index
        self.max_steps = max_steps

        self.packet = None
        self.is_start = True

    def get_reward(self, next_packet, next_state):
        """
        Gets the reward of the next state.
        """
        raise NotImplementedError

    def get_terminal(self, next_packet, next_state):
        """
        Checks if the new state is a terminal state.
        """
        raise NotImplementedError

    def step(self, next_packet, next_state):
        """
        Receives a the next state and returns
        (next state, reward, terminal, is_start).
        To be done at the start of get_output in the agent.
        """
        self.terminal = self.get_terminal(next_packet, next_state)

        if self.is_start:
            self.packet = next_packet
            self._state = next_state

            return (self._state, self._reward, self._terminal, self.is_start)

        self.packet = next_packet
        self._state = next_state
        self.is_start = False

        self._reward = self.get_reward(next_packet, next_state)

        return (self._state, self._reward, self._terminal, self.is_start)

    def action_space(self):
        return 8