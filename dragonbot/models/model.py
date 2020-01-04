import torch
import torch.nn as nn

RANDOM_SEED = 999
torch.manual_seed(RANDOM_SEED)

class Model(nn.Module):
    def __init__(self, state_space, act_n):
        nn.Module.__init__(self)

        self.state_space = state_space
        self.act_n = act_n

        self.hidden = nn.Linear(state_space, 16)

        self.policy = nn.Sequential(
            nn.Linear(16, act_n),
            nn.Tanh()
        )

        self.value = nn.Linear(16, 1)

        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)

    def forward(self, inp):
        hid = self.hidden(inp)
        policy = self.policy(hid)
        value = self.value(hid)

        return policy, value

    def step(self, inp, stochastic=True):
        return self(inp)

    def train_supervised(self, states, actions):
        self.optimizer.zero_grad()
        states = self.noise(states)

        self.lstm.reset_hidden()

        new_acts, policy, value = self(states)

        policy_loss = self.il_weight * self.loss(policy, actions.argmax(1))
        policy_loss = policy_loss.mean()

        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.cpu().detach().numpy()

    def train_reinforce(self, rollouts):
        self.lstm.reset_hidden()

        states, acts, rewards, advs = [torch.from_numpy(tensor).to(self.device)
                                       for tensor in rollouts]

        states = states.permute(0, 3, 1, 2)

        actions, policy, value = self(states)

        policy_loss = advs.unsqueeze(1) * self.loss(policy, acts.argmax(1))
        policy_loss = policy_loss.mean()

        value_loss = self.mse_loss(value, rewards.unsqueeze(1))

        rnd_loss = self.mse_loss(self.rnd(states),
                                 self.rnd_target(states).detach())
 
        loss = policy_loss + value_loss + rnd_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def reset_hidden_state(self):
        """
        Resets the hidden state for the LSTM
        """
        self.lstm.reset_hidden()

    def save(self, save_path):
        """
        Saves the model at the given save path

        save_path : The path to save the model at
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        """
        Loads the model at the given load path

        load_path : The of the model to load
        """
        self.load_state_dict(torch.load(load_path))