import torch
import torch.nn.functional as F

class MultiCategoricalDistribution():

    def __init__(self, action_dims):
        # action_dims = [21,21,11,21,2,2,3,2]
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution(self, action_logits):
        # create a list of categorical distribution for each dimension
        self.distribution = [torch.distributions.Categorical(logits=F.softmax(split)) for split in torch.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions):
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self):
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self):
        # computes mode of each categorical dist.
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def get_actions(self, deterministic=False):
        if deterministic:
            return self.mode()
        return self.sample()