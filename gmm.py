import torch
from torch import nn
from torch.distributions.mixture_same_family import MixtureSameFamily


class GaussianMixture(MixtureSameFamily):
    def __init__(self, pi, mu, sigma):
        super().__init__(
            mixture_distribution=torch.distributions.Categorical(pi),
            component_distribution=torch.distributions.Normal(mu, sigma),
        )


class GMMSimple(nn.Module):
    def __init__(self, n_components=5):
        super().__init__()

        self.pi = nn.Parameter(
            data=torch.ones(n_components).reshape(1, -1) / n_components
        )
        self.mu = nn.Parameter(data=torch.linspace(-2, 2, n_components).reshape(1, -1))
        self.sigma = nn.Parameter(
            data=torch.ones(n_components).reshape(1, -1) / n_components
        )

    def forward(self):
        pi, mu, sigma = self.pi, self.mu, self.sigma

        pi = nn.Softmax(dim=-1)(pi)
        mu = mu
        sigma = nn.Softplus()(sigma)  # .clamp(1e-2, 1e2)

        return pi, mu, sigma

    def log_prob(self, x):
        return GaussianMixture(*self()).log_prob(x)

    def sample(self, n=None):
        return GaussianMixture(*self()).sample(n)  # type: ignore


def train_model(
    model: GMMSimple,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    n_epochs: int,
):
    # standardize the data
    data_mean = x.mean()
    data_std = x.std()
    x = (x - data_mean) / data_std

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        loss = -model.log_prob(x).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # average losses
    losses = torch.tensor(losses)
    losses = losses[torch.isfinite(losses)]
    return losses.mean().item()


def extract_gmm_parameters(model: GMMSimple, data_mean: float, data_std: float):
    with torch.inference_mode():
        pi, mu, sigma = model()

    pi = pi.detach().numpy().reshape(-1)
    mu = mu.detach().numpy().reshape(-1)
    sigma = sigma.detach().numpy().reshape(-1)

    # denormalize the mu and sigma
    mu = mu * data_std + data_mean
    sigma = sigma * data_std

    return pi, mu, sigma
