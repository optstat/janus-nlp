import lightgbm as lgb
import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.get_sampler import GetSampler
from torch import Tensor
from lightgbm import LGBMRegressor
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors import FullyBayesianPosterior
from botorch.acquisition import qExpectedImprovement
import torch.nn as nn
import torch.optim as optim
# Set the seed for reproducibility
torch.manual_seed(1)
# Double precision is highly recommended for BoTorch.
# See https://github.com/pytorch/botorch/discussions/1444
torch.set_default_dtype(torch.float64)


# Define a simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

train_X = torch.rand(1000, 2) ** 2
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
Y += 0.1 * torch.rand_like(Y)
bounds = torch.stack([torch.zeros(2), 2 * torch.ones(2)])


# Instantiate and train the autoencoder
input_dim = train_X.shape[1]
latent_dim = 2  # Set this based on the expected number of distinct features

autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    reconstructed = autoencoder(train_X)
    loss = loss_fn(reconstructed, train_X)
    loss.backward()
    optimizer.step()



class EnsembleLightGBMModel(Model):
    num_samples: int
    models: list[LGBMRegressor]
    
    def __init__(self, num_models: int = 10, **kwargs):
        super().__init__()
        self._num_outputs = 1
        self.num_models = num_models
        self.kwargs = kwargs
        # Initialize a list to hold multiple LightGBM models
        self.models = [LGBMRegressor(**kwargs) for _ in range(num_models)]
    
    @property
    def num_outputs(self) -> int:
        """Defines the number of outputs, required by BoTorch."""
        return self._num_outputs


    #create a setter for num_ouputs
    def set_num_outputs(self, num_outputs: int) -> None:
        self.num_outputs = num_outputs


    def fit(self, X: Tensor, y: Tensor) -> None:
        # Convert PyTorch tensors to numpy arrays
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().squeeze()
        # Train each model in the ensemble
        for i, model in enumerate(self.models):
            # Optionally, you can use different random seeds or subsets of data
            # For simplicity, we'll just fit each model on the entire dataset
            model.set_params(random_state=i)
            model.fit(X_np, y_np)


    def forward(self, X: Tensor) -> Tensor:
        x_np = X.detach().cpu().numpy().squeeze()
        # Collect predictions from each model in the ensemble
        y = torch.from_numpy(np.array([model.predict(x_np) for model in self.models]))
        # Reshape to match (batch_size) x s x q x m, where:
        # `batch_size` is the number of input points,
        # `s` is the number of models (ensemble size),
        # `q` is 1 (single query),
        # `m` is 1 (scalar output dimension)
        samples = y.T.reshape(X.shape[0], self.num_models, 1, self.num_outputs)
        return samples

    def posterior(self, X: Tensor, observation_noise=False, **kwargs) -> Posterior:
        samples = self.forward(X)
        mean = samples.mean(dim=1).squeeze(-1)  # Shape: (batch_size, 1)
        variance = samples.var(dim=1).squeeze(-1) + 1e-6  # Shape: (batch_size,)

        # Expand variance to match required covariance matrix shape
        cov_matrix = torch.diag_embed(variance.expand(X.shape[0], self._num_outputs))
        mvn = MultivariateNormal(mean, cov_matrix)
        return GPyTorchPosterior(mvn)    

# Encode the training data
train_X_latent = autoencoder.encode(train_X).detach()

# Modify the EnsembleLightGBMModel to operate on the encoded data
class LatentEnsembleLightGBMModel(EnsembleLightGBMModel):
    def fit(self, X: Tensor, y: Tensor) -> None:
        # Convert latent space PyTorch tensors to numpy arrays
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().squeeze()
        for i, model in enumerate(self.models):
            model.set_params(random_state=i)
            model.fit(X_np, y_np)

# Instantiate and fit the modified model on the latent data
latent_model = LatentEnsembleLightGBMModel(num_models=20, num_leaves=31, n_estimators=100)
latent_model.fit(train_X_latent, Y)


# Define acquisition function in the latent space
best_f = Y.max().item() + 1e-4
acq_func = qLogExpectedImprovement(model=latent_model, best_f=best_f)

# Optimize acquisition function to select next candidate in latent space
bounds_latent = torch.stack([torch.zeros(latent_dim), torch.ones(latent_dim)])
candidates_latent, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds_latent,
    q=1,
    num_restarts=5,
    raw_samples=10,
    options={"with_grad": False},
)

# Decode the candidate from latent space back to original space
candidates = autoencoder.decode(candidates_latent).detach()
new_Y = 1 - (candidates- 0.5).norm(dim=-1, keepdim=True)
new_Y += 0.1 * torch.rand_like(new_Y)
# Update training data with new evaluations
train_X = torch.cat([train_X, candidates])
Y = torch.cat([Y, new_Y])

# Encode updated data to latent space
train_X_latent = autoencoder.encode(train_X).detach()

# Retrain the surrogate model in the latent space
latent_model.fit(train_X_latent, Y)
# Define test points and encode them to latent space
test_X = torch.rand(5, 2) ** 2  # Example with 5 test points in the original 1000-dimensional space
test_X_latent = autoencoder.encode(test_X)

# Get posterior in the latent space
posterior = latent_model.posterior(test_X_latent)
mean = posterior.mean
variance = posterior.variance

print("Mean predictions:\n", mean)
print("Uncertainty (variance):\n", variance)