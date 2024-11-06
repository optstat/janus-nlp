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

# Set the seed for reproducibility
torch.manual_seed(1)
# Double precision is highly recommended for BoTorch.
# See https://github.com/pytorch/botorch/discussions/1444
torch.set_default_dtype(torch.float64)

train_X = torch.rand(1000, 2) ** 2
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
Y += 0.1 * torch.rand_like(Y)
bounds = torch.stack([torch.zeros(2), 2 * torch.ones(2)])

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

# Instantiate the model with desired hyperparameters
lightGBMModel = EnsembleLightGBMModel(num_models=20, num_leaves=31, n_estimators=100)
lightGBMModel.fit(train_X, Y)
best_f = Y.max().item() + 1e-4  # Add a small jitter to best_f
optimize_acqf(
    qLogExpectedImprovement(model=lightGBMModel, best_f=Y.max()),
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=10,
    options={"with_grad": False},
)

# Define the test points where you want to make predictions
test_X = torch.rand(5, 2)** 2  # Example with 5 test points in a 2-dimensional space

# Get the posterior, which includes the mean and variance
posterior = lightGBMModel.posterior(test_X)

# Extract mean and covariance from the posterior distribution
mean = posterior.mean  # Mean predictions, shape: (5, 1)
variance = posterior.variance  # Variance (uncertainty), shape: (5, 1)

print("Mean predictions:\n", mean)
print("Uncertainty (variance):\n", variance)