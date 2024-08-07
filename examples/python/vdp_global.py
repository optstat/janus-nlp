import torch
import janus_nlp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set the device
device = torch.device("cpu")
dtype = torch.double

# Define parameter bounds
p20min, p20max = 10.0, 25.0
ftmin, ftmax   = 0.1, 1.0
x1f, x2f       = -1.5, -25.0
x10, x20       = 1.0,  2.0

# Define normalization and standardization functions
def normalize(X, bounds):
    return (X - bounds[0]) / (bounds[1] - bounds[0])

def denormalize(X, bounds):
    return X * (bounds[1] - bounds[0]) + bounds[0]

def vdp(x):
    p2 = x[:, 0:1]
    print(f"p2: {p2}")    
    assert p20min <= p2 <= p20max, "p2 out of bounds"

    ft = x[:, 1:2]    
    x10t = torch.ones_like(p2) * x10
    x20t = torch.ones_like(p2) * x20
    janus_nlp.set_x0(x10t, x20t)
    x1ft = torch.ones_like(p2) * x1f
    x2ft = torch.ones_like(p2) * x2f
    janus_nlp.set_xf(x1ft, x2ft)

    p1 = janus_nlp.calc_p10(p2)
    print(f"p1: {p1}")
    [roots, errors] = janus_nlp.vdp_solve(p1, p2, ft)    
    return errors


class VDPTargetFunction:
    def __init__(self):
        self.config_space = {'p2': (p20min, p20max), 'ft': (ftmin, ftmax)}

    def train(self, config):
        x = np.array([[config['p2'], config['ft']]])
        return vdp(torch.tensor(x, dtype=dtype)).item()


def acquisition_function(rf_model, X_candidates, best_f):
    # Predict using individual estimators to calculate the variance
    predictions = np.array([tree.predict(X_candidates) for tree in rf_model.estimators_])
    mu = predictions.mean(axis=0)
    sigma = predictions.std(axis=0)  # Estimate standard deviation as uncertainty
    improvement = best_f - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return -ei  # We want to maximize EI, hence minimize negative EI


def optimize_acquisition_function(rf_model, X_train, y_train, batch_size=3):
    # Define the bounds for candidates
    bounds = [(p20min, p20max), (ftmin, ftmax)]
    
    # Generate candidate points
    X_candidates = np.random.uniform(0, 1, (1000, len(bounds)))
    X_candidates[:, 0] = X_candidates[:, 0] * (p20max - p20min) + p20min
    X_candidates[:, 1] = X_candidates[:, 1] * (ftmax - ftmin) + ftmin
    
    # Evaluate the acquisition function over candidates
    ei_values = acquisition_function(rf_model, X_candidates, np.min(y_train))

    # Select the top batch_size candidates
    batch_indices = np.argsort(ei_values)[:batch_size]
    return X_candidates[batch_indices]

def plot(X_train, y_train, incumbent):
    plt.figure()
    plt.scatter(X_train[:, 0], y_train, c="blue", alpha=0.6, marker="o")
    plt.scatter([incumbent[0]], [vdp(torch.tensor(incumbent, dtype=dtype)).item()], c="red", zorder=10, marker="x")
    plt.xlabel("p2")
    plt.ylabel("Objective Value")
    plt.title("VDP Function Optimization")
    plt.show()

if __name__ == "__main__":
    model = VDPTargetFunction()

    # Generate initial data
    X_train = np.random.uniform(0, 1, (5, 2))
    X_train[:, 0] = X_train[:, 0] * (p20max - p20min) + p20min
    X_train[:, 1] = X_train[:, 1] * (ftmax - ftmin) + ftmin
    y_train = np.array([model.train({'p2': x[0], 'ft': x[1]}) for x in X_train])

    # Set up the random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Optimization loop
    for _ in range(10):  # Number of iterations
        X_next_batch = optimize_acquisition_function(rf_model, X_train, y_train, batch_size=3)
        y_next_batch = np.array([model.train({'p2': x[0], 'ft': x[1]}) for x in X_next_batch])
        
        # Update the dataset
        X_train = np.vstack((X_train, X_next_batch))
        y_train = np.append(y_train, y_next_batch)
        
        # Refit the model
        rf_model.fit(X_train, y_train)

    # Best solution found
    best_idx = np.argmin(y_train)
    incumbent = X_train[best_idx]
    print("Best x:", incumbent)
    print("Best y:", y_train[best_idx])

    # Plot the results
    #plot(X_train, y_train, incumbent)