import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d


# Fit polynomial latent dynamics for each individual's concentration
def fit_latent_dynamics(t, Z, degree):
    """Fit polynomial coefficients for latent dynamics."""
    T = np.vstack([t**k for k in range(degree + 1)]).T
    coeffs = np.linalg.lstsq(T, Z, rcond=None)[0]
    return coeffs


def latent_variable(t, phi):
    """Compute the polynomial latent variable Q(t) based on the given parameters."""
    return sum(phi[i] * (t ** i) for i in range(len(phi)))

# Function to build feature library for SINDy including latent variable impact
def build_feature_library_latent(X, Z, Q_t):
    A = np.column_stack([X, Z, Q_t, X * Z, X * Q_t, Z * Q_t, X**2, Z**2])
    return A

# Fine-tune for individual-specific dynamics using their fitted latent coefficients
def fine_tune_individual_dynamics(X_i, dX_dt_i, feature_library, beta_est, lambda_reg=0.1):
    def loss_function(beta_i):
        mse_loss = np.mean((dX_dt_i - feature_library @ beta_i) ** 2)
        reg_loss = lambda_reg * np.sum((beta_est - beta_i) ** 2)
        return mse_loss + reg_loss

    res = minimize(loss_function, beta_est, method="L-BFGS-B")
    return res.x

# Now you can proceed with predicting dX/dt incorporating the fitted coefficients
# Update the predict function to include latent influences
def predict_dX_dt(X, V, individual_betas, time_points, latent_coeffs_list):
    """Predict dX/dt for all individuals considering dynamically computed latent variable Q."""
    dX_dt_predictions = []

    for i, beta_individual in enumerate(individual_betas):
        Q_t = latent_variable(time_points, latent_coeffs_list[i])  # Use the individual's latent coeffs

        dX_dt_i = (beta_individual[0] * X[i] * (1 - X[i] / 10) +
                   beta_individual[1] * V +
                   beta_individual[2] * Q_t +
                   beta_individual[3] * X[i] * V +
                   beta_individual[4] * X[i] * Q_t +
                   beta_individual[5] * V * Q_t)
        
        dX_dt_predictions.append(dX_dt_i)
    
    return np.array(dX_dt_predictions)

# Function to predict cancer volume from predicted dX/dt with latent influences
def predict_cancer_volume(dX_dt, initial_volume, time_points):
    predicted_volumes = []
    
    for dX in dX_dt:
        def dXdt(t, x):
            return dX[int(np.clip(t, 0, len(dX) - 1))]  # Ensure appropriate indexing

        # Integrate to obtain the cancer volume over time
        sol = solve_ivp(dXdt, [time_points[0], time_points[-1]], [initial_volume], t_eval=time_points)
        predicted_volume = sol.y[0]  # Extracting the predicted cancer volume
        predicted_volumes.append(predicted_volume)

    return np.array(predicted_volumes)


def predict_dX_dt_latent(X, V, individual_betas, time_points, latent_coeffs_list):
    """Predict dX/dt for all individuals considering dynamically computed latent variable Q."""
    dX_dt_predictions = []

    for i, beta_individual in enumerate(individual_betas):
        Q_t = latent_variable(time_points, latent_coeffs_list[i])  # Use the individual's latent coeffs

        dX_dt_i = (beta_individual[0] * X[i] * (1 - X[i] / 100) +
                   beta_individual[1] * V +
                   beta_individual[2] * Q_t +
                   beta_individual[3] * X[i] * V +
                   beta_individual[4] * X[i] * Q_t +
                   beta_individual[5] * V * Q_t)
        
        dX_dt_predictions.append(dX_dt_i)
    
    return np.array(dX_dt_predictions)
