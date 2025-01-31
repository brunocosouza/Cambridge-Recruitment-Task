import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d

# Build feature library for SINDy without latent variables
def build_feature_library_no_latent(X):
    A = np.column_stack([X, X**2])  # Only include X and its square to fit dynamics
    return A

def fit_population_dynamics_sindy(X, Z, time_points):
    """ Fit the population dynamics using SINDy approach. """
    A = build_feature_library_no_latent(X)  # Create feature space excluding latent variables
    dX_dt = np.gradient(Z, time_points)  # Calculate dX/dt

    lasso = Lasso(alpha=0.01, fit_intercept=False)
    print(A.shape, dX_dt.shape)
    lasso.fit(A, dX_dt)
    return lasso.coef_  # Returns estimated coefficients for the population model


def fit_population_dynamics_no_latent(t, dX_dt, feature_library):
    """Fit the population dynamics using SINDy approach without latent variables."""
    # dX/dt is the change in cancer volume, we assume Z correlates with this
    Y = dX_dt  # Actual dX/dt from individual data

    lasso = Lasso(alpha=0.01, fit_intercept=False)  # Regularization parameter
    lasso.fit(feature_library, Y)  # Fit the model
    return lasso.coef_

# Fine-tune for individual-specific dynamics using their fitted latent coefficients
def fine_tune_individual_dynamics(X_i, dX_dt_i, feature_library, beta_est, lambda_reg=0.1):
    def loss_function(beta_i):
        mse_loss = np.mean((dX_dt_i - feature_library @ beta_i) ** 2)
        reg_loss = lambda_reg * np.sum((beta_est - beta_i) ** 2)
        return mse_loss + reg_loss

    res = minimize(loss_function, beta_est, method="L-BFGS-B")
    return res.x


def predict_dX_dt_no_latent(X, Z, individual_betas_no_latent, time_points):
    dX_dt_predictions = []
    
    for i, beta_individual in enumerate(individual_betas_no_latent):
        if len(beta_individual) == 2:
            # Only using the respective X[i] for computation as we don't have latent variables
            dX_dt_i = (beta_individual[0] * X[i] * (1 - X[i] / 10) +
                        beta_individual[1] * Z)
        else:
            # If additional coefficients exist (for debugging, if needed)
            dX_dt_i = (beta_individual[0] * X[i] * (1 - X[i] / 10) +
                        beta_individual[1] * Z +
                        beta_individual[2] * np.sin(np.array(time_points)))  # Fall back to previous behavior

        dX_dt_predictions.append(dX_dt_i)
    
    return np.array(dX_dt_predictions) 


# Function to predict cancer volume from predicted dX/dt without latent effects
def predict_cancer_volume_no_latent(dX_dt, initial_volume, time_points):
    predicted_volumes = []
    
    for dX in dX_dt:
        def dXdt(t, x):
            return dX[int(np.clip(t, 0, len(dX) - 1))]  # Ensure appropriate indexing

        # Solve the ODE to obtain cancer volume over time
        sol = solve_ivp(dXdt, [time_points[0], time_points[-1]], [initial_volume], t_eval=time_points)
        predicted_volume = sol.y[0]  # Extract the predicted cancer volume from the solution
        predicted_volumes.append(predicted_volume)

    return np.array(predicted_volumes)