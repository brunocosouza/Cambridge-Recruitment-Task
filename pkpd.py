
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d


def latent_variable(t, phi):
    """ Compute the polynomial latent variable Q(t) based on the given parameters. """
    return sum(phi[i] * (t ** i) for i in range(len(phi)))

def baseline_pkpd_model(t, y, params, phi):
    C, x = y  # Drug concentration and cancer volume
    k_elim, k_growth, k_drug_effect = params

    # Compute the latent variable Q(t)
    Q = latent_variable(t, phi)

    # Drug administration modeled as Bernoulli process (0 or 1)
    a_t = np.random.binomial(1, 0.5)  # Probability of administering drug at time t

    # Define the rate equations
    dCdt = -k_elim * C + (a_t * 5)  # Drug concentration change
    dXdt = k_growth * x * (1 - (x / 100)) + Q - k_drug_effect * a_t * C  # Cancer volume change influenced by Q

    return [dCdt, dXdt]


def treatment_pkpd_model(t, y, params, phi, treatment_params):
    C, x = y
    k_elim, k_growth, k_drug_effect = params

    Q = latent_variable(t, phi)  # Compute latent variable

    # Use treatment_params to decide on drug administration
    a_t = np.random.binomial(1, treatment_params['probability'])  # Drug given based on probability
    dCdt = -k_elim * C + (a_t * treatment_params['dose'])  # Incorporate dose from treatment
    dXdt = k_growth * x * (1 - (x / 100)) + Q - k_drug_effect * a_t * C

    return [dCdt, dXdt]


def generate_baseline_pkpd_data(num_individuals, population_params, phi, t):
    results_concentration = []
    results_volume = []

    for _ in range(num_individuals):
        y0 = [10, 1]  # Initial drug concentration and cancer volume
        # Solve ODE using solve_ivp
        res = solve_ivp(baseline_pkpd_model, [t[0], t[-1]], y0, args=(population_params, phi), t_eval=t)
        results_concentration.append(res.y[0])
        results_volume.append(res.y[1])

    results_concentration = np.array(results_concentration)
    results_volume = np.array(results_volume)

    mean_concentration = np.mean(results_concentration, axis=0)
    mean_volume = gaussian_filter1d(np.mean(results_volume, axis=0), sigma=2)

    # Calculate latent variables for visualization
    latent_values = latent_variable(t, phi)

    return results_concentration, results_volume, mean_concentration, mean_volume, latent_values


# Now define the generate_pkpd_data function
def generate_treatment_pkpd_data(num_individuals, population_params, phi, t, treatment_params):
    results_concentration = []
    results_volume = []

    for _ in range(num_individuals):
        y0 = [10, 1]  # Initial drug concentration and cancer volume
        res = solve_ivp(treatment_pkpd_model, [t[0], t[-1]], y0, args=(population_params, phi, treatment_params), t_eval=t)
        results_concentration.append(res.y[0])
        results_volume.append(res.y[1])

    results_concentration = np.array(results_concentration)
    results_volume = np.array(results_volume)

    mean_concentration = np.mean(results_concentration, axis=0)
    mean_volume = np.mean(results_volume, axis=0)

    # Calculate latent variables for visualization
    latent_values = latent_variable(t, phi)

    return results_concentration, results_volume, mean_concentration, mean_volume, latent_values


################################################################################################################################
def generate_new_predictions(num_individuals, population_params, phi, time_points, treatment_params):
    results_concentration_new = []
    results_volume_new = []

    for _ in range(num_individuals):
        y0 = [1, 1]  # Initial drug concentration and cancer volume

        # Solve ODE with the provided treatment parameters
        res = solve_ivp(treatment_pkpd_model, [time_points[0], time_points[-1]], y0, 
                         args=(population_params, phi, treatment_params), t_eval=time_points)

        
        results_concentration_new.append(res.y[0])
        results_volume_new.append(res.y[1])

    return np.array(results_concentration_new), np.array(results_volume_new)


def generate_iterative_treatment_new_predictions(num_individuals, population_params, phi, t, treatments):
    gt_results = []

    for _ in range(num_individuals):
        y0 = [10, 1]  # Initial drug concentration and cancer volume
        individual_gt_results = []
        
        for treatment_params in treatments:
            # Solve for each treatment
            res = solve_ivp(treatment_pkpd_model, [t[0], t[-1]], y0, 
                             args=(population_params, phi, treatment_params), t_eval=t)
            individual_gt_results.append(res.y[1])  # Store predicted cancer volume
        
        gt_results.append(individual_gt_results)  # Append individual results

    return np.array(gt_results)  # Return a 3D array


