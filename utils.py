import numpy as np
from pkpd import *
from latentINSITE import predict_cancer_volume, predict_dX_dt
from baselineINSITE import predict_cancer_volume_no_latent, predict_dX_dt_no_latent

# Function to calculate Mean Squared Error
def calculate_mse(true_values, predicted_values):
    return np.mean((true_values - predicted_values) ** 2)

# Function to calculate R² (Goodness-of-Fit)
def calculate_r_squared(true_values, predicted_values):
    ss_res = np.sum((true_values - predicted_values) ** 2)  # Residual sum of squares
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)  # R² value


# Function to calculate individual metrics including latent variables
def calculate_average_individual_metrics(treatments, num_individuals, population_params, phi, time_points, actual_volumes, individual_betas, latent_coeffs_list, latent=True):
    individual_mse = []
    
    for treatment_index in range(len(treatments)):
        # Generate actual volumes for the current treatment
        actual_volume_for_treatment = actual_volumes[:, treatment_index, :]

        # Generate predictions for the current treatment using latent model
        results_concentration_new, results_volume_new = generate_new_predictions(num_individuals, population_params, phi, time_points, treatments[treatment_index])
        V = np.mean(results_volume_new, axis=0)  # Average cancer volume across individuals


        if latent:
            # Predict dX/dt for all individuals including latent influence in prediction
            dX_dt_predicted = predict_dX_dt(results_concentration_new, V, individual_betas, time_points, latent_coeffs_list)
            # Predict cancer volumes for all individuals with new treatment conditions
            predicted_cancer_volumes_new = predict_cancer_volume(dX_dt_predicted, initial_volume=1, time_points=time_points)
        else:
            dX_dt_predicted = predict_dX_dt_no_latent(results_concentration_new, V, individual_betas, time_points)
            # Predict cancer volumes for all individuals with the current treatment
            predicted_cancer_volumes_new = predict_cancer_volume_no_latent(dX_dt_predicted, initial_volume=1, time_points=time_points)

        # Calculate metrics for each individual
        for i in range(num_individuals):
            mse = calculate_mse(actual_volume_for_treatment[i], predicted_cancer_volumes_new[i])

            if len(individual_mse) <= i:
                individual_mse.append([])  # Initialize inner list for individual MSE

            individual_mse[i].append(mse)

    # Average MSE and standard deviation for each individual
    average_mse = [np.mean(mse) for mse in individual_mse]
    std_deviation_mse = [np.std(mse) for mse in individual_mse]  # Standard deviation for each individual

    return average_mse, std_deviation_mse

def calculate_average_treatment_mse(treatments, num_individuals, population_params, phi, time_points, actual_volumes, individual_betas, latent_coeffs_list, latent=True):
    average_mse = []
    std_deviation_mse = []

    for treatment_index, treatment in enumerate(treatments):
        # Generate predictions for the current treatment regimen
        results_concentration_new, results_volume_new = generate_new_predictions(num_individuals, population_params, phi, time_points, treatment)
        V = np.mean(results_volume_new, axis=0)  # Average cancer volume across individuals

        # Calculate dX/dt for the new predictions
        if len(results_volume_new) == 0:
            print("No results generated, skipping this treatment.")
            continue
        
        if latent:
            # Predict dX/dt for all individuals including latent influence in prediction
            dX_dt_predicted = predict_dX_dt(results_concentration_new, V, individual_betas, time_points, latent_coeffs_list)
            # Predict cancer volumes for all individuals with new treatment conditions
            predicted_cancer_volumes_new = predict_cancer_volume(dX_dt_predicted, initial_volume=1, time_points=time_points)
            # Predict cancer volumes for all individuals with the current treatment
            #predicted_cancer_volumes_new = predict_cancer_volume(dX_dt_new, initial_volume=1, time_points=time_points)
        else:
            dX_dt_predicted = predict_dX_dt_no_latent(results_concentration_new, V, individual_betas, time_points)
            # Predict cancer volumes for all individuals with the current treatment
            predicted_cancer_volumes_new = predict_cancer_volume_no_latent(dX_dt_predicted, initial_volume=1, time_points=time_points)
        
        # Calculate MSE for each individual using the specified treatment index
        mse_list = []
        for i in range(len(results_volume_new)):  # Use the actual number of results generated
            mse = calculate_mse(actual_volumes[i][treatment_index], predicted_cancer_volumes_new[i])  # Assuming calculate_mse function is defined
            mse_list.append(mse)
        
        # Aggregate MSE for this treatment
        average_mse.append(np.mean(mse_list))
        std_deviation_mse.append(np.std(mse_list))  # Calculate standard deviation of MSEs

    return average_mse, std_deviation_mse  # Return both average and standard deviation

