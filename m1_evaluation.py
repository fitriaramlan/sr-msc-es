import numpy as np
import math
import pandas as pd
import os
import itertools
from sklearn.metrics import mean_squared_error
from data_splitter import split_data, split_data_kfold
from model_trainer import get_models
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import re
from collections import Counter
import jax.numpy as jnp
import jax
# Configure JAX to use CPU only to avoid GPU/CUDA initialization issues in multiprocessing
jax.config.update('jax_platform_name', 'cpu')
import sympy as sp
from scipy.stats import spearmanr
from interval import interval, inf, imath
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42           # TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']


class ModelEvaluator:
    def __init__(self):
        self.start_time = time.time()

    def sqrt(self, value):
        return np.sqrt(np.maximum(value, 0))

    def calculate_sigma_err(self, y_true, y_pred):
        sigma_err = np.std(y_true - y_pred)
        return sigma_err

    def calculate_mse(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def gaussian_likelihood(self, y_true, y_pred):
        """Calculate gaussian likelihood"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        resid = y_true - y_pred
        sigma = float(np.std(resid))
        
        # Edge case: perfect fit or constant prediction
        # Avoid division by zero
        if sigma <= 0:
            sigma = 1e-8
        
        n = len(y_true)
        ll = (-(n / 2) * jnp.log(2 * jnp.pi * sigma ** 2)) - ((1 / (2 * sigma ** 2)) * jnp.sum((y_true - y_pred) ** 2))
        
        return float(ll), sigma

    def gaussian_likelihood_with_sigma(self, y_true, y_pred, sigma_err):
        """Compute Gaussian log-likelihood with provided sigma for Fisher Information calculations"""
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        n = len(y_true)
        log_likelihood = (-(n / 2) * jnp.log(2 * jnp.pi * sigma_err ** 2)) - (
                (1 / (2 * sigma_err ** 2)) * (jnp.sum((y_true - y_pred) ** 2)))
        return log_likelihood

    def count_constants(self, equation):
        """Extract and count numerical constants in symbolic expression"""
        if not isinstance(equation, str):
            equation = str(equation)
        constants = re.findall(r'\b\d+\.\d+\b', equation)
        constants_count = Counter(constants)
        return constants_count, sum(constants_count.values())

    def calculate_aic(self, log_likelihood, k):
        """Calculate AIC"""
        return (-2 * log_likelihood) + (2 * k)

    def calculate_aicc(self, log_likelihood, k, n):
        aic = (-2 * log_likelihood) + (2 * k)
        adjusted = (2 * k * (k + 1)) / (n - k - 1)
        return aic + adjusted

    def calculate_bic(self, log_likelihood, k, n):
        """Calculate BIC"""
        return (-2 * log_likelihood) + (jnp.log(n) * k)

    def fisher_information_matrix(self, log_likelihood_func, params):
        """Compute Fisher Information Matrix"""
        hessian = jax.hessian(log_likelihood_func)(params)
        return -hessian

    def calculate_mdl(self, log_likelihood, fisher_info_matrix, params, equation):
        """Calculate MDL"""
        k = len(params)
        _, u = self.count_distinct_symbols(equation)
        w, _ = self.count_total_symbols(equation)
        mdl = -log_likelihood + (w * np.log(u)) - ((k / 2) * np.log(3)) + np.sum(
            [0.5 * np.log(fisher_info_matrix[i, i]) + np.log(np.abs(params[i])) for i in range(k)])
        return mdl

    def count_distinct_symbols(self, equation):
        """Count unique symbols (variables, constants, operators) in expression"""
        if not isinstance(equation, str):
            equation = str(equation)
        pattern = r'[a-zA-Z_]\w*|\d+\.\d+|\d+|[-+*/^]'
        tokens = re.findall(pattern, equation)
        unique_symbols = set(tokens)
        return unique_symbols, len(unique_symbols)

    def count_total_symbols(self, equation):
        """Count total symbols (with repetition) in expression."""
        if not isinstance(equation, str):
            equation = str(equation)
        pattern = r'[a-zA-Z_]\w*|\d+\.\d+|\d+|[-+*/^]'
        tokens = re.findall(pattern, equation)
        return len(tokens), tokens

    def ensure_column_lengths_consistent(self, results):
        """Validate that all result columns have consistent lengths (for DataFrame creation)"""
        lengths = {key: len(values) for key, values in results.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"Inconsistent column lengths detected: {lengths}. All columns must have the same length.")

    def compute_extrapolation_divergence(self, yhat_int, yhat_ext):
        range_interp = np.max(yhat_int) - np.min(yhat_int)
        range_extrap = np.max(yhat_ext) - np.min(yhat_ext)
        # if abs(range_interp) < 0.0000001:
        #     return 1000000
        # return range_extrap / range_interp
        return range_extrap / (range_interp + 1)

    def compute_prediction_variance_mv(self, expr, X, perturb_percent=0.03):
        """
        Compute prediction variance via Multiple Variables Perturbation
        Perturb all variables with all 2^N combinations of up and down
        """
        symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        if not symbols:
            return 0.0
        
        n_vars = len(symbols)
        symbol_names = [str(s) for s in symbols]
        symbol_indices = []
        for name in symbol_names:
            if name.startswith("x") and name[1:].isdigit():
                symbol_indices.append(int(name[1:]))
            elif name.startswith("feature_") and name.split("_")[-1].isdigit():
                # handle feature_1 style naming (convert to zero-based)
                symbol_indices.append(int(name.split("_")[-1]) - 1)
            else:
                raise ValueError(f"Unsupported symbol name '{name}' in expression '{expr}'")
        X_sub = X[:, symbol_indices]
        
        # Generate all combinations of + and - perturbations for each variable
        # For 2 variables: [(1.03, 1.03), (1.03, 0.97), (0.97, 1.03), (0.97, 0.97)]
        combinations = list(itertools.product([1 + perturb_percent, 1 - perturb_percent], repeat=n_vars))
        
        predictions = []
        try:
            f = sp.lambdify(symbols, expr, modules='numpy')

            # Evaluate the expression for each combination
            for combo_idx, combo in enumerate(combinations):
                X_perturbed = X_sub.copy()

                # Apply the perturbation to each variable according to the combination
                for var_idx, perturbation_factor in enumerate(combo):
                    X_perturbed[:, var_idx] *= perturbation_factor
                
                try:
                    y_pred = f(*X_perturbed.T)
                    predictions.append(y_pred)
                except Exception as eval_error:
                    # Failed during evaluation of a specific perturbation combination
                    # This could be due to: divide by zero, log(<=0), sqrt(<0), etc.
                    print(f"WARNING: Sensitivity calculation failed (MVP) during evaluation: "
                          f"Equation={expr}, Combo={combo_idx}/{len(combinations)}, "
                          f"Error={type(eval_error).__name__}: {str(eval_error)}")
                    return 1000000  # Bad equation → high sensitivity (finite value instead of inf)
        except Exception as e:
            # Equation failed during lambdify or other setup (e.g., unsupported operations)
            # Return high value to indicate BAD behavior (not 0.0 which looks "good")
            print(f"WARNING: Sensitivity calculation failed (MVP) during setup: "
                  f"Equation={expr}, Error={type(e).__name__}: {str(e)}")
            return 1000000  # Bad equation → high sensitivity (finite value instead of inf)
        
        if len(predictions) < 2:
            return 0.0
        
        # matrix of predictions shape (n_combinations, n_samples)
        pred_matrix = np.vstack(predictions)

        # Check for NaN/inf values - if ANY are found, penalize heavily
        finite_mask = np.isfinite(pred_matrix)
        n_invalid = pred_matrix.size - np.sum(finite_mask)
        
        # If ANY predictions are invalid (NaN or inf), return large penalty
        # This indicates the equation is unstable/unreliable
        if n_invalid > 0:
            print(f"WARNING: Prediction variance (MVP) detected {n_invalid} invalid predictions (NaN/inf) for equation {expr}, returning penalty")
            return 1000000
        
        # All predictions are valid - compute variance normally
        sample_variances = []
        for sample_idx in range(pred_matrix.shape[1]):
            sample_predictions = pred_matrix[:, sample_idx]
            
            if len(sample_predictions) < 2:
                # Not enough predictions for this sample
                sample_variances.append(1000000)
            else:
                sample_var = np.var(sample_predictions)
                # Check if variance itself is inf or nan
                if not np.isfinite(sample_var):
                    sample_variances.append(1000000)
                else:
                    sample_variances.append(sample_var)
        
        # Average variance across samples
        if len(sample_variances) == 0:
            return 1000000
        
        variance = np.mean(sample_variances)
        
        # Final check: ensure variance is finite
        if not np.isfinite(variance):
            print(f"WARNING: Prediction variance (MVP) is {variance} for equation {expr}, returning finite penalty")
            return 1000000
        
        return variance


    def compute_prediction_variance_sv(self, expr, X, perturb_percent=0.03):
        """
        Compute prediction variance via Single Variable Perturbation
        
        Perturb each variable one at a time
        For example, for 2 variables: [(x1+), (x1-), (x2+), (x2-)]
        """
        symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        if not symbols:
            return 0.0
            
        symbol_names = [str(s) for s in symbols]
        symbol_indices = []
        for name in symbol_names:
            if name.startswith("x") and name[1:].isdigit():
                symbol_indices.append(int(name[1:]))
            elif name.startswith("feature_") and name.split("_")[-1].isdigit():
                symbol_indices.append(int(name.split("_")[-1]) - 1)
            else:
                raise ValueError(f"Unsupported symbol name '{name}' in expression '{expr}'")
        X_sub = X[:, symbol_indices]
        
        # Perturb each variable one at a time
        predictions = []
        try:
            f = sp.lambdify(symbols, expr, modules='numpy')

            # Perturb each variable one at a time
            for var_idx in range(len(symbols)):
                X_perturbed_up = X_sub.copy()
                X_perturbed_down = X_sub.copy()

                # Perturb only the current variable
                X_perturbed_up[:, var_idx] *= (1 + perturb_percent)
                X_perturbed_down[:, var_idx] *= (1 - perturb_percent)

                # Evaluate the expression
                try:
                    y_pred_up = f(*X_perturbed_up.T)
                except Exception as eval_error_up:
                    # Failed during evaluation of up perturbation
                    print(f"WARNING: Sensitivity calculation failed (SVP) during evaluation (UP): "
                          f"Equation={expr}, Variable={var_idx}, "
                          f"Error={type(eval_error_up).__name__}: {str(eval_error_up)}")
                    return 1000000  # Bad equation → high sensitivity (finite value instead of inf)
                
                try:
                    y_pred_down = f(*X_perturbed_down.T)
                except Exception as eval_error_down:
                    # Failed during evaluation of down perturbation
                    print(f"WARNING: Sensitivity calculation failed (SVP) during evaluation (DOWN): "
                          f"Equation={expr}, Variable={var_idx}, "
                          f"Error={type(eval_error_down).__name__}: {str(eval_error_down)}")
                    return 1000000  # Bad equation → high sensitivity (finite value instead of inf)
                
                predictions.extend([y_pred_up, y_pred_down])
        except Exception as e:
            # Equation failed during lambdify or other setup (e.g., unsupported operations)
            # Return high value to indicate BAD behavior (not 0.0 which looks "good")
            print(f"WARNING: Sensitivity calculation failed (SVP) during setup: "
                  f"Equation={expr}, Error={type(e).__name__}: {str(e)}")
            return 1000000  # Bad equation → high sensitivity (finite value instead of inf)
            
        if len(predictions) < 2:
            return 0.0
        pred_matrix = np.vstack(predictions)
        
        # Check for NaN/inf values - if ANY are found, penalize heavily
        finite_mask = np.isfinite(pred_matrix)
        n_invalid = pred_matrix.size - np.sum(finite_mask)
        
        # If ANY predictions are invalid (NaN or inf), return large penalty
        # This indicates the equation is unstable/unreliable
        if n_invalid > 0:
            print(f"WARNING: Prediction variance (SVP) detected {n_invalid} invalid predictions (NaN/inf) for equation {expr}, returning penalty")
            return 1000000
        
        # All predictions are valid - compute variance normally
        sample_variances = []
        for sample_idx in range(pred_matrix.shape[1]):
            sample_predictions = pred_matrix[:, sample_idx]
            
            if len(sample_predictions) < 2:
                # Not enough predictions for this sample
                sample_variances.append(1000000)
            else:
                sample_var = np.var(sample_predictions)
                # Check if variance itself is inf or nan
                if not np.isfinite(sample_var):
                    sample_variances.append(1000000)
                else:
                    sample_variances.append(sample_var)
        
        # Average variance across samples
        if len(sample_variances) == 0:
            return 1000000
        
        variance = np.mean(sample_variances)
        
        # Final check: ensure variance is finite
        if not np.isfinite(variance):
            print(f"WARNING: Prediction variance (SVP) is {variance} for equation {expr}, returning finite penalty")
            return 1000000
        
        return variance

    def calculate_hybrid_score(self, base_metric, extrap_div, sens_interp, sens_extrap, lambda1=0.5, lambda2=0.5, lambda3=0.5):
        """
        Calculate hybrid score combining base metric with extrapolation measures.
        base_metric: The base metric value (AIC, BIC, MDL, PSM, or MSE)
        extrap_div: Extrapolation diversity measure
        sens_interp: Sensitivity for interpolation
        sens_extrap: Sensitivity for extrapolation
        lambda1, lambda2, lambda3: Weighting parameters
        
        Returns:
            Hybrid score value
        """
        score = base_metric + lambda1 * extrap_div + lambda2 * sens_interp + lambda3 * sens_extrap
        return score
    
    def get_weight_combinations(self):
        """Generate weight combinations for lambda parameters (excluding 0.0)."""
        weights = [0.1, 0.3, 0.5, 0.7, 1.0]
        
        combinations = []
        for w1 in weights:
            for w2 in weights:
                for w3 in weights:
                    combinations.append((w1, w2, w3))
        
        # Strategic weight combinations for different evaluation focuses
        special_combinations = [
            # Equal weights - balanced approach
            (0.5, 0.5, 0.5),
            
            # Extrapolation focus - prioritise extrapolation divergence and extrapolation sensitivity
            (1.0, 0.2, 1.0),  # High extrap_div, low interp_sens, high extrap_sens
            (0.8, 0.3, 0.9),  # Strong extrapolation focus
            
            # Sensitivity focus - prioritize both interpolation and extrapolation sensitivity
            (0.3, 1.0, 1.0),  # Low extrap_div, high interp_sens, high extrap_sens
            (0.4, 0.9, 0.8),  # Strong sensitivity focus
            
            # No penalty - minimal weights on all penalty terms
            (0.1, 0.1, 0.1),  # Minimal penalties
            (0.0, 0.0, 0.0),  # No penalties at all
            
            # Light penalty - moderate weights
            (0.3, 0.3, 0.3),  # Light penalties
            (0.4, 0.2, 0.4),  # Light penalties with slight extrapolation focus
            
            # Heavy penalty - strong weights on all terms
            (1.0, 1.0, 1.0),  # Maximum penalties
            (0.8, 0.8, 0.8),  # Heavy penalties
            (0.9, 0.7, 0.9),  # Heavy penalties with extrapolation emphasis
        ]
        
        all_combinations = list(set(combinations + special_combinations))
        return sorted(all_combinations)
    
    def calculate_all_hybrid_scores(self, base_metric, extrap_div, sens_interp, sens_extrap):
        """Calculate hybrid scores for all weight combinations."""
        weight_combinations = self.get_weight_combinations()
        results = {}
        
        for lambda1, lambda2, lambda3 in weight_combinations:
            score = self.calculate_hybrid_score(base_metric, extrap_div, sens_interp, sens_extrap, lambda1, lambda2, lambda3)
            results[(lambda1, lambda2, lambda3)] = score
        
        return results



std_epsilon = 0.3

fit_values = []
X_synth_values, y_synth_values = {}, {}
equations_list = []
complexity_values = []
yhat_gpe_values = []


def save_synthetic_data(X_synth, y_synth, feature_names, filename, output_dir):
    if X_synth.shape[0] == y_synth.shape[0] and X_synth.shape[0] > 0:
        df = pd.DataFrame(X_synth, columns=feature_names)
        df["target"] = y_synth
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"{filename} saved.")
    else:
        print(f"Skipping {filename}: X_synth ({X_synth.shape[0]}) & y_synth ({y_synth.shape[0]}) length mismatch.")


def save_run_results_cv(gp_results, model_selection_criteria, hybrid_scores_sv, hybrid_scores_mv, 
                        run_num, output_dir):
    """Save results for a specific run in CV mode"""
    # Filter results for current run
    run_gp_results = [r for r in gp_results if r[0] == run_num]
    run_model_selection = [r for r in model_selection_criteria if r[0] == run_num]
    run_hybrid_sv = [r for r in hybrid_scores_sv if r[0] == run_num]
    run_hybrid_mv = [r for r in hybrid_scores_mv if r[0] == run_num]
    
    # Save GP results
    if run_gp_results:
        gp_results_df = pd.DataFrame(run_gp_results, columns=[
            "Run", "Fold", "Equation_Index",
            "RMSE_Train", "RMSE_Val", "RMSE_Test_Interpolation", "RMSE_Test_Extrapolation"
        ])
        gp_results_df.to_csv(os.path.join(output_dir, f"gp_model_evaluation_results_cv_run_{run_num}.csv"), index=False)
        print(f"Saved: gp_model_evaluation_results_cv_run_{run_num}.csv")
    
    # Save model selection criteria
    if run_model_selection:
        model_selection_df = pd.DataFrame(run_model_selection, columns=[
            "Run", "Fold", "Equation_Index", "Complexity", "Constant_Num", "Total_Symbols", "Distinct_Symbols",
            "MSE_Train", "MSE_Val", "MSE_Test_Interpolation", "MSE_Test_Extrapolation",
            "Sigma_Train", "Sigma_Val", "Log_likelihood_Train", "Log_likelihood_Val",
            "AIC_Train", "AIC_Val",
            "BIC_Train", "BIC_Val", "MDL_Train", "MDL_Val",
            "PSM_Train", "PSM_Val",
            "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
        ])
        model_selection_df.to_csv(os.path.join(output_dir, f"gp_model_selection_criteria_cv_run_{run_num}.csv"), index=False)
        print(f"Saved: gp_model_selection_criteria_cv_run_{run_num}.csv")
    
    # Save hybrid scores SV
    if run_hybrid_sv:
        hybrid_sv_df = pd.DataFrame(run_hybrid_sv, columns=[
            "Run", "Fold", "Equation_Index",
            "Hybrid_AIC_SVP", "Hybrid_BIC_SVP", "Hybrid_MDL_SVP", "Hybrid_PSM_SVP", 
            "Hybrid_MSE_Train_SVP", "Hybrid_MSE_Val_SVP", "Hybrid_MSE_Test_Int_SVP", "Hybrid_MSE_Test_Ext_SVP",
            "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Extrapolation_Divergence"
        ])
        hybrid_sv_df.to_csv(os.path.join(output_dir, f"hybrid_model_scores_sv_cv_run_{run_num}.csv"), index=False)
        print(f"Saved: hybrid_model_scores_sv_cv_run_{run_num}.csv")
    
    # Save hybrid scores MV
    if run_hybrid_mv:
        hybrid_mv_df = pd.DataFrame(run_hybrid_mv, columns=[
            "Run", "Fold", "Equation_Index",
            "Hybrid_AIC_MVP", "Hybrid_BIC_MVP", "Hybrid_MDL_MVP", "Hybrid_PSM_MVP", 
            "Hybrid_MSE_Train_MVP", "Hybrid_MSE_Val_MVP", "Hybrid_MSE_Test_Int_MVP", "Hybrid_MSE_Test_Ext_MVP",
            "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
        ])
        hybrid_mv_df.to_csv(os.path.join(output_dir, f"hybrid_model_scores_mv_cv_run_{run_num}.csv"), index=False)
        print(f"Saved: hybrid_model_scores_mv_cv_run_{run_num}.csv")


def save_run_results_noncv(gp_results, model_selection_criteria, hybrid_scores_sv, hybrid_scores_mv,
                           run_num, output_dir):
    """Save results for a specific run in non-CV mode"""
    # Filter results for current run
    run_gp_results = [r for r in gp_results if r[0] == run_num]
    run_model_selection = [r for r in model_selection_criteria if r[0] == run_num]
    run_hybrid_sv = [r for r in hybrid_scores_sv if r[0] == run_num]
    run_hybrid_mv = [r for r in hybrid_scores_mv if r[0] == run_num]
    
    # Save GP results
    if run_gp_results:
        gp_results_df = pd.DataFrame(run_gp_results, columns=[
            "Run", "Equation_Index",
            "RMSE_Train", "RMSE_Test_Interpolation", "RMSE_Test_Extrapolation"
        ])
        gp_results_df.to_csv(os.path.join(output_dir, f"gp_model_evaluation_results_run_{run_num}.csv"), index=False)
        print(f"Saved: gp_model_evaluation_results_run_{run_num}.csv")
    
    # Save model selection criteria
    if run_model_selection:
        model_selection_df = pd.DataFrame(run_model_selection, columns=[
            "Run", "Equation_Index", "Complexity", "Constant_Num", "Total_Symbols", "Distinct_Symbols",
            "MSE_Train", "MSE_Test_Interpolation", "MSE_Test_Extrapolation",
            "Sigma_Train", "Log_likelihood_Train",
            "AIC_Train", "BIC_Train", "MDL_Train", "PSM_Train",
            "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
        ])
        model_selection_df.to_csv(os.path.join(output_dir, f"gp_model_selection_criteria_run_{run_num}.csv"), index=False)
        print(f"Saved: gp_model_selection_criteria_run_{run_num}.csv")
    
    # Save hybrid scores SV
    if run_hybrid_sv:
        hybrid_sv_df = pd.DataFrame(run_hybrid_sv, columns=[
            "Run", "Equation_Index",
            "Hybrid_AIC_SVP", "Hybrid_BIC_SVP", "Hybrid_MDL_SVP", "Hybrid_PSM_SVP", 
            "Hybrid_MSE_Train_SVP", "Hybrid_MSE_Test_Int_SVP", "Hybrid_MSE_Test_Ext_SVP",
            "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Extrapolation_Divergence"
        ])
        hybrid_sv_df.to_csv(os.path.join(output_dir, f"hybrid_model_scores_sv_run_{run_num}.csv"), index=False)
        print(f"Saved: hybrid_model_scores_sv_run_{run_num}.csv")
    
    # Save hybrid scores MV
    if run_hybrid_mv:
        hybrid_mv_df = pd.DataFrame(run_hybrid_mv, columns=[
            "Run", "Equation_Index",
            "Hybrid_AIC_MVP", "Hybrid_BIC_MVP", "Hybrid_MDL_MVP", "Hybrid_PSM_MVP", 
            "Hybrid_MSE_Train_MVP", "Hybrid_MSE_Test_Int_MVP", "Hybrid_MSE_Test_Ext_MVP",
            "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
        ])
        hybrid_mv_df.to_csv(os.path.join(output_dir, f"hybrid_model_scores_mv_run_{run_num}.csv"), index=False)
        print(f"Saved: hybrid_model_scores_mv_run_{run_num}.csv")


def check_predictions_for_nan(predictions_dict, equation_index, equation_str, output_dir, run=None, fold=None):
    """
    Check if any predictions contain NaN or infinity values.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with keys like 'train', 'val', 'test_int', 'test_ext', 'synthetic'
        and values as numpy arrays of predictions
    equation_index : int
        Index of the equation being evaluated
    equation_str : str
        String representation of the equation
    output_dir : str
        Directory to save NaN/infinity reports
    run : int, optional
        Run number
    fold : int, optional
        Fold number (for CV mode)
    
    Returns:
    --------
    bool : True if NaN or infinity detected, False otherwise
    dict : Dictionary with invalid counts per prediction type
    """
    has_invalid = False
    invalid_counts = {}
    
    for pred_type, pred_array in predictions_dict.items():
        if pred_array is not None and len(pred_array) > 0:
            # Check for both NaN and infinity
            invalid_count = np.sum(~np.isfinite(pred_array))
            invalid_counts[pred_type] = invalid_count
            if invalid_count > 0:
                has_invalid = True
    
    if has_invalid:
        # Log which predictions have NaN
        run_str = f"_run{run}" if run is not None else ""
        fold_str = f"_fold{fold}" if fold is not None else ""
        safe_name = f"equation_{equation_index}{run_str}{fold_str}".replace(' ', '_')
        
        # Save invalid values report
        invalid_report = []
        invalid_report.append(f"NaN/Infinity DETECTED in predictions for equation {equation_index}")
        invalid_report.append(f"Equation: {equation_str}")
        invalid_report.append(f"Run: {run if run is not None else 'N/A'}")
        invalid_report.append(f"Fold: {fold if fold is not None else 'N/A'}")
        invalid_report.append("\nInvalid (NaN/infinity) counts per prediction type:")
        
        invalid_csv_rows = []
        for pred_type, invalid_count in invalid_counts.items():
            if invalid_count > 0:
                pred_array = predictions_dict[pred_type]
                nan_count = np.sum(np.isnan(pred_array))
                inf_count = np.sum(np.isinf(pred_array))
                invalid_report.append(f"  {pred_type}: {invalid_count} invalid values (NaN: {nan_count}, Infinity: {inf_count}) out of {len(pred_array)}")
                # Find which indices have NaN or infinity
                invalid_indices = np.where(~np.isfinite(pred_array))[0]
                nan_indices = np.where(np.isnan(pred_array))[0]
                inf_indices = np.where(np.isinf(pred_array))[0]
                invalid_report.append(f"    NaN at indices: {list(nan_indices)}")
                invalid_report.append(f"    Infinity at indices: {list(inf_indices)}")
                
                # Add to CSV
                for idx in invalid_indices:
                    is_nan = np.isnan(pred_array[idx])
                    is_inf = np.isinf(pred_array[idx])
                    invalid_csv_rows.append({
                        'equation_index': equation_index,
                        'equation': equation_str,
                        'run': run if run is not None else None,
                        'fold': fold if fold is not None else None,
                        'prediction_type': pred_type,
                        'row_index': int(idx),
                        'is_nan': is_nan,
                        'is_inf': is_inf,
                        'total_predictions': len(pred_array),
                        'invalid_count': invalid_count
                    })
        
        # Save report
        txt_file = os.path.join(output_dir, f"invalid_predictions_{safe_name}.txt")
        with open(txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NaN/INFINITY PREDICTIONS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write("\n".join(invalid_report))
            f.write("\n" + "="*80 + "\n")
        
        if invalid_csv_rows:
            csv_file = os.path.join(output_dir, f"invalid_predictions_{safe_name}.csv")
            pd.DataFrame(invalid_csv_rows).to_csv(csv_file, index=False)
        
        print(f"\n[WARNING] Equation {equation_index} produces NaN/infinity predictions - SKIPPING")
        print(f"  Equation: {equation_str}")
        for pred_type, invalid_count in invalid_counts.items():
            if invalid_count > 0:
                pred_array = predictions_dict[pred_type]
                nan_count = np.sum(np.isnan(pred_array))
                inf_count = np.sum(np.isinf(pred_array))
                print(f"  {pred_type}: {invalid_count}/{len(pred_array)} invalid (NaN: {nan_count}, Infinity: {inf_count})")
        print(f"  Reports saved to: {txt_file}")
        if invalid_csv_rows:
            print(f"  Details saved to: {csv_file}")
    
    return has_invalid, invalid_counts


def evaluate_models(bandwidths, feature_names, output_dir, num_runs=2, use_cv=True, n_splits=3, start_run=1, end_run=None):
    # Determine run range
    if end_run is None:
        end_run = num_runs
    start_run = max(1, start_run)
    end_run = min(num_runs, end_run)
    
    gp_results = []
    model_selection_criteria = []
    hybrid_scores_sv = []
    hybrid_scores_mv = []
    
    yhat_gpe_values = []
    
    # List to collect all equations across all runs (one Pareto front per run/fold)
    all_equations_list = []
    
    # Track skipped equations due to NaN
    skipped_equations = []

    def to_float(x):
        """handling SymPy expressions"""
        try:
            return float(x)
        except Exception:
            try:
                if hasattr(x, 'evalf'):
                    x = x.evalf()
                return float(x)
            except Exception:
                try:
                    return float(sp.N(x).evalf())
                except Exception:
                    return float('nan')

    for bandwidth in bandwidths:
        if use_cv:
            # CV mode
            any_fold = False

            # for run in range(num_runs):
            for run in range(start_run - 1, end_run):
                start_time = time.time()
                # Folds will be iterated inside each run
                # Use run as random_state to get different CV splits for each run
                for (fold_idx,
                     X_train, y_train,
                     X_val, y_val,
                     X_test_interpolation, y_test_interpolation,
                     X_test_extrapolation, y_test_extrapolation) in split_data_kfold(
                        bandwidth, feature_names, output_dir, n_splits=n_splits, random_state=run
                ):  # find the data split
                    any_fold = True
                    num_features = X_train.shape[1]
                    models = get_models()
                    evaluator = ModelEvaluator()

                    print(f"Run {run + 1}/{num_runs} - BW: {bandwidth:.3f} - Fold: {fold_idx}/{n_splits}")

                    # per-run, per-fold synthetic stores
                    X_synth_values, y_synth_values = {}, {}

                    for model_name, model in models.items():   # find the model
                        model.set_params(random_state=run)
                        
                        # Check for NaN before fitting
                        if np.any(np.isnan(X_train)):
                            print(f"\n[ERROR] NaN detected in X_train before fitting {model_name}")
                            print(f"  Shape: {X_train.shape}")
                            nan_rows = np.where(np.any(np.isnan(X_train), axis=1))[0]
                            nan_cols = np.where(np.any(np.isnan(X_train), axis=0))[0]
                            print(f"  Rows with NaN: {nan_rows}")
                            print(f"  Columns with NaN: {nan_cols}")
                            
                            # Save NaN details to file
                            nan_csv_rows = []
                            for col_idx in nan_cols:
                                col_nan_rows = np.where(np.isnan(X_train[:, col_idx]))[0]
                                col_name = feature_names[col_idx] if col_idx < len(feature_names) else f"col_{col_idx}"
                                print(f"    Column {col_idx} ({col_name}): NaN at rows {col_nan_rows}")
                                for row_idx in col_nan_rows:
                                    nan_csv_rows.append({
                                        'data_name': f'X_train_{model_name}_run{run+1}_fold{fold_idx}',
                                        'row_index': int(row_idx),
                                        'column_name': col_name,
                                        'column_index': int(col_idx),
                                        'value': None
                                    })
                            
                            if nan_csv_rows:
                                safe_name = f"X_train_{model_name}_run{run+1}_fold{fold_idx}".replace(' ', '_')
                                csv_file = os.path.join(output_dir, f"nan_detection_{safe_name}.csv")
                                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                                print(f"  Saved NaN details to: {csv_file}")
                            
                            raise ValueError(f"NaN values found in X_train for {model_name}")
                        
                        if np.any(np.isnan(y_train)):
                            print(f"\n[ERROR] NaN detected in y_train before fitting {model_name}")
                            print(f"  Shape: {y_train.shape}")
                            nan_indices = np.where(np.isnan(y_train))[0]
                            print(f"  NaN at indices: {nan_indices}")
                            
                            # Save NaN details to file
                            nan_csv_rows = [{
                                'data_name': f'y_train_{model_name}_run{run+1}_fold{fold_idx}',
                                'row_index': int(idx),
                                'column_name': 'target',
                                'column_index': None,
                                'value': None
                            } for idx in nan_indices]
                            
                            if nan_csv_rows:
                                safe_name = f"y_train_{model_name}_run{run+1}_fold{fold_idx}".replace(' ', '_')
                                csv_file = os.path.join(output_dir, f"nan_detection_{safe_name}.csv")
                                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                                print(f"  Saved NaN details to: {csv_file}")
                            
                            raise ValueError(f"NaN values found in y_train for {model_name}")
                        
                        fit = model.fit(X_train, y_train)
                        fit_values.append(fit)

                        # predictions
                        yhat_train = model.predict(X_train)
                        yhat_val = model.predict(X_val)
                        yhat_test_int = model.predict(X_test_interpolation)
                        yhat_test_ext = (model.predict(X_test_extrapolation)
                                    if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0
                                    else np.array([]))

                        # distances & residuals
                        centroid = np.mean(X_train, axis=0)  # find the centroid
                        dist_train = np.linalg.norm(X_train - centroid, axis=1)
                        dist_val = np.linalg.norm(X_val - centroid, axis=1)
                        dist_test_int = np.linalg.norm(X_test_interpolation - centroid, axis=1)
                        dist_ext = (np.linalg.norm(X_test_extrapolation - centroid, axis=1)
                                    if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0
                                    else np.array([]))

                        res_train = y_train.flatten() - yhat_train
                        res_val = y_val.flatten() - yhat_val
                        res_test_int = y_test_interpolation.flatten() - yhat_test_int
                        res_ext = ((y_test_extrapolation.flatten() - yhat_test_ext)
                                   if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0
                                   else np.array([]))

                        all_distances = np.concatenate([dist_train, dist_val, dist_test_int, dist_ext])
                        all_residuals = np.concatenate([np.abs(res_train), np.abs(res_val), np.abs(res_test_int), np.abs(res_ext)])

                        if np.any(np.isnan(all_residuals)) or np.any(np.isnan(all_distances)):
                            print("Found NaNs in inputs to LinearRegression; skipping this model/run")
                            continue

                        reg = LinearRegression().fit(all_distances.reshape(-1, 1), all_residuals)
                        line_x = np.linspace(all_distances.min(), all_distances.max(), 1000)
                        line_y = reg.predict(line_x.reshape(-1, 1))

                        plt.figure(figsize=(10, 8))
                        plt.scatter(dist_train, np.abs(res_train), alpha=0.5, label='Train Residuals', color='red', edgecolors='none')
                        plt.scatter(dist_val, np.abs(res_val), alpha=0.5, label='Val Residuals', color='blue', edgecolors='none')
                        plt.scatter(dist_test_int, np.abs(res_test_int), alpha=0.5, label='Test Interp Residuals', color='orange', edgecolors='none')
                        if dist_ext.size > 0:
                            plt.scatter(dist_ext, np.abs(res_ext), alpha=0.5, label='Extrap Residuals', color='green', edgecolors='none')
                        plt.plot(line_x, line_y, color='black', label='Fit Line')
                        plt.xlabel('Distance from Centroid', fontsize=16)
                        plt.ylabel('Absolute Residuals', fontsize=16)
                        plt.title(f'Residuals vs. Distance | {model_name} | Run {run + 1} | Fold {fold_idx}', fontsize=18)
                        plt.legend(fontsize=12)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"Residuals_vs_Distance_{model_name}_Run{run + 1}_Fold{fold_idx}.pdf"))  # save the figure
                        plt.savefig(os.path.join(output_dir, f"Residuals_vs_Distance_{model_name}_Run{run + 1}_Fold{fold_idx}.png"))  # save the figure
                        plt.close()

                        # synthetic generation from low-density training area
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_train)
                        log_scores = kde.score_samples(X_train)
                        num_required = max(10, int(0.1 * len(X_train)))
                        lowest_density_indices = np.argsort(log_scores)[:num_required]
                        X_outside_kde = X_train[lowest_density_indices]

                        X_synth, y_synth = [], []
                        if X_outside_kde.shape[0] > 0:
                            num_generated = 0
                            # while num_generated < 20% of original dataset
                            num_synthetic_samples = max(10, int(0.2 * len(X_train)))  # 20% of original dataset
                            while num_generated < num_synthetic_samples:
                                s = X_outside_kde[num_generated % X_outside_kde.shape[0]] + np.random.normal(0, std_epsilon, size=num_features)
                                X_synth.append(s)
                                y_synth.append(model.predict(s.reshape(1, -1))[0])
                                num_generated += 1

                            X_synth_values[model_name] = np.array(X_synth)
                            y_synth_values[model_name] = np.array(y_synth)
                            
                            if model_name == "PySR":  # Only print once
                                print(f"      Synthetic data: {len(X_synth_values[model_name])} samples generated (20% of train size)")

                            save_synthetic_data(
                                X_synth_values[model_name], y_synth_values[model_name], feature_names,
                                f"X_synth_{model_name.lower()}_run{run + 1}_fold{fold_idx}.csv", output_dir
                            )

                    # PySR metrics per equation (with VAL/EXT)
                    if "PySR" in models:
                        gp_model = models["PySR"]
                        yhat_gpe = None
                        min_rmse = float("inf")

                        for j in range(gp_model.equations_.shape[0]):
                            equations_df = gp_model.equations_
                            equation = gp_model.sympy(j)
                            complexity = equations_df.iloc[j, 0]
                            # constants count (k)
                            _, k = evaluator.count_constants(equation)
                            w, _ = evaluator.count_total_symbols(equation)
                            _, u = evaluator.count_distinct_symbols(equation)

                            # predictions
                            yhat_train_pysr = gp_model.predict(X_train, index=j)
                            yhat_val_pysr = gp_model.predict(X_val, index=j)
                            yhat_test_int_pysr = gp_model.predict(X_test_interpolation, index=j)
                            yhat_test_ext_pysr = (gp_model.predict(X_test_extrapolation, index=j)
                                             if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0
                                             else np.array([]))
                            
                            # synthetic for PySR
                            if "PySR" not in X_synth_values or X_synth_values["PySR"] is None or X_synth_values["PySR"].size == 0:
                                if len(X_synth_values) > 0:
                                    first_model = next(iter(X_synth_values))
                                    X_synth = X_synth_values[first_model]
                                    y_synth = y_synth_values[first_model]
                                else:
                                    # Skip this equation if no synthetic data is available
                                    continue
                            else:
                                X_synth = X_synth_values["PySR"]
                                y_synth = y_synth_values["PySR"]
                            
                            yhat_train_ext_pysr = gp_model.predict(X_synth, index=j)
                            
                            # Check for NaN in predictions - skip this equation if found
                            equation_str = str(equation)
                            predictions_dict = {
                                'train': yhat_train_pysr,
                                'val': yhat_val_pysr,
                                'test_int': yhat_test_int_pysr,
                                'test_ext': yhat_test_ext_pysr if len(yhat_test_ext_pysr) > 0 else None,
                                'synthetic': yhat_train_ext_pysr
                            }
                            
                            has_invalid, invalid_counts = check_predictions_for_nan(
                                predictions_dict, j, equation_str, output_dir, run=run+1, fold=fold_idx
                            )
                            
                            if has_invalid:
                                skipped_equations.append({
                                    'run': run + 1,
                                    'fold': fold_idx,
                                    'equation_index': j,
                                    'equation': equation_str,
                                    'nan_counts': invalid_counts  # Keep name for backward compatibility
                                })
                                continue  # Skip this equation

                            # Calculate metrics (with NaN filtering)
                            mse_train = mean_squared_error(y_train, yhat_train_pysr)
                            mse_val = mean_squared_error(y_val, yhat_val_pysr)
                            mse_test_int = mean_squared_error(y_test_interpolation, yhat_test_int_pysr)
                            mse_test_ext = mean_squared_error(y_test_extrapolation, yhat_test_ext_pysr) if len(y_test_extrapolation) > 0 else np.nan

                            loglike_train, sigma_train = evaluator.gaussian_likelihood(y_train, yhat_train_pysr)
                            loglike_val, sigma_val = evaluator.gaussian_likelihood(y_val, yhat_val_pysr)
                            
                            aic_train = evaluator.calculate_aic(loglike_train, k)
                            aic_val = evaluator.calculate_aic(loglike_val, k)
                            bic_train = evaluator.calculate_bic(loglike_train, k, len(y_train))
                            bic_val = evaluator.calculate_bic(loglike_val, k, len(y_val))
                            fim = evaluator.fisher_information_matrix(
                                lambda p: evaluator.gaussian_likelihood_with_sigma(y_train, yhat_train_pysr, p[0]),
                                jnp.array([sigma_train], dtype=jnp.float32))
                            mdl_train = evaluator.calculate_mdl(loglike_train, fim, [sigma_train], equation)
                            fim_val = evaluator.fisher_information_matrix(
                                lambda p: evaluator.gaussian_likelihood_with_sigma(y_val, yhat_val_pysr, p[0]),
                                jnp.array([sigma_val], dtype=jnp.float32))
                            mdl_val = evaluator.calculate_mdl(loglike_val, fim_val, [sigma_val], equation)
                            
                            # PSM (PySR Score Metric) - read from PySR equations CSV file (score column)
                            # Read score directly from gp_model.equations_ DataFrame
                            try:
                                if 'score' in equations_df.columns:
                                    pysr_score = to_float(equations_df.iloc[j]['score'])
                                    # Use the same score for validation variant
                                    pysr_score_val = pysr_score
                                else:
                                    # Fallback if score column doesn't exist
                                    pysr_score = np.nan
                                    pysr_score_val = np.nan
                                    print(f"WARNING: 'score' column not found in equations_df for equation {j}")
                            except (KeyError, IndexError, ValueError) as e:
                                # Fallback if score cannot be read
                                pysr_score = np.nan
                                pysr_score_val = np.nan
                                print(f"WARNING: Could not read score from equations_df for equation {j}: {type(e).__name__}: {str(e)}")

                            rmse_train = np.sqrt(mse_train)
                            rmse_val = np.sqrt(mse_val)
                            rmse_test_int = np.sqrt(mse_test_int)
                            rmse_test_ext = np.sqrt(mse_test_ext)
                            
                            # Compute extrapolation and sensitivity measures
                            extrap_div = evaluator.compute_extrapolation_divergence(yhat_train_pysr, yhat_train_ext_pysr)
                            sens_interp_sv = evaluator.compute_prediction_variance_sv(equation, X_train)
                            sens_extrap_sv = evaluator.compute_prediction_variance_sv(equation, X_synth)
                            sens_interp_mv = evaluator.compute_prediction_variance_mv(equation, X_train)
                            sens_extrap_mv = evaluator.compute_prediction_variance_mv(equation, X_synth)
                            
                            gp_results.append([run + 1, fold_idx, j, rmse_train, rmse_val, rmse_test_int, rmse_test_ext])

                            model_selection_criteria.append([
                                run + 1, fold_idx, j, complexity, k, w, u,
                                mse_train, mse_val, mse_test_int, mse_test_ext,
                                sigma_train, sigma_val, float(loglike_train), float(loglike_val), aic_train, aic_val,
                                bic_train, bic_val, mdl_train, mdl_val, pysr_score, pysr_score_val,
                                to_float(sens_interp_sv), to_float(sens_extrap_sv), to_float(sens_interp_mv), to_float(sens_extrap_mv), to_float(extrap_div)
                            ])

                            if not np.isnan(rmse_train) and rmse_train < min_rmse:
                                min_rmse = rmse_train
                                yhat_gpe = yhat_train_ext_pysr

                            hybrid_scores_sv.append([
                                run + 1, fold_idx, j,
                                to_float(evaluator.calculate_hybrid_score(aic_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(bic_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(mdl_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(pysr_score, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(mse_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(mse_val, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(mse_test_int, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(evaluator.calculate_hybrid_score(mse_test_ext, extrap_div, sens_interp_sv, sens_extrap_sv)),
                                to_float(sens_interp_sv), to_float(sens_extrap_sv), to_float(extrap_div)
                            ])

                            hybrid_scores_mv.append([
                                run + 1, fold_idx, j,
                                to_float(evaluator.calculate_hybrid_score(aic_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(bic_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(mdl_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(pysr_score, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(mse_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(mse_val, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(mse_test_int, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(evaluator.calculate_hybrid_score(mse_test_ext, extrap_div, sens_interp_mv, sens_extrap_mv)),
                                to_float(sens_interp_mv), to_float(sens_extrap_mv), to_float(extrap_div)
                            ])

                        # save yhat_gpe for this run/fold
                        if yhat_gpe is not None:
                            yhat_gpe_df = pd.DataFrame(yhat_gpe, columns=["yhat_gpe"])
                            yhat_gpe_filename = os.path.join(output_dir, f"yhat_gpe_run_{run + 1}_fold_{fold_idx}.csv")
                            yhat_gpe_df.to_csv(yhat_gpe_filename, index=False)
                            print(f"Saved yhat_gpe values for Run {run + 1}, Fold {fold_idx} to {yhat_gpe_filename}")

                        # Save PySR Pareto front equations for this run/fold
                        if "PySR" in models:
                            gp_model = models["PySR"]
                            equations_df = gp_model.equations_.copy()
                            equations_df['run'] = run + 1
                            equations_df['fold'] = fold_idx
                            equations_df['equation_index'] = range(len(equations_df))
                            
                            # Add equation strings
                            equation_strings = []
                            for j in range(len(equations_df)):
                                try:
                                    equation_str = str(gp_model.sympy(j))
                                    equation_strings.append(equation_str)
                                except:
                                    equation_strings.append("Error parsing equation")
                            equations_df['equation_string'] = equation_strings
                            
                            # Save individual run/fold file
                            pysr_equations_filename = os.path.join(output_dir, f"pysr_equations_run_{run + 1}_fold_{fold_idx}.csv")
                            equations_df.to_csv(pysr_equations_filename, index=False)
                            print(f"Saved PySR Pareto front equations for Run {run + 1}, Fold {fold_idx} to {pysr_equations_filename}")
                            
                            # Append to combined list
                            all_equations_list.append(equations_df)

                    end_time = time.time()
                    print("Fold runtime (s): ", end_time - start_time)

                    # per-fold distribution plot only on last run
                    if run == num_runs - 1:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(X_train[:, 0], X_train[:, 1], c='red', label="Train (fold)", alpha=0.7)
                        plt.scatter(X_val[:, 0], X_val[:, 1], c='blue', label="Val (fold)", alpha=0.7)
                        plt.scatter(X_test_interpolation[:, 0], X_test_interpolation[:, 1], c='orange', label="Test Interpolation", alpha=0.7)
                        if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0:
                            plt.scatter(X_test_extrapolation[:, 0], X_test_extrapolation[:, 1], c='green', label="Test Extrapolation", alpha=0.7)
                        if 'X_synth' in locals() and X_synth is not None and X_synth.shape[0] > 0:
                            plt.scatter(X_synth[:, 0], X_synth[:, 1], c='purple', label="Synthetic", alpha=0.7)
                        plt.xlabel(feature_names[0])
                        plt.ylabel(feature_names[1])
                        plt.title(f"Distribution | Fold {fold_idx}")
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(os.path.join(output_dir, f"plot_last_run_fold_{fold_idx}.png"))
                        plt.savefig(os.path.join(output_dir, f"plot_last_run_fold_{fold_idx}.pdf"))
                        plt.close()
                
                # Save results for this run after all folds complete
                if any_fold:
                    save_run_results_cv(gp_results, model_selection_criteria, hybrid_scores_sv, hybrid_scores_mv,
                                       run + 1, output_dir)
                    print(f"Saved results for Run {run + 1} after all folds completed.")
                    
                    # Save summary of skipped equations for this run
                    run_skipped = [eq for eq in skipped_equations if eq['run'] == run + 1]
                    if run_skipped:
                        # Flatten nan_counts for CSV
                        skipped_rows = []
                        for eq in run_skipped:
                            row = {
                                'run': eq['run'],
                                'fold': eq['fold'],
                                'equation_index': eq['equation_index'],
                                'equation': eq['equation']
                            }
                            # Add NaN counts per prediction type
                            for pred_type, count in eq['nan_counts'].items():
                                row[f'nan_count_{pred_type}'] = count
                            skipped_rows.append(row)
                        
                        skipped_df = pd.DataFrame(skipped_rows)
                        skipped_file = os.path.join(output_dir, f"skipped_equations_run_{run + 1}.csv")
                        skipped_df.to_csv(skipped_file, index=False)
                        print(f"  Skipped {len(run_skipped)} equations due to NaN/infinity predictions (saved to {skipped_file})")

            if not any_fold:
                print(f"No inside/outside split found for bandwidth={bandwidth:.2f}. Skipping.")

        else:
            # original MODE / non-CV
            num_features = None
            models = get_models()
            evaluator = ModelEvaluator()

            # for run in range(num_runs):
            for run in range(start_run - 1, end_run):
                start_time = time.time()
                print(f"Run {run + 1}/{num_runs} - Bandwidth: {bandwidth:.3f}")
                
                # Use run as random_state to get different train/test splits for each run
                data_split = split_data(bandwidth, feature_names, output_dir, random_state=run)
                if data_split is None:
                    print(f"No split data for bandwidth={bandwidth:.2f} in run {run + 1}. Skipping.")
                    continue

                (X_train, y_train,
                 X_test_interpolation, y_test_interpolation,
                 X_test_extrapolation, y_test_extrapolation) = data_split
                
                if num_features is None:
                    num_features = X_train.shape[1]

                # per-run synthetic stores
                X_synth_values, y_synth_values = {}, {}

                for model_name, model in models.items():
                    model.set_params(random_state=run)
                    
                    # Check for NaN before fitting
                    if np.any(np.isnan(X_train)):
                        print(f"\n[ERROR] NaN detected in X_train before fitting {model_name}")
                        print(f"  Shape: {X_train.shape}")
                        nan_rows = np.where(np.any(np.isnan(X_train), axis=1))[0]
                        nan_cols = np.where(np.any(np.isnan(X_train), axis=0))[0]
                        print(f"  Rows with NaN: {nan_rows}")
                        print(f"  Columns with NaN: {nan_cols}")
                        
                        # Save NaN details to file
                        nan_csv_rows = []
                        for col_idx in nan_cols:
                            col_nan_rows = np.where(np.isnan(X_train[:, col_idx]))[0]
                            col_name = feature_names[col_idx] if col_idx < len(feature_names) else f"col_{col_idx}"
                            print(f"    Column {col_idx} ({col_name}): NaN at rows {col_nan_rows}")
                            for row_idx in col_nan_rows:
                                nan_csv_rows.append({
                                    'data_name': f'X_train_{model_name}_run{run+1}',
                                    'row_index': int(row_idx),
                                    'column_name': col_name,
                                    'column_index': int(col_idx),
                                    'value': None
                                })
                        
                        if nan_csv_rows:
                            safe_name = f"X_train_{model_name}_run{run+1}".replace(' ', '_')
                            csv_file = os.path.join(output_dir, f"nan_detection_{safe_name}.csv")
                            pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                            print(f"  Saved NaN details to: {csv_file}")
                        
                        raise ValueError(f"NaN values found in X_train for {model_name}")
                    
                    if np.any(np.isnan(y_train)):
                        print(f"\n[ERROR] NaN detected in y_train before fitting {model_name}")
                        print(f"  Shape: {y_train.shape}")
                        nan_indices = np.where(np.isnan(y_train))[0]
                        print(f"  NaN at indices: {nan_indices}")
                        
                        # Save NaN details to file
                        nan_csv_rows = [{
                            'data_name': f'y_train_{model_name}_run{run+1}',
                            'row_index': int(idx),
                            'column_name': 'target',
                            'column_index': None,
                            'value': None
                        } for idx in nan_indices]
                        
                        if nan_csv_rows:
                            safe_name = f"y_train_{model_name}_run{run+1}".replace(' ', '_')
                            csv_file = os.path.join(output_dir, f"nan_detection_{safe_name}.csv")
                            pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                            print(f"  Saved NaN details to: {csv_file}")
                        
                        raise ValueError(f"NaN values found in y_train for {model_name}")
                    
                    fit = model.fit(X_train, y_train)
                    fit_values.append(fit)

                    yhat_train = model.predict(X_train)
                    yhat_test_int = model.predict(X_test_interpolation)
                    yhat_test_ext = model.predict(X_test_extrapolation) if X_test_extrapolation.shape[0] > 0 else np.array([])

                    # distances & residuals
                    centroid = np.mean(X_train, axis=0)
                    train_distances = np.linalg.norm(X_train - centroid, axis=1)
                    test_interpolation_distances = np.linalg.norm(X_test_interpolation - centroid, axis=1)
                    test_extrapolation_distances = np.linalg.norm(X_test_extrapolation - centroid, axis=1) if X_test_extrapolation.shape[0] > 0 else np.array([])

                    y_train_residuals = y_train.flatten() - yhat_train
                    y_test_interpolation_residuals = y_test_interpolation.flatten() - yhat_test_int
                    y_test_extrapolation_residuals = y_test_extrapolation.flatten() - yhat_test_ext if X_test_extrapolation.shape[0] > 0 else np.array([])

                    all_distances = np.concatenate([train_distances, test_interpolation_distances, test_extrapolation_distances])
                    all_residuals = np.concatenate([np.abs(y_train_residuals), np.abs(y_test_interpolation_residuals), np.abs(y_test_extrapolation_residuals)])

                    if np.any(np.isnan(all_residuals)) or np.any(np.isnan(all_distances)):
                        print("Found NaNs in inputs to LinearRegression")
                        continue

                    reg = LinearRegression().fit(all_distances.reshape(-1, 1), all_residuals)
                    line_x = np.linspace(all_distances.min(), all_distances.max(), 1000)
                    line_y = reg.predict(line_x.reshape(-1, 1))

                    plt.figure(figsize=(10, 8))
                    plt.scatter(train_distances, np.abs(y_train_residuals), alpha=0.5, label='Train Residuals', color='red', edgecolors='none')
                    plt.scatter(test_interpolation_distances, np.abs(y_test_interpolation_residuals), alpha=0.5, label='Test Interpolation Residuals', color='blue', edgecolors='none')
                    if X_test_extrapolation.shape[0] > 0:
                        plt.scatter(test_extrapolation_distances, np.abs(y_test_extrapolation_residuals), alpha=0.5, label='Test Extrapolation Residuals', color='green', edgecolors='none')
                    plt.plot(line_x, line_y, color='black', label='Fit Line')
                    plt.xlabel('Distance from Centroid', fontsize=16)
                    plt.ylabel('Absolute Residuals', fontsize=16)
                    plt.title(f'Residuals vs. Distance\nModel: {model_name}, Run: {run + 1}', fontsize=18)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"Residuals_vs_Distance_{model_name}_Run{run + 1}.pdf"))
                    plt.savefig(os.path.join(output_dir, f"Residuals_vs_Distance_{model_name}_Run{run + 1}.png"))
                    plt.close()

                    # synthetic generation from low-density training region
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
                    kde.fit(X_train)
                    log_scores = kde.score_samples(X_train)
                    num_required = max(10, int(0.1 * len(X_train)))
                    lowest_density_indices = np.argsort(log_scores)[:num_required]
                    X_outside_kde = X_train[lowest_density_indices]

                    X_synth = []
                    y_synth = []
                    if X_outside_kde.shape[0] > 0:
                        num_generated = 0
                        
                        num_synthetic_samples = max(10, int(0.2 * len(X_train)))  # 20% of original dataset
                        while num_generated < num_synthetic_samples:
                            synthetic_sample = X_outside_kde[num_generated % X_outside_kde.shape[0]] + np.random.normal(0, std_epsilon, size=num_features)
                            X_synth.append(synthetic_sample)
                            y_synth.append(model.predict(synthetic_sample.reshape(1, -1))[0])
                            num_generated += 1

                        X_synth_values[model_name] = np.array(X_synth)
                        y_synth_values[model_name] = np.array(y_synth)
                        
                        if model_name == "PySR":  # Only print once
                            print(f"    Synthetic data: {len(X_synth_values[model_name])} samples generated (20% of train size)")

                        save_synthetic_data(
                            X_synth_values[model_name], y_synth_values[model_name], feature_names,
                            f"X_synth_{model_name.lower()}_{run + 1}.csv", output_dir
                        )

                # PySR per-equation metrics (TRAIN/INTERP/EXTRAP)
                if "PySR" in models:
                    gp_model = models["PySR"]
                    yhat_gpe = None
                    min_rmse = float("inf")

                    for j in range(gp_model.equations_.shape[0]):
                        equations_df = gp_model.equations_
                        equation = gp_model.sympy(j)
                        complexity = equations_df.iloc[j, 0]
                        _, k = evaluator.count_constants(equation)
                        w, _ = evaluator.count_total_symbols(equation)
                        _, u = evaluator.count_distinct_symbols(equation)

                        # predictions
                        yhat_train_pysr = gp_model.predict(X_train, index=j)
                        yhat_test_int_pysr = gp_model.predict(X_test_interpolation, index=j)
                        yhat_test_ext_pysr = (gp_model.predict(X_test_extrapolation, index=j)
                                             if X_test_extrapolation is not None and X_test_extrapolation.shape[0] > 0
                                             else np.array([]))

                        # synthetic for PySR
                        if "PySR" not in X_synth_values or X_synth_values["PySR"] is None or X_synth_values["PySR"].size == 0:
                            if len(X_synth_values) > 0:
                                first_model = next(iter(X_synth_values))
                                X_synth = X_synth_values[first_model]
                                y_synth = y_synth_values[first_model]
                            else:
                                # Skip this equation if no synthetic data is available
                                continue
                        else:
                            X_synth = X_synth_values["PySR"]
                            y_synth = y_synth_values["PySR"]

                        yhat_train_ext_pysr = gp_model.predict(X_synth, index=j)
                        
                        # Check for NaN in predictions - skip this equation if found
                        equation_str = str(equation)
                        predictions_dict = {
                            'train': yhat_train_pysr,
                            'test_int': yhat_test_int_pysr,
                            'test_ext': yhat_test_ext_pysr if len(yhat_test_ext_pysr) > 0 else None,
                            'synthetic': yhat_train_ext_pysr
                        }
                        
                        has_invalid, invalid_counts = check_predictions_for_nan(
                            predictions_dict, j, equation_str, output_dir, run=run+1
                        )
                        
                        if has_invalid:
                            skipped_equations.append({
                                'run': run + 1,
                                'fold': None,
                                'equation_index': j,
                                'equation': equation_str,
                                'nan_counts': invalid_counts  # Keep name for backward compatibility
                            })
                            continue  # Skip this equation

                        # Calculate metrics (with NaN filtering)
                        mse_train = mean_squared_error(y_train, yhat_train_pysr)
                        mse_test_int = mean_squared_error(y_test_interpolation, yhat_test_int_pysr)
                        mse_test_ext = mean_squared_error(y_test_extrapolation, yhat_test_ext_pysr) if len(y_test_extrapolation) > 0 else np.nan
                        
                        loglike_train, sigma_train = evaluator.gaussian_likelihood(y_train, yhat_train_pysr)
                        
                        aic_train = evaluator.calculate_aic(loglike_train, k)
                        bic_train = evaluator.calculate_bic(loglike_train, k, len(y_train))
                        fim_train = evaluator.fisher_information_matrix(
                            lambda p: evaluator.gaussian_likelihood_with_sigma(y_train, yhat_train_pysr, p[0]),
                            jnp.array([sigma_train], dtype=jnp.float32))
                        mdl_train = evaluator.calculate_mdl(loglike_train, fim_train, [sigma_train], equation)
                        
                        # PSM (PySR Score Metric) - read from PySR equations CSV file (score column)
                        # Read score directly from gp_model.equations_ DataFrame
                        equations_df = gp_model.equations_
                        try:
                            if 'score' in equations_df.columns:
                                pysr_score = to_float(equations_df.iloc[j]['score'])
                                # Use the same score for extrapolation variant
                                pysr_score_ext = pysr_score
                            else:
                                # Fallback if score column doesn't exist
                                pysr_score = np.nan
                                pysr_score_ext = np.nan
                                print(f"WARNING: 'score' column not found in equations_df for equation {j}")
                        except (KeyError, IndexError, ValueError) as e:
                            # Fallback if score cannot be read
                            pysr_score = np.nan
                            pysr_score_ext = np.nan
                            print(f"WARNING: Could not read score from equations_df for equation {j}: {type(e).__name__}: {str(e)}")

                        rmse_train = np.sqrt(mse_train)
                        rmse_test_int = np.sqrt(mse_test_int)
                        rmse_test_ext = np.sqrt(mse_test_ext)
                        
                        # Compute extrapolation and sensitivity measures
                        extrap_div = evaluator.compute_extrapolation_divergence(yhat_train_pysr, yhat_train_ext_pysr)
                        sens_interp_sv = evaluator.compute_prediction_variance_sv(equation, X_train)
                        sens_extrap_sv = evaluator.compute_prediction_variance_sv(equation, X_synth)
                        sens_interp_mv = evaluator.compute_prediction_variance_mv(equation, X_train)
                        sens_extrap_mv = evaluator.compute_prediction_variance_mv(equation, X_synth)
                        
                        gp_results.append([run + 1, j, rmse_train, rmse_test_int, rmse_test_ext])

                        model_selection_criteria.append([
                            run + 1, j, complexity, k, w, u,
                            mse_train, mse_test_int, mse_test_ext,
                            sigma_train, float(loglike_train), aic_train, bic_train, mdl_train, pysr_score,
                            to_float(sens_interp_sv), to_float(sens_extrap_sv), to_float(sens_interp_mv), to_float(sens_extrap_mv), to_float(extrap_div)
                        ])

                        if not np.isnan(rmse_train) and rmse_train < min_rmse:
                            min_rmse = rmse_train
                            yhat_gpe = yhat_train_ext_pysr

                        hybrid_scores_sv.append([
                            run + 1, j,
                            to_float(evaluator.calculate_hybrid_score(aic_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(bic_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(mdl_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(pysr_score, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(mse_train, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(mse_test_int, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(evaluator.calculate_hybrid_score(mse_test_ext, extrap_div, sens_interp_sv, sens_extrap_sv)),
                            to_float(sens_interp_sv), to_float(sens_extrap_sv), to_float(extrap_div)
                        ])

                        hybrid_scores_mv.append([
                            run + 1, j,
                            to_float(evaluator.calculate_hybrid_score(aic_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(bic_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(mdl_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(pysr_score, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(mse_train, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(mse_test_int, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(evaluator.calculate_hybrid_score(mse_test_ext, extrap_div, sens_interp_mv, sens_extrap_mv)),
                            to_float(sens_interp_mv), to_float(sens_extrap_mv), to_float(extrap_div)
                        ])

                    # Save yhat_gpe values into CSV file
                    if yhat_gpe is not None:
                        yhat_gpe_df = pd.DataFrame(yhat_gpe, columns=["yhat_gpe"])
                        yhat_gpe_filename = os.path.join(output_dir, f"yhat_gpe_run_{run + 1}.csv")
                        yhat_gpe_df.to_csv(yhat_gpe_filename, index=False)
                        print(f"Saved yhat_gpe values for Run {run + 1} to {yhat_gpe_filename}")

                    # Save PySR Pareto front equations for this run
                    equations_df = gp_model.equations_.copy()
                    equations_df['run'] = run + 1
                    equations_df['equation_index'] = range(len(equations_df))
                    
                    # Add equation strings
                    equation_strings = []
                    for j in range(len(equations_df)):
                        try:
                            equation_str = str(gp_model.sympy(j))
                            equation_strings.append(equation_str)
                        except:
                            equation_strings.append("Error parsing equation")
                    equations_df['equation_string'] = equation_strings
                    
                    # Save individual run file
                    pysr_equations_filename = os.path.join(output_dir, f"pysr_equations_run_{run + 1}.csv")
                    equations_df.to_csv(pysr_equations_filename, index=False)
                    print(f"Saved PySR Pareto front equations for Run {run + 1} to {pysr_equations_filename}")
                    
                    # Append to combined list
                    all_equations_list.append(equations_df)

                end_time = time.time()
                print("ETA: ", end_time - start_time)
                
                # Save results for this run
                save_run_results_noncv(gp_results, model_selection_criteria, hybrid_scores_sv, hybrid_scores_mv,
                                     run + 1, output_dir)
                print(f"Saved results for Run {run + 1}.")
                
                # Save summary of skipped equations for this run
                run_skipped = [eq for eq in skipped_equations if eq['run'] == run + 1]
                if run_skipped:
                    # Flatten nan_counts for CSV
                    skipped_rows = []
                    for eq in run_skipped:
                        row = {
                            'run': eq['run'],
                            'fold': eq['fold'],
                            'equation_index': eq['equation_index'],
                            'equation': eq['equation']
                        }
                        # Add NaN counts per prediction type
                        for pred_type, count in eq['nan_counts'].items():
                            row[f'nan_count_{pred_type}'] = count
                        skipped_rows.append(row)
                    
                    skipped_df = pd.DataFrame(skipped_rows)
                    skipped_file = os.path.join(output_dir, f"skipped_equations_run_{run + 1}.csv")
                    skipped_df.to_csv(skipped_file, index=False)
                    print(f"  Skipped {len(run_skipped)} equations due to NaN predictions (saved to {skipped_file})")

            # Distribution figure (last run)
            plt.figure(figsize=(10, 6))
            plt.scatter(X_train[:, 0], X_train[:, 1], c='red', label="Train Data", alpha=0.7)
            plt.scatter(X_test_interpolation[:, 0], X_test_interpolation[:, 1], c='blue', label="Test Interpolation", alpha=0.7)
            if X_test_extrapolation.shape[0] > 0:
                plt.scatter(X_test_extrapolation[:, 0], X_test_extrapolation[:, 1], c='green', label="Test Extrapolation", alpha=0.7)
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
            plt.title("Distribution Dataset")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"plot_run_last.png"))
            plt.savefig(os.path.join(output_dir, f"plot_run_last.pdf"))
            plt.close()

        # SAVE RESULTS (per mode)
        if use_cv:
            gp_results_df = pd.DataFrame(gp_results, columns=[
                "Run", "Fold", "Equation_Index",
                "RMSE_Train", "RMSE_Val", "RMSE_Test_Interpolation", "RMSE_Test_Extrapolation"
            ])
            gp_results_df.to_csv(os.path.join(output_dir, "gp_model_evaluation_results_cv.csv"), index=False)

            model_selection_df = pd.DataFrame(model_selection_criteria, columns=[
                "Run", "Fold", "Equation_Index", "Complexity", "Constant_Num", "Total_Symbols", "Distinct_Symbols",
                "MSE_Train", "MSE_Val", "MSE_Test_Interpolation", "MSE_Test_Extrapolation",
                "Sigma_Train", "Sigma_Val", "Log_likelihood_Train", "Log_likelihood_Val",
                "AIC_Train", "AIC_Val",
                "BIC_Train", "BIC_Val", "MDL_Train", "MDL_Val",
                "PSM_Train", "PSM_Val",
                "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
            ])
            model_selection_df.to_csv(os.path.join(output_dir, "gp_model_selection_criteria_cv.csv"), index=False)

            hybrid_sv_df = pd.DataFrame(hybrid_scores_sv, columns=[
                "Run", "Fold", "Equation_Index",
                "Hybrid_AIC_SVP", "Hybrid_BIC_SVP", "Hybrid_MDL_SVP", "Hybrid_PSM_SVP", 
                "Hybrid_MSE_Train_SVP", "Hybrid_MSE_Val_SVP", "Hybrid_MSE_Test_Int_SVP", "Hybrid_MSE_Test_Ext_SVP",
                "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Extrapolation_Divergence"
            ])
            hybrid_sv_df.to_csv(os.path.join(output_dir, "hybrid_model_scores_sv_cv.csv"), index=False)
            print("Saved: hybrid_model_scores_sv_cv.csv")

            hybrid_mv_df = pd.DataFrame(hybrid_scores_mv, columns=[
                "Run", "Fold", "Equation_Index",
                "Hybrid_AIC_MVP", "Hybrid_BIC_MVP", "Hybrid_MDL_MVP", "Hybrid_PSM_MVP", 
                "Hybrid_MSE_Train_MVP", "Hybrid_MSE_Val_MVP", "Hybrid_MSE_Test_Int_MVP", "Hybrid_MSE_Test_Ext_MVP",
                "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
            ])
            
            hybrid_mv_df.to_csv(os.path.join(output_dir, "hybrid_model_scores_mv_cv.csv"), index=False)
            print("Saved: hybrid_model_scores_mv_cv.csv")

            # Spearman correlation vs. Test MSE per run (averaged across folds)
            metrics_to_compare = ["AIC_Train", "BIC_Train", "MDL_Train", "MSE_Train", "PSM_Train"]
            corr_rows = []
            for run, group in model_selection_df.groupby("Run"):
                # Aggregate test MSE across all folds for each equation
                equation_tests = group.groupby("Equation_Index")["MSE_Test_Interpolation"].mean()
                equation_metrics = {metric: group.groupby("Equation_Index")[metric].mean() for metric in metrics_to_compare}
                
                row = {"Run": run}
                for metric in metrics_to_compare:
                    try:
                        x = pd.to_numeric(equation_metrics[metric], errors='coerce')
                        y = pd.to_numeric(equation_tests, errors='coerce')
                        valid = ~(x.isna() | y.isna())
                        rho, _ = spearmanr(x[valid], y[valid]) if valid.sum() > 1 else (np.nan, None)
                    except Exception:
                        rho = np.nan
                    row[metric + "_vs_MSE_Test"] = rho
                corr_rows.append(row)
            pd.DataFrame(corr_rows).to_csv(os.path.join(output_dir, "spearman_correlations_vs_test_mse_cv.csv"), index=False)
            print("Saved: spearman_correlations_vs_test_mse_cv.csv")
            
            # Save all equations from all runs in one file
            if all_equations_list:
                all_equations_df = pd.concat(all_equations_list, ignore_index=True)
                combined_equations_filename = os.path.join(output_dir, "pysr_equations_all_runs_cv.csv")
                all_equations_df.to_csv(combined_equations_filename, index=False)
                print(f"Saved: {combined_equations_filename} ({len(all_equations_list)} runs combined)")
            
            print("Saved cross-validated results.")
        else:
            gp_results_df = pd.DataFrame(gp_results, columns=[
                "Run", "Equation_Index",
                "RMSE_Train", "RMSE_Test_Interpolation", "RMSE_Test_Extrapolation"
            ])
            gp_results_df.to_csv(os.path.join(output_dir, "gp_model_evaluation_results.csv"), index=False)

            model_selection_df = pd.DataFrame(model_selection_criteria, columns=[
                "Run", "Equation_Index", "Complexity", "Constant_Num", "Total_Symbols", "Distinct_Symbols",
                "MSE_Train", "MSE_Test_Interpolation", "MSE_Test_Extrapolation",
                "Sigma_Train", "Log_likelihood_Train",
                "AIC_Train", "BIC_Train", "MDL_Train", "PSM_Train",
                "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
            ])
            model_selection_df.to_csv(os.path.join(output_dir, "gp_model_selection_criteria.csv"), index=False)
            print("Saved: gp_model_selection_criteria.csv")

            hybrid_sv_df = pd.DataFrame(hybrid_scores_sv, columns=[
                "Run", "Equation_Index",
                "Hybrid_AIC_SVP", "Hybrid_BIC_SVP", "Hybrid_MDL_SVP", "Hybrid_PSM_SVP", 
                "Hybrid_MSE_Train_SVP", "Hybrid_MSE_Test_Int_SVP", "Hybrid_MSE_Test_Ext_SVP",
                "Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP", "Extrapolation_Divergence"
            ])
            hybrid_sv_df.to_csv(os.path.join(output_dir, "hybrid_model_scores_sv.csv"), index=False)
            print("Saved: hybrid_model_scores_sv.csv")

            hybrid_mv_df = pd.DataFrame(hybrid_scores_mv, columns=[
                "Run", "Equation_Index",
                "Hybrid_AIC_MVP", "Hybrid_BIC_MVP", "Hybrid_MDL_MVP", "Hybrid_PSM_MVP", 
                "Hybrid_MSE_Train_MVP", "Hybrid_MSE_Test_Int_MVP", "Hybrid_MSE_Test_Ext_MVP",
                "Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP", "Extrapolation_Divergence"
            ])
            
            hybrid_mv_df.to_csv(os.path.join(output_dir, "hybrid_model_scores_mv.csv"), index=False)
            print("Saved: hybrid_model_scores_mv.csv")

            # Spearman correlation vs. INTERP MSE per run
            metrics_to_compare = ["AIC_Train", "BIC_Train", "MDL_Train", "MSE_Train", "PSM_Train"]
            correlation_results = []
            for run, group in model_selection_df.groupby("Run"):
                correlations = {"Run": run}
                for metric in metrics_to_compare:
                    try:
                        x = pd.to_numeric(group[metric], errors='coerce')
                        y = pd.to_numeric(group["MSE_Test_Interpolation"], errors='coerce')
                        valid = ~(x.isna() | y.isna())
                        rho, _ = spearmanr(x[valid], y[valid]) if valid.sum() > 1 else (np.nan, None)
                    except Exception:
                        rho = np.nan
                    correlations[metric + "_vs_MSE_Test"] = rho
                correlation_results.append(correlations)
            pd.DataFrame(correlation_results).to_csv(os.path.join(output_dir, "spearman_correlations_vs_test_mse.csv"), index=False)
            print("Saved: spearman_correlations_vs_test_mse.csv")
            
            # Save all equations from all runs in one file
            if all_equations_list:
                all_equations_df = pd.concat(all_equations_list, ignore_index=True)
                combined_equations_filename = os.path.join(output_dir, "pysr_equations_all_runs.csv")
                all_equations_df.to_csv(combined_equations_filename, index=False)
                print(f"Saved: {combined_equations_filename} ({len(all_equations_list)} runs combined)")
