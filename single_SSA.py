import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool

np.random.seed(45)


def ssa_model(nu, a, nu_p, alpha):
    """
    Equation (Callingham et al. 2015):
    S_nu = a * (nu/nu_p)^alpha * (1 - exp(-tau)) / tau
    where tau = (nu/nu_p)^(alpha - 2.5)
    
    Limits:
    - High Freq (nu >> nu_p, tau -> 0): S_nu ~ nu^alpha (optically thin)
    - Low Freq  (nu << nu_p, tau -> inf): S_nu ~ nu^2.5 (optically thick)
    """
    nu = np.array(nu)
    ratio = nu / nu_p
    tau_exponent = alpha - 2.5 #(optically thin regime has alpha < 0, so tau grows at low freq)
    tau = ratio ** tau_exponent
    term1 = ratio ** alpha
    
    # Calculate term2: (1 - e^-tau) / tau
    # Numerical stability handling:
    # If tau is very small (high freq), (1-e^-tau)/tau approaches 1.
    # If tau is very large (low freq), (1-e^-tau)/tau approaches 1/tau.
    term2 = np.empty_like(tau)
    # Mask for small tau to use Taylor expansion or numpy's expm1 for precision
    # Using a threshold of 1e-4 for small tau approximation
    small_tau = tau < 1e-4
    large_tau = ~small_tau
    
    # For small tau: (1 - (1 - tau + ...)) / tau ~= 1
    if np.any(small_tau):
        term2[small_tau] = -np.expm1(-tau[small_tau]) / tau[small_tau]
        
    # For large tau: (1 - exp(-tau)) / tau
    if np.any(large_tau):
        term2[large_tau] = (1.0 - np.exp(-tau[large_tau])) / tau[large_tau]

    S_nu = a * term1 * term2
    return S_nu


#mcmc priors

def log_prior(theta):
    a, nu_p, alpha = theta
    
    # User specified ranges:
    # a: (0, 10000)
    # nu_p: (0.01, 5) GHz
    # alpha: (-3.5, 3.5)
    
    if (0 < a < 10000) and (0.01 < nu_p < 5) and (-2.5 < alpha < 0):
        return 0.0
    return -np.inf

def log_likelihood(theta, nu, flux, flux_err):
    a, nu_p, alpha = theta
    model = ssa_model(nu, a, nu_p, alpha)
    
    if np.any(np.isnan(model)) or np.any(np.isinf(model)):
        return -np.inf

    sigma2 = flux_err ** 2
    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta, nu, flux, flux_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, nu, flux, flux_err)



#fiitting

if __name__ == "__main__":
    filename = 'data.csv' 
    
    try:
        data = pd.read_csv(filename, header=None)
        
        data = data.sort_values(by=0)
        
        nu_obs = data[0].values
        flux_obs = data[1].values
        err_obs = data[2].values
        print(f"Loaded {len(nu_obs)} data points.")
    except Exception as e:
        print(f"Error loading data: {e}")
        

    
    max_idx = np.argmax(flux_obs)
    init_a = flux_obs[max_idx]
    init_nu_p = nu_obs[max_idx]
    init_alpha = -0.7
    
    initial_guess = [init_a, init_nu_p, init_alpha]
    
    ndim = 3
    nwalkers = 50
    nsteps = 5000 
    burn_in = 1000

    print(f"Initial Guess: a={init_a:.2f}, nu_p={init_nu_p:.2f}, alpha={init_alpha:.2f}")

    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    
    pos[:, 0] = np.abs(pos[:, 0]) 
    pos[:, 1] = np.clip(pos[:, 1], 0.02, 4.9)
    pos[:, 2] = np.clip(pos[:, 2], -3.4, 3.4)

    print("Running MCMC...")
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(nu_obs, flux_obs, err_obs), pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)
    
    labels = ["a", "nu_p", "alpha"]
    best_fit_params = []
    
    print("\n--- MCMC Results (Median +/- 1 sigma) ---")
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        best_fit_params.append(mcmc[1])
        print(f"{labels[i]}: {mcmc[1]:.4f} (+{q[1]:.4f} / -{q[0]:.4f})")
    
    a_fit, nu_p_fit, alpha_fit = best_fit_params


    print("\n--- Independent Slope Estimation ---")
    
    mask_thick = nu_obs < nu_p_fit
    mask_thin = nu_obs > nu_p_fit
    
    def fit_power_law(x, y):
        if len(x) < 3: return np.nan, np.nan
        slope, intercept, _, _, std_err = linregress(np.log10(x), np.log10(y))
        return slope, std_err

    alpha_thick, err_thick = fit_power_law(nu_obs[mask_thick], flux_obs[mask_thick])
    if not np.isnan(alpha_thick):
        print(f"Alpha Thick (Data < {nu_p_fit:.2f}): {alpha_thick:.4f} +/- {err_thick:.4f}")
        print("Note: SSA Model predicts exactly +2.5 here.")
    else:
        print("Alpha Thick: Not enough data points.")

    alpha_thin, err_thin = fit_power_law(nu_obs[mask_thin], flux_obs[mask_thin])
    if not np.isnan(alpha_thin):
        print(f"Alpha Thin  (Data > {nu_p_fit:.2f}): {alpha_thin:.4f} +/- {err_thin:.4f}")
        print(f"Note: MCMC fitted alpha ({alpha_fit:.4f}) should match this closely.")
    else:
        print("Alpha Thin: Not enough data points.")


#plotting
    
    fig = corner.corner(
        flat_samples, 
        labels=[r"$a$", r"$\nu_p$", r"$\alpha$"], 
        truths=[a_fit, nu_p_fit, alpha_fit],
        show_titles=True
    )
    plt.savefig("ssa_corner_plot.png")
    print("\nCorner plot saved as 'ssa_corner_plot.png'")
    plt.close()

    plt.figure(figsize=(10, 10))
    
    plt.errorbar(nu_obs, flux_obs, yerr=err_obs, fmt='o', color='black', label='Data', zorder=5)
    
    nu_smooth = np.logspace(np.log10(min(nu_obs)*0.5), np.log10(max(nu_obs)*2.0), 200)
    flux_smooth = ssa_model(nu_smooth, a_fit, nu_p_fit, alpha_fit)
    
    plt.plot(nu_smooth, flux_smooth, color='red', linewidth=2, 
             label=f'SSA\n$\\nu_p$={nu_p_fit:.2f}, $\\alpha$={alpha_fit:.2f}')

    if not np.isnan(alpha_thick):
        slope, intercept, _, _, _ = linregress(np.log10(nu_obs[mask_thick]), np.log10(flux_obs[mask_thick]))
        y_reg = (10**intercept) * (nu_obs[mask_thick]**slope)
        plt.plot(nu_obs[mask_thick], y_reg, '--', color='blue', alpha=0.6, 
                 label=f'alpha_thick: {alpha_thick:.2f}')

    if not np.isnan(alpha_thin):
        slope, intercept, _, _, _ = linregress(np.log10(nu_obs[mask_thin]), np.log10(flux_obs[mask_thin]))
        y_reg = (10**intercept) * (nu_obs[mask_thin]**slope)
        plt.plot(nu_obs[mask_thin], y_reg, '--', color='green', alpha=0.6, 
                 label=f'alpha_thin: {alpha_thin:.2f}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('Flux Density (mJy)', fontsize=14)
    #plt.title(f'SSA Model Fit: $\\alpha$={alpha_fit:.2f}', fontsize=14)
    plt.legend(fontsize=10)
    #plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("ssa_model_fit.png")
    print("Model fit plot saved as 'ssa_model_fit.png'")
    plt.show()
