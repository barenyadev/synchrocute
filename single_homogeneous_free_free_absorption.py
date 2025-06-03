import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

#Data: frequency (MHz), flux (mJy), uncertainty (mJy)
data = """571.000        36.0200        3.60000            
650.000        35.5800        3.50000      
691.000        33.0000        3.30000           
700.000        37.5700        3.70000           
1180.00        48.9000        4.80000
1283.00        40.0000        4.00000 
1340.00        34.0000        3.40000  
3000.00        34.6000        3.40000    
5000.00        26.0000        2.60000   
5000.00        27.0000        2.70000"""

lines = data.strip().split('\n')
freq, flux, flux_err = [], [], []

for line in lines:
    values = line.split()
    freq.append(float(values[0]))
    flux.append(float(values[1]))
    flux_err.append(float(values[2]))

freq = np.array(freq)
flux = np.array(flux)
flux_err = np.array(flux_err)

#hom. ffa
def ffa_model(nu, a, alpha, nu_turnover):
    """
    Homogeneous Free-Free Absorption model
    S_nu = a * (nu^-alpha) * exp(-tau)
    where tau = (nu/nu_turnover)^-2.1
    
    Parameters:
    nu: frequency
    a: scaling factor
    alpha: spectral index
    nu_turnover: turnover frequency
    """
    tau = (nu / nu_turnover)**(-2.1)
    return a * (nu**(-alpha)) * np.exp(-tau)

#guess_initial
# a: scaling factor (order of magnitude of flux)
# alpha: typical spectral index for synchrotron emission (~0.7)
# nu_turnover: somewhere in the middle of frequency range
initial_guess = [1000, 0.5, 1000]

try:
    popt, pcov = curve_fit(ffa_model, freq, flux, p0=initial_guess, 
                          sigma=flux_err, absolute_sigma=True,
                          bounds=([0, -2, 100], [1e6, 3, 10000]))
    
    a_fit, alpha_fit, nu_turnover_fit = popt

    param_errors = np.sqrt(np.diag(pcov))
    a_err, alpha_err, nu_turnover_err = param_errors
    
    model_flux = ffa_model(freq, *popt)
    
    chi_squared = np.sum(((flux - model_flux) / flux_err)**2)
    degrees_of_freedom = len(freq) - len(popt)  # n_data - n_parameters
    reduced_chi_squared = chi_squared / degrees_of_freedom
    
    freq_model = np.logspace(np.log10(min(freq)*0.45), np.log10(max(freq)*2), 1000) #change this to set the range of the model fit (S_nu)
    flux_model = ffa_model(freq_model, *popt)
    
    # Calculate the actual turnover frequency (where S_nu peaks)
    # For FFA model, the peak occurs where d(S_nu)/d(nu) = 0
    # This gives nu_peak = nu_turnover * (2.1/alpha)^(1/2.1)
    nu_peak = nu_turnover_fit * (2.1/alpha_fit)**(1/2.1)
    flux_peak = ffa_model(nu_peak, *popt)
    

    plt.figure(figsize=(40, 40))

    plt.errorbar(freq, flux, yerr=flux_err, fmt='o', color='black', 
                capsize=10, capthick=5, markersize=50, linewidth=5,
                label='Data')
    
    plt.plot(freq_model, flux_model, 'b-', linewidth=5, 
            label='Best fit FFA model', color='black')
    
    #plt.axvline(x=nu_peak, color='red', linestyle='--', linewidth=2,
               #label=f'Peak frequency: {nu_peak:.1f} MHz')
    #plt.plot(nu_peak, flux_peak, 'go', markersize=10, markeredgecolor='red',
            #markeredgewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(300, 10000)  # Set x-axis range
    plt.ylim(10, 100)      # Set y-axis range
    plt.xlabel('Frequency [MHz]', fontsize=70)
    plt.ylabel('Flux Density [mJy]', fontsize=70)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.tick_params(axis='both', which='minor', labelsize=40)
    #plt.title('IC 2402 Core (Homogeneous FFA)', fontsize=30)
    
    #textstr = f'''Homogeneous FFA
#α_inj = {alpha_fit:.3f} ± {alpha_err:.3f}
#ν_turnover = {nu_peak:.1f} MHz
#Reduced χ² = {reduced_chi_squared:.3f}'''
    
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    #plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=25,
            #verticalalignment='top', bbox=props)
    
    #plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=12)
    plt.tight_layout()
    
    print("=" * 50)
    print("FREE-FREE ABSORPTION MODEL FIT RESULTS")
    print("=" * 50)
    print(f"Scaling factor (a): {a_fit:.2f} ± {a_err:.2f}")
    print(f"Spectral index (α): {alpha_fit:.3f} ± {alpha_err:.3f}")
    print(f"Turnover frequency (ν_turnover): {nu_turnover_fit:.1f} ± {nu_turnover_err:.1f} MHz")
    print(f"Peak frequency: {nu_peak:.1f} MHz")
    print(f"Peak flux density: {flux_peak:.1f} mJy")
    print(f"Chi-squared: {chi_squared:.2f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"Reduced chi-squared: {reduced_chi_squared:.3f}")
    print("=" * 50)
    plt.savefig('ffa.png', dpi=100)
    plt.show()
    
except Exception as e:
    print(f"Error in fitting: {e}")
    print("Try adjusting initial parameter guesses or bounds.")
