import numpy as np
import matplotlib.pyplot as plt


n_samples = 10000
n_ensembles = 1000
rms_inputs = [1, 2, 5]

plt.figure(figsize=(15, 10))

for i, rms in enumerate(rms_inputs):
  
    voltage = np.random.normal(0, rms, n_samples)
    power = voltage**2
    
 
    v_var = np.var(voltage)
    p_mean = np.mean(power)
    
  
    plt.subplot(2, 3, i+1)
    plt.hist(power, bins=50, color='skyblue', alpha=0.7, density=True)
    plt.axvline(p_mean, color='red', linestyle='--', label=f'Mean Power: {p_mean:.2f}')
    plt.axvline(v_var, color='black', linestyle=':', label=f'Voltage Var: {v_var:.2f}')
    plt.title(f'Power Dist (RMS={rms})')
    plt.legend()


t = np.linspace(0, 1, n_samples)
sine_wave = 5 * np.sin(2 * np.pi * 2 * t) 
voltage_with_sine = np.random.normal(0, 2, n_samples) + sine_wave
power_with_sine = voltage_with_sine**2

plt.subplot(2, 3, 4)
plt.hist(power_with_sine, bins=50, color='salmon', alpha=0.7)
plt.title('Power Hist with Sine Wave')


variances = [np.var(np.random.normal(0, 2, n_samples)) for _ in range(n_ensembles)]

plt.subplot(2, 3, 5)
plt.hist(variances, bins=30, color='lightgreen', alpha=0.7)
plt.title('Distribution of Ensemble Variances')

plt.tight_layout()
plt.show()

sample_sizes = np.logspace(2, 6, 20, dtype=int)  
errors = []

for N in sample_sizes:
    v = np.random.normal(0, 2, N) 
    v_var = np.var(v)
    p_mean = np.mean(v**2)
    errors.append(abs(p_mean - v_var))

plt.subplot(2, 3, 6)
plt.loglog(sample_sizes, errors, marker='o', color='purple')
plt.xlabel('Number of Samples (N)')
plt.ylabel('|Mean Power - Voltage Var|')
plt.title('Convergence of Mean Power to Variance')
plt.grid(True, which="both", ls="-")