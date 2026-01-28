import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.fft import rfft, rfftfreq, irfft
from statsmodels.tsa.stattools import adfuller
import os

# Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.autolayout': True,
    'grid.alpha': 0.3
})

def load_data(path):
    if not os.path.exists(path):
        print("Data not found, generating random data for testing.")
        return np.random.rand(17856, 170, 3) * 100
    return np.load(path)['data']

# Analysis 1: Figure 3 (Frequency Spectrum)
def plot_frequency_spectrum(data):
    node_idx = 100 
    signal = data[:, node_idx, 0]
    signal_centered = signal - np.mean(signal)
    
    N = len(signal)
    yf = rfft(signal_centered)
    xf = rfftfreq(N, d=5/60)
    
    magnitude = np.abs(yf)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_mask = xf > 0
    periods = 1 / xf[valid_mask]
    mags = magnitude[valid_mask]
    plot_mask = (periods >= 2) & (periods <= 50)
    
    ax.plot(periods[plot_mask], mags[plot_mask], color='#2c3e50', linewidth=2)
    
    for peak_hour in [24, 12, 8]:
        ax.axvline(peak_hour, color='#e74c3c', linestyle='--', alpha=0.6)
        ax.text(peak_hour, ax.get_ylim()[1]*0.9, f'{peak_hour}h', 
                color='#e74c3c', ha='center', fontweight='bold')
        
    ax.set_title(f'Figure 3: Frequency Spectrum (Node {node_idx})', fontweight='bold')
    ax.set_xlabel('Period (Hours)')
    ax.set_ylabel('Amplitude')
    ax.set_xscale('log')
    ax.set_xticks([6, 8, 12, 24, 48])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('02_fft_spectrum.png', dpi=300)
    plt.show()

# Analysis 2: Figure 4 (Trend vs Noise Decomposition)
def plot_trend_noise_decomposition(data):
    node_idx = 100
    signal = data[:, node_idx, 0] 
    
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, d=5/60)
    
    cutoff_hours = 4.0
    cutoff_freq = 1 / cutoff_hours
    
    yf_clean = yf.copy()
    yf_clean[xf > cutoff_freq] = 0
    
    trend_signal = irfft(yf_clean)
    noise_signal = signal - trend_signal
    
    days_to_plot = 10
    steps_to_plot = 288 * days_to_plot
    time_axis = np.arange(steps_to_plot) / 288
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(time_axis, signal[:steps_to_plot], color='#bdc3c7', alpha=0.5, label='Raw Signal (Noisy)')
    ax1.plot(time_axis, trend_signal[:steps_to_plot], color='#e67e22', linewidth=2.5, label='Extracted Trend')
    ax1.set_title(f'Figure 4a: Spectral Decomposition (Node {node_idx})', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Traffic Flow')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(time_axis, noise_signal[:steps_to_plot], color='#2980b9', linewidth=1, alpha=0.9, label='Residual Noise')
    ax2.set_title('Figure 4b: Residual High-Frequency Noise', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Time (Days)', fontweight='bold')
    ax2.set_ylabel('Residual', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.fill_between(time_axis, 0, noise_signal[:steps_to_plot], color='#2980b9', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('02_trend_decomposition.png', dpi=300)
    plt.show()

# Analysis 3: Figure 5 (Stationarity & Concept Drift)
def plot_stationarity_analysis(data):
    node_idx = 100
    signal = data[:, node_idx, 0]
    
    window_size = 144 
    series = pd.Series(signal)
    rolmean = series.rolling(window=window_size).mean()
    rolstd = series.rolling(window=window_size).std()
    
    steps_per_day = 288
    idx_3am = np.arange(36, len(signal), steps_per_day)
    idx_5pm = np.arange(204, len(signal), steps_per_day)
    
    data_3am = signal[idx_3am]
    data_5pm = signal[idx_5pm]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :])
    view_steps = steps_per_day * 14
    time_axis = np.arange(view_steps) / steps_per_day
    
    ax1.plot(time_axis, signal[:view_steps], color='#bdc3c7', alpha=0.3, label='Raw Signal')
    ax1.plot(time_axis, rolmean[:view_steps], color='#e74c3c', linewidth=2, label='Rolling Mean (12h)')
    ax1.plot(time_axis, rolstd[:view_steps], color='#2c3e50', linewidth=2, linestyle='--', label='Rolling Std (12h)')
    
    ax1.set_title(f'Figure 5a: Non-Stationarity Proof', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Flow Statistics')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = fig.add_subplot(gs[1, :])
    
    sns.kdeplot(data_3am, ax=ax2, fill=True, color='#2980b9', alpha=0.6, linewidth=0, label='3:00 AM')
    sns.kdeplot(data_5pm, ax=ax2, fill=True, color='#e67e22', alpha=0.6, linewidth=0, label='5:00 PM')
    
    mean_3am = np.mean(data_3am)
    mean_5pm = np.mean(data_5pm)
    
    ax2.axvline(mean_3am, color='#2980b9', linestyle='--', linewidth=2)
    ax2.axvline(mean_5pm, color='#e67e22', linestyle='--', linewidth=2)
    
    ax2.set_title(f'Figure 5b: Distribution Shift', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Traffic Flow')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    
    y_pos = ax2.get_ylim()[1] * 0.85 
    ax2.text(mean_3am + 5, y_pos, 'Low Mean\nLow Variance', 
             color='#2980b9', ha='left', va='center', fontweight='bold')
    ax2.text(mean_5pm - 5, y_pos, 'High Mean\nHigh Variance', 
             color='#e67e22', ha='right', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('02_stationarity_shift.png', dpi=300)
    plt.show()

# Analysis 4: Figure 6 (Spatial Heterogeneity)
def calculate_noise_energy(signal, sample_rate_hours=5/60, cutoff_hours=4.0):
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, d=sample_rate_hours)
    cutoff_freq = 1 / cutoff_hours
    yf_noise = yf.copy()
    yf_noise[xf <= cutoff_freq] = 0 
    return np.std(irfft(yf_noise))

def plot_noise_heterogeneity_and_handling(data):
    num_nodes = data.shape[1]
    noise_energies = []
    
    for i in range(num_nodes):
        signal = data[:, i, 0] - np.mean(data[:, i, 0])
        noise_energies.append(calculate_noise_energy(signal))
    noise_energies = np.array(noise_energies)
    
    noisiest_node = np.argmax(noise_energies)
    signal_noisy = data[:, noisiest_node, 0]
    
    N = len(signal_noisy)
    yf = rfft(signal_noisy)
    xf = rfftfreq(N, d=5/60)
    cutoff_freq = 1 / 4.0
    
    yf_trend = yf.copy(); yf_trend[xf > cutoff_freq] = 0
    yf_noise = yf.copy(); yf_noise[xf <= cutoff_freq] = 0
    
    trend_part = irfft(yf_trend)
    noise_part = irfft(yf_noise)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    norm = plt.Normalize(noise_energies.min(), noise_energies.max())
    colors = cm.coolwarm(norm(noise_energies))
    
    bars = ax1.bar(np.arange(num_nodes), noise_energies, color=colors, width=0.8, alpha=0.9)
    
    threshold = np.mean(noise_energies) + 2 * np.std(noise_energies)
    ax1.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label='+2 Std Threshold')
    
    ax1.text(noisiest_node, noise_energies[noisiest_node] + 1, f'Node {noisiest_node}\n(Max Noise)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')

    ax1.set_title('Figure 6a: Spatial Heterogeneity of Noise Energy', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Sensor Node Index')
    ax1.set_ylabel('High-Freq Noise Energy')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    ax2 = fig.add_subplot(gs[1])
    view_steps = 288 * 3
    t = np.arange(view_steps) / 288
    
    ax2.plot(t, signal_noisy[:view_steps], color='#95a5a6', alpha=0.4, linewidth=1.5, label='Raw Signal')
    ax2.plot(t, trend_part[:view_steps], color='#2c3e50', linewidth=2.5, label='Trend')
    ax2.fill_between(t, trend_part[:view_steps], signal_noisy[:view_steps], 
                     color='#e74c3c', alpha=0.3, label='Filtered Noise')
    
    ax2.set_title(f'Figure 6b: Noise Handling (Node {noisiest_node})', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Time (Days)')
    ax2.set_ylabel('Flow')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('02_noise_heterogeneity.png', dpi=300)
    plt.show()

# Execution
if __name__ == "__main__":
    NPZ_PATH = r'data/PEMS08.npz'
    data = load_data(NPZ_PATH)
    
    print("Generating Figure 3 (FFT)...")
    plot_frequency_spectrum(data)
    
    print("Generating Figure 4 (Decomposition)...")
    plot_trend_noise_decomposition(data)
    
    print("Generating Figure 5 (Stationarity)...")
    plot_stationarity_analysis(data)
    
    print("Generating Figure 6 (Spatial Heterogeneity)...")
    plot_noise_heterogeneity_and_handling(data)
