# Task Statement

An EEG file (.mat format) is attached with this mail for performing the following tasks. The file has 19 channel EEG recordings of a participant which is of duration 10 seconds at a sampling frequency 256 Hz (Dimension: 19 X 2560) . The data is already preprocessed (i.e.) band-pass filtered, eye and muscle artefacts removed.

Plot the EEG signals before and after downsampling.

1. Power Spectral Density (PSD) Analysis  
   Compute the Power Spectral Density (PSD) for each EEG channel in the provided preprocessed signal.     
Use appropriate techniques (e.g., Welch's method) to calculate the PSD and plot corresponding figures for each channel as subplots.  
2. Mean Power Calculation and Visualization  
   Calculate the mean power of each channel across the signal.  
   Plot the mean power values for all channels in a clear and labeled graph (line/bar/ of your choice).  
3. Downsampling and Reanalysis  
   Downsample the EEG signal to 128 Hz while preserving signal integrity.  
   Repeat Tasks 1 and 2 on the downsampled signal.  
Hint: Use loadmat from scipy.io to load the data.  For psd and resampling use scipy.signal.

## Notes:
- there are 2560 samples- 256 per second for 10 seconds
- 19 channels- each for an electrode
- the amplitude is in micro volt
- 
# 1. Plotting the EEG Signals Before Downsampling

## Code
```python
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

f = loadmat('/home/zapdos/Downloads/EEGfile.mat')
sampling_rate = 256
samples = 2560
channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
groups = [['FP1', 'FP2'],
          ['F7', 'F3', 'FZ', 'F4', 'F8'],
          ['C3', 'CZ', 'C4'],
          ['T7', 'T8'],
          ['P7', 'P3', 'PZ', 'P4', 'P8'],
          ['O1', 'O2']]
channel_to_index = {ch: i for i, ch in enumerate(channels)}
group_names = ["Frontal Polar", "Frontal", "Central", "Temporal", "Parietal", "Occipital"]

time = np.arange(0, samples) / sampling_rate  

for group, name in zip(groups, group_names):
    n_channels = len(group)
    ncols = 2
    nrows = int(np.ceil(n_channels / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.5 * nrows))
    fig.suptitle(f"{name} Lobe", fontsize=16)
    
    axes = np.array(axes).reshape(-1)


    for i, ch_name in enumerate(group):
        idx = channel_to_index[ch_name]
        data = f["S06V01"][idx].squeeze()
        axes[i].plot(time, data)
        axes[i].set_title(ch_name + f" [{idx}]")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Voltage (mV)")


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/home/zapdos/Obsidian Vault/BCI Task 2/attachments/{name.lower().replace(' ', '_')}_lobe.png")
```

## Frontal Polar
![[frontal_polar_lobe.png]]


## Frontal
![[frontal_lobe.png]]

## Central
![[central_lobe.png]]

## Temporal
![[temporal_lobe.png]]

## Parietal 
![[parietal_lobe.png]]

## Occipital
![[occipital_lobe.png]]

# 2. PSD Analysis

## Code
```python
from scipy.io import loadmat
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np

f = loadmat('/home/zapdos/Downloads/EEGfile.mat')
sampling_rate = 256
samples = 2560
channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
groups = [['FP1', 'FP2'],
          ['F7', 'F3', 'FZ', 'F4', 'F8'],
          ['C3', 'CZ', 'C4'],
          ['T7', 'T8'],
          ['P7', 'P3', 'PZ', 'P4', 'P8'],
          ['O1', 'O2']]
channel_to_index = {ch: i for i, ch in enumerate(channels)}
group_names = ["Frontal Polar", "Frontal", "Central", "Temporal", "Parietal", "Occipital"]

time = np.arange(0, samples) / sampling_rate  

for group, name in zip(groups, group_names):
    n_channels = len(group)
    ncols = 2
    nrows = int(np.ceil(n_channels / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.5 * nrows))
    fig.suptitle(f"{name} Lobe", fontsize=16)
    
    axes = np.array(axes).reshape(-1)

    for i, ch_name in enumerate(group):
        idx = channel_to_index[ch_name]
        data = f["S06V01"][idx].squeeze()
        freqs, psd = welch(data, fs=sampling_rate)
        mask = freqs < 40
        axes[i].plot(freqs[mask], psd[mask])
        axes[i].set_title("PSD: " + ch_name + f" [{idx}]")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Power (mV²/Hz)")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/home/zapdos/Obsidian Vault/BCI Task 2/attachments/psd_{name.lower().replace(' ', '_')}_lobe.png")
```

## Frontal Polar
![[psd_frontal_polar_lobe.png]]

## Frontal
![[psd_frontal_lobe.png]]

## Central
![[psd_central_lobe.png]]

## Temporal
![[psd_temporal_lobe.png]]

## Parietal
![[psd_parietal_lobe.png]]

## Occipital
![[psd_occipital_lobe.png]]

# Mean Voltage per Channel

## Code
```python
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

f = loadmat('/home/zapdos/Downloads/EEGfile.mat')

channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

channel_to_index = {ch: i for i, ch in enumerate(channels)}


mean_voltages = [np.mean(f["S06V01"][channel_to_index[ch]].squeeze()) for ch in channels]
    

bar = plt.bar(channels, mean_voltages, color='skyblue')
plt.bar_label(bar, fmt='%.2f', padding=8)
plt.xlabel("Channel")
plt.ylabel("Mean Voltage (µV)")
plt.title("Mean Voltage per Channel")
plt.xticks(rotation=45)
plt.show()
```

![[mean_voltages.png]]

# Downsampling to 128 Hz

## Frontal Polar
![[downsampled_frontal_polar_lobe.png]]


## Frontal
![[downsampled_frontal_lobe.png]]

## Central
![[downsampled_central_lobe.png]]

## Temporal
![[downsampled_temporal_lobe.png]]

## Parietal 
![[downsampled_parietal_lobe.png]]

## Occipital
![[downsampled_occipital_lobe.png]]

# PSD of Downsampled:

## Code
```python
from scipy.io import loadmat
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

f = loadmat('/home/zapdos/Downloads/EEGfile.mat')
sampling_rate = 256
samples = 2560

new_sampling_rate = 128
new_samples = 1280

channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
groups = [['FP1', 'FP2'],
          ['F7', 'F3', 'FZ', 'F4', 'F8'],
          ['C3', 'CZ', 'C4'],
          ['T7', 'T8'],
          ['P7', 'P3', 'PZ', 'P4', 'P8'],
          ['O1', 'O2']]
channel_to_index = {ch: i for i, ch in enumerate(channels)}
group_names = ["Frontal Polar", "Frontal", "Central", "Temporal", "Parietal", "Occipital"]

time = np.arange(0, samples) / sampling_rate  
new_time = np.arange(0, new_samples) / new_sampling_rate

for group, name in zip(groups, group_names):
    n_channels = len(group)
    ncols = 2
    nrows = int(np.ceil(n_channels / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.5 * nrows))
    fig.suptitle(f"Downsampled PSD: {name} Lobe", fontsize=16)
    
    axes = np.array(axes).reshape(-1)

    for i, ch_name in enumerate(group):
        idx = channel_to_index[ch_name]
        data = f["S06V01"][idx].squeeze()
        new_data = resample(data, new_samples)

        freqs, psd = welch(data, fs=sampling_rate)
        new_freqs, new_psd = welch(new_data, fs=new_sampling_rate)

        axes[i].plot(freqs, psd, label='Original', color='blue')
        axes[i].plot(new_freqs, new_psd, label='Downsampled', color='orange')
        axes[i].set_title(ch_name + f" [{idx}]")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Power (μV²/Hz)")
        axes[i].set_xlim(0, 40)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/home/zapdos/Obsidian Vault/BCI Task 2/attachments/psd_downsampled_{name.lower().replace(' ', '_')}_lobe.png")
```

## Frontal Polar

![[psd_downsampled_frontal_polar_lobe.png]]

  

## Frontal

![[psd_downsampled_frontal_lobe.png]]

  

## Central

![[psd_downsampled_central_lobe.png]]

  

## Temporal

![[psd_downsampled_temporal_lobe.png]]

  

## Parietal
![[psd_downsampled_parietal_lobe.png]]
## Occipital
![[psd_downsampled_occipital_lobe.png]]

# Inaccuracies noted

Due to the same settings being used in Welch function, but change in sampling rate, the window size is different in both cases. This leads to difference in PSD diagram.

By proportionally decreasing the nperseg to 128 (earlier 256), we get the following diagrams:

## Frontal Polar
![[psd_downsampled_tweakedfrontal_polar_lobe.png]]

  

## Frontal
![[psd_downsampled_tweakedfrontal_lobe.png]]

  

## Central
![[psd_downsampled_tweakedcentral_lobe.png]]

  

## Temporal
![[psd_downsampled_tweakedtemporal_lobe.png]]

  

## Parietal
![[psd_downsampled_tweakedparietal_lobe.png]]

## Occipital
![[psd_downsampled_tweakedoccipital_lobe.png]]


# Mean Voltages of Downsampled Data

`scipi.signal.resample` uses a FFT based approach to convert to frequency domain, add or remove zeroes in the middle according to up or downsampling, then transforms back to time domain.

![[line_mean_voltages.png]]

![[line_mean_voltages_downsampled.png]]

![[line_mean_voltages_raw_downsampled.png]]

![[line_mean_voltages_raw_downsampled_10.png]]

![[line_mean_voltages_all.png]]
