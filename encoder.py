import numpy as np

import torch, torch.nn as nn
import snntorch as snn

def TCE(x, threshold):

    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    spike_stream = np.zeros_like(x)
    last_vals = x[0].copy()
    for i in range(1, len(x)):
        diff = x[i] - last_vals
        for j in range(x.shape[1]):
            if diff[j] >= threshold:
                spike_stream[i, j] = 1
                last_vals[j] = x[i, j]
    return spike_stream


def PC(x, num_neurons=8, time_window=10):
    x = np.clip(x, -1, 1)
    x_pos = (x + 1) / 2  # shift to [0,1]
    q = int(round(x_pos * (2**num_neurons - 1)))
    bits = np.array([int(b) for b in np.binary_repr(q, width=num_neurons)])
    
    # spike trains: shape = (num_neurons, time_window)
    spikes = np.zeros((num_neurons, time_window))
    for i, bit in enumerate(bits):
        if bit == 1:
            spikes[i, np.random.randint(0, time_window)] = 1  # one spike at random time
    
    return spikes, bits