from .CoderBase import CoderBase
import numpy as np

class TBR(CoderBase):
    def __init__(self, f_factor):
        self.f_factor = f_factor
        self.threshold = None
        self.start_point = None

    def encode(self, signal, start_point=None, threshold=None):
        if start_point is not None:
            self.start_point = start_point
        else:
            self.start_point = signal[0]
        
        diff = np.pad(np.diff(signal), (1,0))
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = np.mean(diff) + self.f_factor*np.std(diff)

        spikes = np.zeros(signal.shape)
        spikes[diff >  self.threshold] =  1
        spikes[diff < -self.threshold] = -1
        return spikes

    def decode(self, spikes, start_point=None, threshold=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            raise ValueError('start_point is not set')
        if threshold is not None:
            self.threshold = threshold
        elif self.threshold is None:
            raise ValueError('threshold is not set')

        return self.start_point + np.cumsum(spikes)*self.threshold


class SF(CoderBase):
    def __init__(self, threshold):
        self.threshold = threshold
        self.start_point = None
        self.base = 0

    def encode(self, signal, start_point=None, base=None):
        if start_point is not None:
            self.start_point = start_point
        else:
            self.start_point = signal[0]
        if base is not None:
            self.base = base
        else:
            self.base = self.start_point

        spikes = np.zeros(signal.shape)
        for i in range(len(signal)-1):
            if signal[i+1] > self.base + self.threshold:
                spikes[i+1] = 1
            elif signal[i+1] < self.base - self.threshold:
                spikes[i+1] = -1
            self.base += spikes[i+1]*self.threshold
        return spikes

    def decode(self, spikes, start_point=None, threshold=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            raise ValueError('start_point is not set')
        if threshold is not None:
            self.threshold = threshold
        elif self.threshold is None:
            raise ValueError('threshold is not set')

        return self.start_point + np.cumsum(spikes)*self.threshold


class MW(CoderBase):
    def __init__(self, threshold, window):
        self.threshold = threshold
        self.window = window
        self.start_point = None
        self.bases = None

    def encode(self, signal, start_point=None):
        if start_point is not None:
            self.start_point = start_point
        else:
            self.start_point = signal[0]

        spikes = np.zeros(signal.shape)
        mean_conv = lambda x: \
                np.convolve(x, np.ones(self.window)/self.window)[:len(x)]
        bases = mean_conv(signal)/mean_conv(np.ones(signal.shape))
        bases = np.concatenate([bases[np.newaxis, 0], bases[:-1]])
        self.bases = bases
        spikes[signal - bases >  self.threshold] = 1
        spikes[signal - bases < -self.threshold] = -1
        return spikes

    def decode(self, spikes, start_point=None, threshold=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            raise ValueError('start_point is not set')
        if threshold is not None:
            self.threshold = threshold
        elif self.threshold is None:
            raise ValueError('threshold is not set')

        return self.start_point + np.cumsum(spikes)*self.threshold


class BSA(CoderBase):
    def __init__(self, threshold, fir):
        self.threshold = threshold
        self.fir = fir
        self.shift = None
        self.gain = None

    def encode(self, signal, shift=None, gain=None):
        if shift is not None:
            self.shift = shift
        elif self.shift is None:
            self.shift = np.min(signal)
        if gain is not None:
            self.gain = gain
        elif self.gain is None:
            self.gain = np.max(signal) - np.min(signal)

        signal = (signal - self.shift)/self.gain
        spikes = np.zeros(signal.shape)
        for t in range(len(signal)):
            err1 = np.sum(np.abs(signal[max(0, t+1-len(self.fir)):t+1]-self.fir[0:min(t+1, len(self.fir))]))
            err2 = np.sum(np.abs(signal[max(0, t+1-len(self.fir)):t+1]))
            if err1 <= (err2 - self.threshold):
                spikes[t] = 1
                signal[max(0, t+1-len(self.fir)):t+1] -= self.fir[0:min(t+1, len(self.fir))]
        return spikes

    def decode(self, spikes, shift=None, gain=None):
        if shift is not None:
            self.shift = shift
        elif self.shift is None:
            raise ValueError('shift is not set')
        if gain is not None:
            self.gain = gain
        elif self.gain is None:
            raise ValueError('gain is not set')

        fir_conv = lambda x: np.convolve(x, self.fir)[:len(x)]
        signal = fir_conv(spikes)*self.gain + self.shift
        return signal