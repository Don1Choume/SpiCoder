from .CoderBase import CoderBase
import numpy as np

class TBR(CoderBase):
    def __init__(self, f_factor):
        self.f_factor = f_factor
        self.threshold = None
        self.start_point = None
        self.previous_signal = None
        self.enc_N = 0
        self.M = 0
        self.V = 0
        self.dec_isFirst = True

    def encode(self, signal, start_point=None, threshold=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            self.start_point = signal
        if self.previous_signal is None:
            self.previous_signal = signal
        
        diff = signal - self.previous_signal
        self.previous_signal = signal

        self.enc_N += 1
        N = self.enc_N
        self.V = self.V*(N-1)/N + ((self.M-diff)**2)*(N-1)/(N**2)
        self.M = self.M*(N-1)/N + diff/N

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self.M + self.f_factor*np.sqrt(self.V)

        spikes = 0
        if diff > self.threshold:
            spikes = 1
        elif diff < -self.threshold:
            spikes = -1
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

        if self.dec_isFirst:
            self.dec_isFirst = False
            signal = self.start_point
        else:
            signal = self.previous_signal + np.sign(spikes)*self.threshold
        self.previous_signal = signal
        return signal


class SF(CoderBase):
    def __init__(self, threshold):
        self.threshold = threshold
        self.start_point = None
        self.base = None
        self.dec_isFirst = True

    def encode(self, signal, start_point=None, base=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            self.start_point = signal
        if base is not None:
            self.base = base
        elif self.base is None:
            self.base = self.start_point
        spikes = 0
        if signal > self.base + self.threshold:
            spikes = 1
        elif signal < self.base - self.threshold:
            spikes = -1
        self.base += spikes*self.threshold
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

        if self.dec_isFirst:
            self.dec_isFirst = False
            signal = self.start_point
        else:
            signal = self.previous_signal + np.sign(spikes)*self.threshold
        self.previous_signal = signal
        return signal


class MW(CoderBase):
    def __init__(self, threshold, window):
        self.threshold = threshold
        self.window = window
        self.signals = np.zeros(self.window)
        self.start_point = None
        self.enc_N = 0
        self.dec_isFirst = True

    def encode(self, signal, start_point=None):
        if start_point is not None:
            self.start_point = start_point
        elif self.start_point is None:
            self.start_point = signal

        if self.enc_N==0:
            base = signal
        else:
            if self.enc_N//self.window == 0:
                base = np.sum(self.signals)/self.enc_N
            else:
                base = np.mean(self.signals)
        self.signals[self.enc_N%self.window] = signal
        self.enc_N += 1
        
        spikes = 0
        if signal > base + self.threshold:
            spikes = 1
        elif signal < base - self.threshold:
            spikes = -1
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

        if self.dec_isFirst:
            self.dec_isFirst = False
            signal = self.start_point
        else:
            signal = self.previous_signal + np.sign(spikes)*self.threshold
        self.previous_signal = signal
        return signal


class BSA(CoderBase):
    def __init__(self, threshold, fir):
        self.threshold = threshold
        self.fir = fir
        self.shift = None
        self.gain = None
        self.sig_hist = np.zeros(len(self.fir))
        self.spk_hist = np.zeros(len(self.fir))
        self.err1 = 0
        self.err2 = 0
        self.enc_N = 0

    def encode(self, signal, shift=None, gain=None):
        if shift is not None:
            self.shift = shift
        elif self.shift is None:
            raise ValueError('shift is not set')
        if gain is not None:
            self.gain = gain
        elif self.gain is None:
            raise ValueError('gain is not set')

        signal = (signal - self.shift)/self.gain
        if signal < 0:
            signal = 0
        self.enc_N += 1

        spikes = 0
        self.sig_hist = np.roll(self.sig_hist, 1)
        self.sig_hist[0] = signal
        idx = min(len(self.fir), self.enc_N)
        self.err1 = np.sum(np.abs(self.sig_hist - self.fir)[:idx])
        self.err2 = np.sum(np.abs(self.sig_hist)[:idx])
        if self.err1 <= (self.err2 - self.threshold):
            spikes = 1
            self.sig_hist -= self.fir
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

        self.spk_hist = np.roll(self.spk_hist, 1)
        self.spk_hist[0] = spikes
        signal = np.sum(self.spk_hist*self.fir)*self.gain + self.shift
        return signal