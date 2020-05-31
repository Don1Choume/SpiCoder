# SpiCoder
Signal-Spike encoder/decoder module for neuroscience.

This code is inspired by following repositories;
* SNN Encoder Tools  
https://github.com/KEDRI-AUT/snn-encoder-tools

* Spike Encoders  
https://github.com/akshaybabloo/Spikes

This code provides encoding/decoding method only for 1d array.
All contributions are welcome.

## Dependency
numpy

## Example
```python
## Batch Encoding and Decoding
from SpiCoder.Batch import TBR as TBRb
from SpiCoder.Batch import BSA as BSAb

## Sequential Encoding and Decoding
from SpiCoder.Sequential import TBR as TBRs
from SpiCoder.Sequential import BSA as BSAs
import pandas as pd
import numpy as np
from scipy import signal as sg


## Load data
data_base = '../data/raw/dataIBM'
data_exts = '.csv'
targ_df = pd.read_csv(data_base+data_exts, header=None).astype('float64')

targ_df.plot()


## Initialize
# TBR
tbrb = TBRb(1.7)
tbrs = TBRs(1.7)

# BSA
fir = sg.firwin(10, 0.01, fs=1)
bsab = BSAb(1, fir)
bsas = BSAs(1, fir)

# initialization is necessary at sequential
bsas.shift = float(np.min(targ_df))
bsas.gain = float(np.max(targ_df) - np.min(targ_df))


## Encode
# batch encode
tbrbe_df = pd.DataFrame(tbrb.encode(targ_df.values.squeeze()))
bsabe_df = pd.DataFrame(bsab.encode(targ_df.values.squeeze()))

# encode result
tbrbe_df.plot()
bsabe_df.plot()

# sequential encode
tbrse_df = pd.DataFrame(targ_df[0].apply(tbrs.encode))
bsase_df = pd.DataFrame(targ_df[0].apply(bsas.encode))

# encode result
tbrse_df.plot()
bsase_df.plot()


## Decode
# batch decode
tbrbd_df = pd.DataFrame(tbrb.decode(tbrbe_df.values.squeeze()))
bsabd_df = pd.DataFrame(bsab.decode(bsabe_df.values.squeeze()))

# decode result
tbrbd_df.plot()
bsabd_df.plot()

# sequential decode
tbrsd_df = pd.DataFrame(tbrse_df[0].apply(tbrs.decode))
bsasd_df = pd.DataFrame(bsase_df[0].apply(bsas.decode))

# decode result
tbrsd_df.plot()
bsasd_df.plot()
```

## Lisence
MIT
