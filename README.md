# Multivariate Singular Spectrum Analysis (mSSA)

Multivariate Singular Spectrum (mSSA) is an algorithm for multivariate time series forecasting and imputation. 
Specifically, mSSA allows you to predict entries that are:

1- At a future time step (i.e. forecasting);

2- Missing/corrupted by noise (i.e. imputation)


This is the official implementation of our  [paper](https://arxiv.org/abs/2006.13448). refer to the paper for  more information about the theory and the algorithm of mSSA.  



## Installation
This work has the following dependencies:

- Python 3.5+ with the libraries: (numpy, pandas, scipy, sklearn)
 

To install the mSSA package form the source, simply clone this repository and then install the package using pip as follows:

`pip3 install .`
 

## Getting Started
To get started, first load the time series example we have provided in `../mSSA/examples/testdata/tables/mixturets_var.csv` using pandas.

```python
import pandas as pd
df = pd.read_csv("mSSA/examples/testdata/tables/mixturets_var.csv")
```
Then initialise and fit your  mSSA model on the time series named `ts_7` as follows:
 
```python
from mssa.mssa import mSSA
model = mSSA()
model.update(df.loc[:,['ts_7']]) 
```
Then you can impute or forecast any entry using the predict function. For example:

```python
df = model.predict('ts_7',1000)
```

will impute the 1000th entry, while 
```python
df = model.predict('ts_7', 100001,100100)
```

will forecast the entrie between 100001 to 100100.

## Example
We provide a running example for both synthetic and real world dataset in a python notebook in the mSSA/examples folder. [Here](/mssa/examples/mSSA_notebook_example.ipynb).
## License 
This work is licensed under the Apache 2.0 License. 
