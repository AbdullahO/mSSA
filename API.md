# mSSA Documentation 

## mSSA class

```
class mssa.mSSA(rank=None, rank_var=None,segment=False, T=int(2.5e7), T_var=None, gamma=0.2, T0=10, col_to_row_ratio=5,
                 agg_interval=None, agg_method='average', uncertainty_quantification=True, p=None, direct_var=True, L=None,
                 persist_L=None, normalize=True, fill_in_missing=False  )
```

Multivariate Singular Spectrum (mSSA) is an algorithm for multivariate time series forecasting and imputation. Specifically, mSSA allows you to predict entries that are:

1. At a future time step (i.e., forecasting);

2. Missing/corrupted by noise (i.e., imputation)

This repository is the implementation of the [paper: On Multivariate Singular Spectrum Analysis](https://arxiv.org/abs/2006.13448). Refer to the paper for more information about the theory and the algorithm of mSSA.  


 
### parameters

Name | Type| Description
:------------------ | :-------------|:------------- 
`rank`     |int | the number of singular values to retain in the means prediction model. Default is `None`. When None, a rank is automatically chosen based on the method suggested in the paper: ['The Optimal Hard Threshold for Singular Values is 4/sqrt(3)'](https://arxiv.org/abs/1305.5870)
`rank_var`     | int |the number of singular values to retain in the variance prediction model. Default is also `None`.
`segment`  | boolean | If True, several sub-models will be built, each with a moving window of length T.
`T` | int| Number of entries in each submodel in the means prediction model, if `segment` is `True`.
`gamma`| float (0,1)| Determines whether the last sub-model is fully updated (Re-do the matrix decomposition), or is incrementally updated via an incremental method (faster, but less accurate). Specifically, it determines the fraction of new observations to total observation, after which the last sub-model is fully updated. Default `0.5`. 
`T0`| int | Minimum number of observations to fit a model. (Default: 10).
`col_to_row_ratio`| int | the ratio of no. columns to the number of rows in each sub-model's page matrix. (Default:5).
`agg_interval` |float| set the interval for the pre-processing aggregation step in seconds (if the index is timestamps) or units (if the index is integers). Default is None, where the interval will be determined by the median difference between timestamps.
`agg_method`| str | Choose one of {`max`, `min`, `average`}. (Default: `average`).
`uncertainty_quantification`| bol|  if true, estimate the time-varying variance.  (Default: True).
`p` | float (0,1) | select the probability of observaing a value in the time series. In other words, it represents the fraction of observed values. If None, it will be computed from the observations. (Default: None)
`direct_var`| boolean | if True, calculate variance by subtracting the mean from the observations in the variance prediction model (recommended); otherwise, the variance model will be built on the squared observations. (Default: True).
`L`|  int | the number of rows in each sub-model. if set, col_to_row_ratio is ignored. (Default:None)
`normalize`| boolean | Normalize the multiple time series before fitting the model. (Default: True).
`fill_in_missing` | boolean | if true, missing values will be filled by carrying the last observations forward, and then carrying the latest observation backward. If false, they will be filled with zeros. (Default:False)

---

 
### Methods
----
#### `update_model(df)`

This function takes a new set of entries in a dataframe and updates the model accordingly. Note that this function is ude for the initial fit as well as updating the model. If `self.segment` is set to True, the new entries might build several sub-models depending on the parameters `T` and `gamma`.    


Name | Type| Description
:------------ | :-------------|:------------- 
`df` |pandas dataframe |  Dataframe of new entries to be included in the new model. 



----

#### `predict(col_name, t1, t2=None, num_models=10, confidence_interval=True, use_imputed=False, return_variance=False, confidence=95, uq_method='Gaussian')`


##### Arguments
Name | Type| Description
:------------ | :-------------|:------------- 
`col_name` |any valid pandas columns name {str, int} |  name of the column to be predicted. It must one of the columns in the original  DF used for training.
`t1` | (int, or valid timestamp) |  The initial timestamp to be predicted.
`t2` | (int, or valid timestamp) |  The last timestamp to be predicted, (Optional, Default = t1).
`num_models` | (int, or valid timestamp) |  If `self.segment`==`True`, several submodels are built, This argument determeines how many of these sub-models will be used to make a forecast. Specicifcally, the last `num_models`  submodels are used. (Optional, Default = 10)
`confidence_interval` | bool |  If true, return the (confidence%) confidence interval along with the prediction (Optional, Default = True)
`confidence` | float, (0,100) | he confidence used to produce the upper and lower bounds. (Default : 95)
`use_imputed`| bool|  If true, use denoised (imputed) observations to forecast  (Optional, Default = False)
`return_variance`| bool | If true, return mean and variance  (Optional, Default = False, overrides confidence_interval)
`uq_method`| string| Specify which model is used to get the lower and upper bound. Choose from  {"Gaussian" ,"Chebyshev"}. Default "Gaussian".
        

##### Returns
Name | Type| Description
:------------ | :-------------|:------------- 
`Results` | Pandas DataFrame |  The prediction dataframe that contains predictions and the associated timestamps. 
