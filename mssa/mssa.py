"""
mSSA implementation for the mSSA variant in the paper:
    Anish Agarwal, Abdullah Alomar, and Devavrat Shah. "On Multivariate Singular Spectrum Analysis."
"""
# Author: Abdullah Alomar <aalomar AT mit DOT edu>
# License: Apache License 2.0


from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from mssa.src.prediction_models.ts_meta_model import TSMM
from mssa.src.tsUtils import index_ts_mapper, index_ts_inv_mapper
from mssa.src.tsUtils import unnormalize


class mSSA(object):
    '''
    :param rank: (int) the number of singular values to retain in the means prediction model
    :param rank_var: (int) the number of singular values to retain in the variance prediction model
    :param gamma: (float) (0,1) fraction of T after which the last sub-model is fully updated
    :param segment: (bool) If True, several sub-models will be built, each with a moving windows of length T.
    :param T: (int) Number of entries in each submodel in the means prediction model
    :param T0: (int) Minimum number of observations to fit a model. (Default 100)
    :param col_to_row_ratio: (int) the ratio of no. columns to the number of rows in each sub-model
    :param agg_interval: (float) set the interval for pre-processing aggregation step in seconds (if index is timestamps)
     or units (if index is integers). Default is None, where the interval will be determined by the median difference
     between timestamps.
    :param agg_method: Choose one of {'max', 'min', 'average'}.
    :param uncertainty_quantification: (bool) if true, estimate the time-varying variance.
    :param p: (float) (0,1) select the probability of observation. If None, it will be computed from the observations.
    :param direct_var: (bool) if True, calculate variance by subtracting the mean from the observations in the variance
    prediction model (recommended), otherwise, the variance model will be built on the squared observations
    :param L:  (int) the number of rows in each sub-model. if set, col_to_row_ratio is ignored.
    :param normalize: (bool) Normalize the multiple time series before fitting the model.
    :param fill_in_missing: if true, missing values will be filled by carrying the last observations forward, and then
     carrying the latest observation backward.
    '''

    def __init__(self, rank=None, rank_var=1, T=int(2.5e7), T_var=None, gamma=0.2, T0=10, col_to_row_ratio=5,
                 agg_method='average', uncertainty_quantification=True, p=None, direct_var=True, L=None,
                 persist_L=None, normalize=True, fill_in_missing=False, segment=False, agg_interval=None):

        # Checks
        if gamma < 0 or gamma >= 1:
            raise ValueError('gamma must be in the range (0,1]')
        if rank is not None:
            if (rank < 1):
                raise ValueError('rank must be > 0')
        if rank_var is not None:
            if (rank_var < 1):
                raise ValueError('rank_var must be > 0')

        if col_to_row_ratio < 1:
            raise ValueError('col_to_row_ratio must be > 0')

        if T < T0:
            raise ValueError('T must be greater than T0')
        if L is not None:
            if T < L:
                raise ValueError('T must be greater than L')

        self.k = rank
        self.p = p
        self.SSVT = False
        self.current_timestamp = None

        ############ Temp ############
        # In current implemntation, we will assume that T_var = T
        T_var = T
        ##############################

        if T_var is None:
            T_var = T
        else:
            T_var = T_var

        self.persist_L = persist_L
        if self.persist_L is None:
            self.persist_L = False
            if L is not None:
                self.persist_L = True
        self.L = L
        self.gamma = gamma
        self.T0 = T0
        self.k_var = rank_var

        self.agg_method = agg_method
        self.normalize = normalize
        self.agg_interval = agg_interval
        self.col_to_row_ratio = col_to_row_ratio
        self.direct_var = direct_var
        self.fill_in_missing = fill_in_missing
        self.segment = segment
        

        ############ Edit #############
        # Probably this number is large enough :), but might want to do it properly
        ###############################
        if not self.segment:
            self.T = int(1e150)
            self.T_var = int(1e150)

        else:
            self.T = T
            self.T_var = T_var

        self.uq = uncertainty_quantification
        if self.k_var == 0:
            self.uq = False

        self.fitted = False
        self.column_names = None

    def _get_time_difference(self, time_stamps):
        """
        get the median time difference  in seconds between consecutive observations
        :param time_stamps: time column of the time series dataframe from which we calculate the median difference  
        """
        if isinstance(time_stamps[0], (int, np.integer)):
            return np.median(np.diff(time_stamps.values))
        else:
            return float(np.median(np.diff(time_stamps.values))) / (1e9)
        return interval

    def _aggregate_df(self, df):
        """
        aggregate the dataframe by to its time columns according to agg_interval and agg_method
        :param df: dataframe to be aggregated 
        """

        # filter previously recorded timestamp
        if isinstance(df.index[0], datetime):
            agg_op = df.groupby(pd.Grouper(freq='%sS' % self.agg_interval, sort=True, closed='right', label='right'))
        elif isinstance(df.index[0], (int, np.integer)):
            agg_op = df.groupby((df.index // self.agg_interval).astype(int))
        else:
            raise ValueError('Dataframe index must be integers or timestamps')

        if self.agg_method == 'average':
            df_agg = agg_op.mean()
        elif self.agg_method == 'min':
            df_agg = agg_op.min()
        elif self.agg_method == 'max':
            df_agg = agg_op.max()
        else:
            raise ValueError("Aggregation method %s is not recognized" % self.agg_method)
        if self.fitted:
            return df_agg.loc[df_agg.index > self.current_timestamp]
        else:
            self.start_time = df_agg.index[0]
            return df_agg

    def update_model(self, df):
        """
        This function takes a new set of entries in a dataframe and update the model accordingly.
        if self.segment is set to True, the new entries might build several sub-models  depending on the parameters T and gamma.
        :param df: dataframe of new entries to be included in the new model
        """
        # check that there is a time columns
        if not (df.index.is_unique):
            raise ValueError("The input dataframe must have an index with unique values")
        if not (isinstance(df.index[0], (int, np.integer))) and not (isinstance(df.index[0], datetime)):
            raise ValueError("Dataframe index must be integers or timestamps")

            # If this is the first time the model is being fit, determine important parameters.
        if not self.fitted:
            self.no_ts = len(df.columns)
            self.column_names = list(df.columns)
            self.ts_model = TSMM(self.k, self.T, self.gamma, self.T0, col_to_row_ratio=self.col_to_row_ratio,
                                 SSVT=self.SSVT, p=self.p, L=self.L, persist_L=self.persist_L,
                                 no_ts=self.no_ts, normalize=self.normalize, fill_in_missing=self.fill_in_missing)
            self.var_model = TSMM(self.k_var, self.T_var, self.gamma, self.T0, col_to_row_ratio=self.col_to_row_ratio,
                                  SSVT=self.SSVT, p=self.p,
                                  L=self.L, persist_L=self.persist_L, no_ts=self.no_ts, normalize=self.normalize,
                                  fill_in_missing=self.fill_in_missing)

            if self.agg_interval is None:  self.agg_interval = self._get_time_difference(df.index)

            if self.segment:
                self.T = self.ts_model.T
                self.T_var = self.var_model.T
        else:
            if self.column_names != list(df.columns):
                raise ValueError('The Dataframe must have the same columns as the original Dataframe')

        df = self._aggregate_df(df)

        obs = (df.values).astype('float')
        # if obs.shape[0] < self.no_ts:
        #     print("Dataframe does not have enough new unseen entries. (Number of new timestamps should be =>  "
        #           "(number of time series)")
            # return

        # lag is the the slack between the variance and timeseries model        
        lag = None
        if self.ts_model.TimeSeries is not None:
            lag = (self.ts_model.TimeSeriesIndex // self.no_ts - self.var_model.TimeSeriesIndex // self.no_ts)
            if lag > 0:
                lagged_obs = self.ts_model.TimeSeries[-lag:, :]
            else:
                lag = None

        # Update mean model
        self.ts_model.update_model(obs)
        self.k = self.ts_model.kSingularValuesToKeep
        # Determine updated models

        models = {k: self.ts_model.models[k] for k in self.ts_model.models if self.ts_model.models[k].updated}

        if self.uq:
            if self.direct_var:

                means = self.ts_model._denoiseTS(models)[
                        self.var_model.TimeSeriesIndex // self.no_ts:self.ts_model.MUpdateIndex // self.no_ts, :]
                if lag is not None:
                    var_obs = np.concatenate([lagged_obs, obs])
                else:
                    var_obs = obs
                var_entries = np.square(var_obs[:len(means), :] - means)
                self.var_model.update_model(var_entries)
            else:
                var_entries = np.square(df.values)
                self.var_model.update_model(var_entries)

        self.current_timestamp = df.index[-1]
        self.fitted = True

    def predict(self, col_name, t1, t2=None, num_models=10, confidence_interval=True, use_imputed=False,
                return_variance=False, confidence=95, uq_method='Gaussian'):
        '''
        Predict 'col_name' at time t1 to time t2.
        :param col_name: name of the column to be predicted. It must one of the columns in the original  DF used for
        training.
        :param t1: (int, or valid timestamp) The initial timestamp to be predicted
        :param t2: (int, or valid timestamp) The last timestamp to be predicted, (Optional, Default = t1)
        :param num_models: (int, >0) Number of submodels used to create a forecast (Optional, Default = 10)
        :param confidence_interval: (bool)  If true, return the (confidence%) confidence interval along with the prediction (Optional, Default = True)
        :param confidence: (float, (0,100)) The confidence used to produce the upper and lower bounds
        :param use_imputed: (bool)  If true, use imputed values to forecast  (Optional, Default = False)
        :param return_variance: (bool)  If true, return mean and variance  (Optional, Default = False, overrides confidence_interval)
        :param uq_method: (string): Choose from  {"Gaussian" ,"Chebyshev"}
        :return:
            DataFrame with the timestamp and predictions
        '''
        # convert t1 and t2 to model index 
        if t2 is None:
            t2 = t1

        if type(t1) != type(t2):
            raise ValueError("Start and end time should have the same type ")

        if (not isinstance(t1, (str, pd.Timestamp)) and isinstance(self.start_time, (pd.Timestamp))) or (
                isinstance(self.start_time, (int, np.integer)) and not isinstance(t1, (int, np.integer))):
            if isinstance(self.start_time, (pd.Timestamp)):
                raise ValueError("The time value should be a valid timestamp ")
            else:
                raise ValueError("The time value should be an integer ")

        if not isinstance(t1, (int, np.integer)):
            t1 = pd.to_datetime(t1)
            t2 = pd.to_datetime(t2)

        t1 = index_ts_mapper(self.start_time, self.agg_interval, t1)
        t2 = index_ts_mapper(self.start_time, self.agg_interval, t2)
        t1 = max(t1, 0)
        t2 = max(t2, 0)
        # check that column is in dataframe
        if not (col_name in self.column_names):
            raise ValueError('Column %s does not exist in the time series dataframe. Choose from %s ' % (
                col_name, self.column_names))

        ts_no = self.column_names.index(col_name)

        # get current update index
        update_index = self.ts_model.MUpdateIndex // self.no_ts
        update_index_var = self.var_model.MUpdateIndex // self.no_ts

        #### Estimate Mean and Variance ########

        # Decide whether its an imputation task, a forecasting task, or a combination thereof
        if t2 < update_index:
            # mean is imputed
            predictions = self._get_imputation_range_local(t1, t2, self.ts_model, ts_no)
            # variance 
            if self.uq and t2 < update_index_var:
                # impute variance
                var = self._get_imputation_range_local(t1, t2, self.var_model, ts_no)
            elif self.uq:
                # impute and forecast variance
                var1 = self._get_imputation_range_local(t1, update_index_var, self.var_model, ts_no)
                var2 = self._get_forecast_range_local(update_index_var, t2, self.var_model, ts_no,
                                                      use_imputed=use_imputed)
                var = np.concatenate([var1, var2])
            else: var = 0
        
        elif t1 > update_index:
            # all variance and mean should be forecasted
            predictions = self._get_forecast_range_local(t1, t2, self.ts_model, ts_no, num_models)
            if self.uq: var = self._get_forecast_range_local(t1, t2, self.var_model, ts_no, num_models, use_imputed=use_imputed)
        
        else:
            # Both mean and variance will be forecasted and imputed
            predictions1 = self._get_imputation_range_local(t1, update_index - 1, self.ts_model, ts_no)
            predictions2 = self._get_forecast_range_local(update_index, t2, self.ts_model, ts_no, num_models)
            predictions = np.concatenate([predictions1, predictions2])
            if self.uq: 
                var_index = max(t1, update_index_var )
                var1 = self._get_imputation_range_local(t1, var_index - 1, self.var_model, ts_no)
                var2 = self._get_forecast_range_local(var_index, t2, self.var_model, ts_no, num_models,
                                                  use_imputed=use_imputed)
                var = np.concatenate([var1, var2])


        if self.uq and not self.direct_var:
            var = var - np.square(predictions)
        df = pd.DataFrame(
            index=index_ts_inv_mapper(self.start_time, self.agg_interval, np.arange(t1, t2 + 1).astype('float')))
        df['Mean Predictions'] = predictions
        
        if not self.uq: 
            return df
        
        var = np.maximum(0, var)
        
        if return_variance:
            df['Variance'] = var
            return df

        #### Confidence Interval ########
        if confidence_interval:
            lb, ub = self._get_prediction_bounds(predictions, var, c=confidence, uq_method=uq_method)
            df['Lower Bound'] = lb
            df['Upper Bound'] = ub

            return df

        return df

    def _get_prediction_bounds(self, predictions, var, c=95, uq_method='Gaussian'):
        '''

        :param predictions:
        :param var:
        :param c:
        :param uq_method:
        :return:
        '''
        if c < 0 or c >= 100:
            raise Exception('confidence interval c must be in the range (0,100): 0 <=c< 100')

        if uq_method == 'Chebyshev':
            alpha = 1. / (np.sqrt(1 - c / 100))
        elif uq_method == 'Gaussian':
            alpha = norm.ppf(1 / 2 + c / 200)
        else:
            raise Exception('uq_method option is not recognized,  available options are: "Gaussian" or "Chebyshev"')

        ci = alpha * np.sqrt(var)

        return predictions - ci, predictions + ci

    def _get_forecast_range_local(self, t1, t2, model, ts_no, num_models=10, use_imputed=False):
        '''

        :param t1:
        :param t2:
        :param model:
        :param ts_no:
        :param num_models:
        :param use_imputed:
        :return:
        '''
        no_ts = self.no_ts
        coeffs = model.models[0].weights
        no_coeff = len(coeffs)
        coeffs = np.zeros(coeffs.shape)
        bias = 0
        for i in range(num_models):
            if len(model.models) - 1 - i < 0:
                num_models = i
                break
            coeffs_model = model.models[len(model.models) - 1 - i].weights
            coeffs[-len(coeffs_model):] += coeffs_model
            bias += (-model.models[len(model.models) - 1 - i].weights.sum() + 1) * \
                    model.models[len(model.models) - 1 - i].norm_mean[ts_no]
            # coeffs_model = np.mean(np.array([m.weights for m in list(model.models.values())[-num_models-1:-1]]), 0)
        bias = bias / num_models
        coeffs = coeffs / (num_models)
        if use_imputed:
            t1_ = min(t1, model.MUpdateIndex // no_ts)
            t1_diff = t1 - t1_
            t1_diff *= t1_diff > 0
            output = np.zeros([t2 - t1_ + 1 + no_coeff])
            output[:no_coeff] = self._get_imputation_range_local(t1_ - no_coeff, t1_ - 1, model, ts_no)
        else:
            start_index_stored_ts = model.TimeSeriesIndex // no_ts - model.T // no_ts
            start_index_stored_ts = max(0, start_index_stored_ts)
            t1_ = min(t1, model.TimeSeriesIndex // no_ts)
            t1_diff = t1 - t1_
            t1_diff *= t1_diff > 0
            output = np.zeros([t2 - t1_ + 1 + no_coeff])
            last_observation_index = t1_ - start_index_stored_ts
            output[:no_coeff] = model.TimeSeries[:, ts_no][last_observation_index - no_coeff:last_observation_index]
            if self.fill_in_missing:
                output[:no_coeff]  =  pd.DataFrame(output[:no_coeff]).fillna(method='ffill').values[:,0]
                output[:no_coeff] = pd.DataFrame(output[:no_coeff]).fillna(method='bfill').values[:,0]
            else:
                output[:no_coeff] = pd.DataFrame(output[:no_coeff]).fillna(value=0).values[:,0]/model.p
        for i in range(0, t2 + 1 - t1_):
            output[i + no_coeff] = np.dot(coeffs.T, output[i:i + no_coeff]) + bias
            # output[i + no_coeff] = sum([a[0]*b for a, b in zip(coeffs,output[i:i + no_coeff])])
        return output[-(t2 - t1 + 1):]

    def _get_imputation_range_local(self, t1, t2, model, ts_no):
        '''

        :param t1:
        :param t2:
        :param model:
        :param ts_no:
        :return:
        '''
        no_ts = self.no_ts
        T_ts = self.T // no_ts
        m1 = int(max((t1) / int(T_ts / 2) - 1, 0))
        m2 = int(max((t2) / int(T_ts / 2) - 1, 0))

        # Get the sub-models parameters
        N1, start1, M1 = model.models[m1].N, model.models[m1].start, model.models[m1].M
        N2, start2, M2 = model.models[m2].N, model.models[m2].start, model.models[m2].M
        last_model = len(model.models) - 1

        # Get normalization constants
        norm_mean = [(model.models[m].norm_mean) for m in range(m1, m2 + 1 + (m2 < last_model))]
        norm_std = [(model.models[m].norm_std) for m in range(m1, m2 + 1 + (m2 < last_model))]

        # calculate tsrow and tscolumn

        tscol2 = int((t2 - model.models[m1].start / no_ts) / N1)
        tsrow2 = int((t2 - model.models[m1].start / no_ts) % N1)
        tscol1 = int((t1 - model.models[m1].start / no_ts) / N1)
        tsrow1 = int((t1 - model.models[m1].start / no_ts) % N1)
        i_index = (t1 - t1 % N1) + tsrow1
        last_model = len(model.models) - 1
        if m1 == m2:
            U1 = model.models[m1].Uk[:, :]
            S1 = model.models[m1].sk[:]
            V1 = model.models[m1].Vk[ts_no::no_ts][tscol1:tscol2 + 1, :]
            p1 = np.dot(U1 * S1[:], V1[:].T)
            if (m1 < last_model - 1 and m1 != 0):
                S2 = model.models[m1 + 1].sk[:]
                V2 = model.models[m1 + 1].Vk[ts_no::no_ts][
                     tscol1 - int(M1 / (2 * no_ts)):tscol2 - int(M1 / (2 * no_ts)) + 1, :]
                U2 = model.models[m1 + 1].Uk[:, :]
                Result = 0.5 * unnormalize(p1.T.flatten()/model.p, norm_mean[0][ts_no], norm_std[0][ts_no])+ 0.5 * unnormalize(
                    np.dot(U2 * S2[:], V2[:].T).T.flatten()/model.p, norm_mean[1][ts_no], norm_std[1][ts_no])
            else:
                Result = unnormalize(p1.T.flatten()/model.p, norm_mean[0][ts_no], norm_std[0][ts_no])

            end = -N2 + tsrow2 + 1
            if end == 0: end = None
            return Result[tsrow1:end]


        else:

            Result = np.zeros([t2 - t1 + 1])

            for m in range(m1, m2 + 1 + (m2 < last_model)):
                N = model.models[m].N
                M = model.models[m].M
                start = 0
                end = M // no_ts * N - 1

                if m == m1:
                    start = t1 - model.models[m].start // no_ts
                elif m == m1 + 1:
                    start = t1 - model.models[m].start // no_ts
                    start *= (start > 0)
                if m == m2:
                    end = t2 - model.models[m].start // no_ts
                elif m == m2 + 1:
                    end = t2 - model.models[m].start // no_ts

                tscol_i = int(start / N)
                tscol_f = int(end / N)
                tsrow_i = int(start % N)
                tsrow_f = int(end % N)
                tsrow_f = -N + tsrow_f + 1
                if tsrow_f == 0: tsrow_f = None
                i = -i_index + model.models[m].start // no_ts + tscol_i * N + tsrow_i
                length = N * (tscol_f - tscol_i + 1) + int(tsrow_f or 0) - tsrow_i
                U = model.models[m].Uk[:]
                S = model.models[m].sk[:]
                V = model.models[m].Vk[ts_no::no_ts][tscol_i:tscol_f + 1, :]
                p = np.dot(U * S, V.T)
                Result[i:i + length] += 0.5 * unnormalize(p.T.flatten()[tsrow_i:tsrow_f]/model.p, norm_mean[m - m1][ts_no],
                                                          norm_std[m - m1][ts_no])
            fix_0_index = int(model.T / (2 * no_ts)) - i_index
            fix_0_index *= (fix_0_index > 0)
            fix_last_index = t2 - model.models[last_model].start // no_ts - int(model.T / (2 * no_ts)) + 1
            fix_last_index *= (fix_last_index > 0)
            Result[:fix_0_index] = 2 * Result[:fix_0_index]

            if fix_last_index > 0: Result[-fix_last_index:] = 2 * Result[-fix_last_index:]

            return Result[:]
