import multiprocessing
from functools import partial
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from tqdm import tqdm

from f1fashiondataset.constants import WEEKS_IN_A_YEAR


def fit_predict_single_model(ts: Tuple[str, pd.Series], model_name: str) -> Dict[str, np.array]:
    """
    This method is a method to fit and compute the prediction of a statistical method on a single time series.
    Two statistical methods are enabled with this function, the exponential smoothing (ets) and tbats model (tbats)
    Ets forecast is computed using the 'statsmodels' package.
    Tbats forecast is computed using the 'tbats' package.
    
    Arguments:
    
    - *ts*: A tuple provided by the pd.DataFrame.iteritems() iterator.
        The first element is the name of the time series and the seconde one is its values.
        
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Only two possibilities : 'ets' for exponential smoothing `tbats` for the tbats model.
        
    Returns:
    
    - *stat_model*: A dict associating the time series name to its statistical forecast. 
    """
    ts_name = ts[0]
    stat_model = {ts_name: {}}
    if model_name == "ets":
        model = ExponentialSmoothing(ts[1].values, seasonal_periods=WEEKS_IN_A_YEAR, seasonal="add")
        fitted_model = model.fit()
    else:
        raise NotImplementedError(model_name)
    stat_model[ts_name] = fitted_model.forecast(WEEKS_IN_A_YEAR)

    return stat_model


def fit_predict(data: pd.DataFrame, model_name: str, processes: int = 1) -> Dict[str, np.array]:
    """
    This method is the method to compute a forecast for each time series present in a pd.DataFrame for
    methods that need to be fit.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
        
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Only two possibilities : 'ets' for exponential smoothing 'tbats' for the tbats model.
        
    - *processes*: How many cpu process do you want to use to fit/compute the statistical forecasts.
        By default, only 1 cpu will be use.
        
    Returns:
    
    - *model_prediction*: A dict linking the time series names to their associated statistical forecasts. 
    """

    model_prediction = {}
    with multiprocessing.Pool(processes=processes) as pool:
        all_single_pred = list(
            tqdm(
                pool.imap(
                    partial(fit_predict_single_model, model_name=model_name),
                    data.iteritems(),
                    chunksize=5,
                )
            )
        )
    for single_pred in all_single_pred:
        model_prediction.update(single_pred)

    return model_prediction


def compute_snaive_prediction(data: pd.DataFrame) -> Dict[str, np.array]:
    """
    This method is the method to compute the `naive` forecast for each time series present in a pd.DataFrame.
    As the `snaive` forecast is a model that only replicated the past year of data, no train is needed.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
  
    Returns:
    
    - *model_prediction*: A dict linking the time series names to their associated statistical forecasts. 
    """

    model_prediction = {}
    for ts_name in data:
        model_prediction[ts_name] = data[ts_name].values[-WEEKS_IN_A_YEAR:]

    return model_prediction


def predict(
    data: pd.DataFrame,
    model_name: str,
    time_split: str = None,
    processes: int = 1
) -> pd.DataFrame:
    """
    This method is the main method to compute the forecast for each time series present in a pd.DataFrame.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
    
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Three possibilities : 'snaive' for the naive forecats,
        'ets' for exponential smoothing and 'tbats' for the tbats model.
    
    - *time_split*: a str with the followinf format 'YYYY-MM-DD'. It delimits where stop each time series 
        and start computing a 1 year forecast.
        
    - *processes*: for methods that need to be train, define how many cpu processes do you want
        to use to fit/compute the statistical forecasts.
        By default, only 1 cpu will be use.
  
    Returns:
    
    - *final_prediction*: A pd.DataFrame with the model predictions.
        In column the time series names.
        In index the time steps.
    """
    if time_split is not None:
        data = data.loc[:time_split]

    if model_name == "ets":
        prediction = fit_predict(data, model_name, processes=processes)
    elif model_name == "snaive":
        prediction = compute_snaive_prediction(data)
    else:
        raise NotImplementedError(model_name)

    delta = pd.to_datetime(data.index[-1]) - pd.to_datetime(data.index[-2])
    prediction_index = [
        str((pd.to_datetime(data.index[-1]) + delta * (i + 1)).date())
        for i in range(WEEKS_IN_A_YEAR)
    ]
    final_prediction = pd.DataFrame(prediction)
    final_prediction.index = prediction_index

    return final_prediction
