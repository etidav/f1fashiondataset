import pandas as pd
from typing import Dict

from f1fashiondataset.predict import predict
from f1fashiondataset.metrics import compute_mase, compute_accuracy


def eval(
    data: pd.DataFrame, prediction: pd.DataFrame, freq: int = 52, threshold: int = 0.05
) -> Dict[str, int]:
    """
    This method is the main method to compute the MASE and the Accuracy on a dataset.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
    
    - *final_prediction*: A pd.DataFrame with the model predictions.
        In column the time series names.
        In index the time steps.
    
    - *y_histo*: a matrix with the past 52 historical data for each time series.  
        
    - *freq*: By default set to 52. With a value set to 52, the seasonal MASE will be computed.
        If you change this value to 1, the simple MASE will be computed.
    
    - *threshold*: Threshold that defines the yoy classification rule.
        yoy <= -0.5 -> decreasing time series
        -0.5 <=yoy <= 0.5 -> flat time series
        yoy <= -0.5 -> increasing time series
  
    Returns:
    
    - *paper_result*: a dict with the two final evaluation metric given in the HERMES paper: MASE and Accuracy
    """
    time_split = prediction.index[0]
    ground_truth = data.loc[time_split:].iloc[:52]
    histo_ground_truth = data.loc[:time_split].iloc[:-1]

    final_mase = compute_mase(
        ground_truth.values, prediction.values, histo_ground_truth.values, freq=freq
    )
    final_accuracy = compute_accuracy(
        ground_truth.values,
        prediction.values,
        histo_ground_truth.iloc[-52:].values,
        threshold=threshold,
    )
    
    paper_result = {"MASE": final_mase, "Accuracy": final_accuracy}

    return paper_result

def compute_benchmarck_metric(
    data: pd.DataFrame,
    model_name: list,
    time_split: str = None,
    freq: int = 52,
    threshold: int = 0.05,
    processes: int = 1,
) -> pd.DataFrame:
    """
    This method is the main method to reproduce the benchmarck results of the HERMES paper.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
        
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Three possibilities : 'snaive' for the naive forecats,
        'ets' for exponential smoothing and 'tbats' for the tbats model.

    - *time_split*: a str with the following format 'YYYY-MM-DD'. It delimits where stop each time series 
        and start computing a 1 year forecast.
        
    - *freq*: By default set to 52. With a value set to 52, the seasonal MASE will be computed.
        If you change this value to 1, the simple MASE will be computed.
    
    - *threshold*: Threshold that defines the yoy classification rule.
        yoy <= -0.5 -> decreasing time series
        -0.5 <=yoy <= 0.5 -> flat time series
        yoy <= -0.5 -> increasing time series
        
    - *processes*: for methods that need to be train, define how many cpu processes do you want to use to fit/compute the statistical forecasts.
        By default, only 1 cpu will be use.
  
    Returns:
    
    - *paper_result*: a pd.DataFrame with the two final evaluation metric given in the HERMES paper: 
        MASE and Accuracy
    """
    if time_split is None:
        time_split = data.index[-53]

    final_eval = {}
    for i in model_name:
        model_pred = predict(data, i, time_split=time_split, processes=processes)
        model_eval = eval(data, model_pred)
        final_eval[i] = model_eval
    
    paper_result = pd.DataFrame(final_eval).T[["MASE", "Accuracy"]]

    return paper_result