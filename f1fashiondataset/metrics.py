import numpy as np
from sklearn.metrics import accuracy_score

from f1fashiondataset.constants import WEEKS_IN_A_YEAR, THRESHOLD


def compute_mase(
    y_true: np.array, y_pred: np.array, y_histo: np.array, freq: int = WEEKS_IN_A_YEAR
) -> int:
    """
    This method is the method to compute the seasonal Mean Absolute Scaled Errror (MASE).
    
    Arguments:
    
    - *y_true*: a matrix with all the ground truth for each time series 
    
    - *y_pred*: a matrix with all the sequence predictions.
    
    - *y_histo*: a matrix with the all the past historical data for each time series.
        
    - *freq*: By default set to 52. If you change this value to 1, the simple MASE will be computed.
  
    Returns:
    
    - *final_mase*: a float representing the final mase. 
        The average mase is computed on all the seqences.
    """
    denominator = np.mean(np.abs(y_histo[freq:] - y_histo[:-freq]), axis=0)
    numerator = np.mean(np.abs(y_true - y_pred), axis=0)
    final_mase = (numerator / denominator).mean()
    return final_mase


def compute_accuracy(
    y_true: np.array, y_pred: np.array, y_histo: np.array, threshold: int = THRESHOLD
) -> float:
    """
    This method is the method to compute the Accuracy based on a year-on-year classification.
    
    Arguments:
    
    - *y_true*: a matrix with all the ground truth for each time series 
    
    - *y_pred*: a matrix with all the sequence predictions.
    
    - *y_histo*: a matrix with the past 52 historical data for each time series.
        
    - *threshold*: Threshold that defines the yoy classification rule.
        yoy <= -0.5 -> decreasing time series
        -0.5 <=yoy <= 0.5 -> flat time series
        yoy <= -0.5 -> increasing time series
  
    Returns:
    
    - *final_accuracy*: a float representing the final accuracy. 
    """
    yoy_true = (np.mean(y_true, axis=0) - np.mean(y_histo, axis=0)) / np.mean(y_histo, axis=0)
    yoy_pred = (np.mean(y_pred, axis=0) - np.mean(y_histo, axis=0)) / np.mean(y_histo, axis=0)

    true_label = 1 * (yoy_true > threshold) - 1 * (yoy_true < -threshold)
    pred_label = 1 * (yoy_pred > threshold) - 1 * (yoy_pred < -threshold)
    final_accuracy = accuracy_score(true_label, pred_label)
    
    return final_accuracy
