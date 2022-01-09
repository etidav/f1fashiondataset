# HERMES: Hybird Error-Corrector Model with inclusion of External Signal for non stationnary time series.

Authors: Etienne DAVID, Sylvain LE CORFF and Jean BELLOT

Paper link: ....

### Abstract
> Developing models and algorithms to draw causal inference for time series is a long standing statistical problem. It is crucial for many applications, in particular for fashion or retail industries, to make optimal inventory decisions and avoid massive wastes. By tracking thousands of fashion trends on social media with state-of-the-art computer vision approaches, we propose a new model for fashion time series forecasting. Our contribution is  twofold. We first provide publicly the first fashion dataset gathering 10000 weekly fashion time series. As influence dynamics are the key of emerging trend detection, we associate with each time series an external weak signal representing behaviors of influencers. Secondly, to leverage such a complex and rich dataset, we propose a new hybrid forecasting model. Our approach combines per-time-series parametric models with seasonal components and a global recurrent neural network to include sporadic external signals. This hybrid model provides state-of-the-art results on the proposed fashion dataset, on the weekly time series of the M4 competition, and illustrates the benefit of the contribution of external weak signals.

## Code Organisation

This repository provide the F1 fashion dataset studied in the HERMES paper and a simple code base to reproduce the final results. For now only two benchmark methods are implemented -> ['snaive', 'ets'].

Fashion dataset is given in the ```data``` directory and divided in two ```.csv``` file :
 - f1_main.csv : the 10000 normalized and anonymized fashion time series.
 - f1_fashion_forward.csv : the 10000 normalized and anonymized external weak signals time series.

```bash
https://github.com/etidav/f1fashiondataset/tree/master/data/f1_main.csv
https://github.com/etidav/f1fashiondataset/tree/master/data/f1_fashion_forward.csv
```

## Reproduce benchmark results

To reproduce the HERMES benchmarks results, a simple code is provided to forecast the F1 dataset using 2 different benchmark methods ('snaive', 'ets').

A python script and a noteboook is provided :

 - Compute_HERMES_benchmarks_results.py : for the python script, make sure you are in a python environnement with the requirements provided is the setup.py file.
         ```bash
         python main.py --data {your_directory_name}/f1_main.csv --model_name snaive 
         ```
 - Compute_HERMES_benchmarks_results.ipynb : A notebook file that provides an example with the 'snaive' model.

For the ets model, it is recommanded to use multiprocessing. A parameter named processes is provided in each scipt (.py or notebook) and allow you to set the number of CPU that you want to use (default value = 1).

## HERMES paper Benchmark results

The following tab summaryzes the results that can be reproduced with this code:

| Model         | Mase        | Accuracy    |
| :-------------| :-----------| :-----------|
| snaive        | 0.880931    | 0.3571      |
| ets           | 0.807054    | 0.4488      |
