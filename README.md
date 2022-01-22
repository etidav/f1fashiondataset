# HERMES: Hybird Error-Corrector Model with inclusion of External Signal for non stationnary time series.

Authors: Etienne DAVID, Sylvain LE CORFF and Jean BELLOT

Paper link: ....

### Abstract
> Developing models and algorithms to draw causal inference for time series is a long standing statistical problem. It is crucial for many applications, in particular for fashion or retail industries, to make optimal inventory decisions and avoid massive wastes. By tracking thousands of fashion trends on social media with state-of-the-art computer vision approaches, we propose a new model for fashion time series forecasting. Our contribution is  twofold. We first provide publicly the first fashion dataset gathering 10000 weekly fashion time series. As influence dynamics are the key of emerging trend detection, we associate with each time series an external weak signal representing behaviors of influencers. Secondly, to leverage such a complex and rich dataset, we propose a new hybrid forecasting model. Our approach combines per-time-series parametric models with seasonal components and a global recurrent neural network to include sporadic external signals. This hybrid model provides state-of-the-art results on the proposed fashion dataset, on the weekly time series of the M4 competition, and illustrates the benefit of the contribution of external weak signals.

## Code Organisation

This repository provides the F1 fashion dataset studied in the HERMES paper and a simple code base to reproduce the final results. For now only two benchmark methods are implemented: `snaive` and  `ets`.  

F1 fashion time series dataset is available at the link bellow :    
http://files.heuritech.com/raw_files/f1_fashion_dataset.tar.xz  

It is divided in two ```.csv``` files:
 - the 10000 normalized and anonymized fashion time series : ```f1_main.csv```
 - the 10000 normalized and anonymized external weak signals time series : ```f1_fashion_forward.csv```

## Reproduce benchmark results

First, you should install this package:
```bash
pip install . # if you only want to install the package to run the benchmark
pip install -e '.[dev]' # if you want to install the package in editable mode with dev dependencies to modify the code
```

To reproduce the HERMES benchmarks results, a simple code is provided to forecast the F1 dataset using 2 different benchmark methods (`snaive`, `ets`).

A python script is provided:

- benchmarck.py: make sure you are in a python environment with the requirements provided is the setup.py file.
```bash
python benchmark.py --help # display the default parameters and their description
python benchmark.py # run the benchmark on an example with snaive and ets
python benchmark --dataset-path DATASET_PATH --model-names snaive ... # run the benchmark on DATASET_PATH with only snaive
python benchmark --dataset-path DATASET_PATH --model-names ets ... # run the benchmark on DATASET_PATH with only ets
python benchmark --dataset-path DATASET_PATH --model-names snaive --model-names ets ... # run the benchmark on DATASET_PATH with ets and snaive
```

For the ets model, it is recommended to use multiprocessing. A parameter named processes is provided and allow you to set the number of CPU that you want to use (default value = 1).

## HERMES paper Benchmark results

The following tab summarizes the results that can be reproduced with this code:

| Model         | Mase        | Accuracy    |
| :-------------| :-----------| :-----------|
| snaive        | 0.881       | 0.357       |
| ets           | 0.807       | 0.449       |
