#!/usr/bin/python3
import pandas as pd
from argparse import ArgumentParser
from typing import Dict, Any
import pathlib
import os

from f1fashiondataset.eval import compute_benchmarck_metric


def options() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="path to the dataset")
    parser.add_argument('--model_name', type=str,  nargs="+", required=True, help="name of the models whose results you want to reproduce --> 2 methods available ['snaive', 'ets'")
    parser.add_argument('--output', type=str, default= None, help="Output directory if you want to save the results")
    parser.add_argument('--time_split', type=str, default=None, help="a str with the following format 'YYYY-MM-DD'. It delimits where stop each time series and start computing a 1 year forecast.")
    parser.add_argument('--processes', type=int, default=1, help="for methods that need to be train, define how many cpu processes do you want to use")
    return parser.parse_args()


def save_csv_file(data: pd.DataFrame, main_dir: str, file_name: str) -> None:
    pathlib.Path(main_dir).mkdir(parents=True, exist_ok=True)
    data.to_csv(os.path.join(main_dir, file_name))

def main() -> None:
    args = vars(options())
    data = pd.read_csv(args['data'], index_col=0)
    hermes_metric = compute_benchmarck_metric(data=data, model_name=args['model_name'], time_split=args['time_split'], processes = args['processes'])
    print(hermes_metric)
    output_dir_path = args['output']
    if output_dir_path is not None:
        hermes_metric_file_name = ('_').join(model_name) + 'benchmarck_results.csv'
        save_csv_file(hermes_metric,output_dir_path,hermes_metric_file_name)

if __name__ == '__main__':
    main()