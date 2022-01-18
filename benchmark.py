import pathlib
from typing import List

import pandas as pd
import typer

from f1fashiondataset.eval import compute_benchmarck_metric


def main(
    dataset_path: pathlib.Path = typer.Option("data/f1_main.csv", help="Path to the dataset"),
    model_names: List[str] = typer.Option(
        ["ets", "snaive"],
        help=
        "Name of the models whose results you want to reproduce --> 2 methods available ['snaive', 'ets']"
    ),
    time_split: str = typer.Option(
        None,
        help=
        "A date with the following format 'YYYY-MM-DD'. It delimits where stop each time series and start computing a 1 year forecast."
    ),
    processes: int = typer.Option(
        1,
        help="For methods that need to be trained, defines how many cpu processes you want to use."
    ),
    output_folder: pathlib.Path = typer.Option(
        None, help="Output directory if you want to save the results."
    )
):
    data = pd.read_csv(dataset_path, index_col=0)
    hermes_metric = compute_benchmarck_metric(
        data=data, model_names=model_names, time_split=time_split, processes=processes
    )
    print(hermes_metric)
    if output_folder is not None:
        hermes_metric_file_name = '_'.join(model_names) + '_benchmark_results.csv'
        output_folder.mkdir(parents=True, exist_ok=True)
        hermes_metric.to_csv(output_folder.joinpath(hermes_metric_file_name))


if __name__ == '__main__':
    typer.run(main)
