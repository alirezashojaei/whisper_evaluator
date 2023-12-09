import argparse
import os

from whisper import available_models

from model_evaluator.calculate_metrics import Evaluator
from model_evaluator.utils import available_models


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument(
        "--model",
        default="basic",
        choices=available_models(os.path.abspath('model') if os.path.exists('model') else None),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/audio",
        help="Path to directory containing audio files",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/reference.xlsx",
        help=(
            "Path to the excel file containing recognized text"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/evaluation.xlsx",
        help=(
            "Path to the output file if you want output"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print out the evaluation results of each file"
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    evaluator = Evaluator(args.model)

    print("Loading Elapsed time: ", str(evaluator.load_model_elapsed_time), " sec")
    print("Using ", evaluator.device)