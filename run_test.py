import argparse
import os

from pandas import read_json, Series

from model_evaluator.calculate_metric import Evaluator


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument(
        "--model",
        default="basic",
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/audio",
        help="Path to directory containing audio files",
    )
    parser.add_argument(
        "--reference-path",
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
        "--transcription-method",
        type=str,
        default="openai-whisper",
        choices=["openai-whisper", "faster-whisper"],
        help=(
            "Choose your desired transcript method and which packages you want to use"
        ),
    )
    parser.add_argument(
        "--load-from-check", action="store_true", help="If you want to continue evaluating from a previous checkpoint, set this option."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print out the evaluation results of each file"
    )
    return parser

def save_result(result):
    args = get_parser().parse_args()

    df = read_json(result["json_dataframe"], orient='columns')

    df["average_wer"] = Series(dtype='float64')
    num_rows = len(df)
    last_row_index = num_rows - 1
    df.at[last_row_index, 'average_wer'] = result['Weighted Average WER']

    df.to_excel(os.path.abspath(args.output_dir), engine="openpyxl", index=False)

def main():
    args = get_parser().parse_args()

    evaluator = Evaluator(args.model, 
                          args.transcription_method, 
                          verbose=args.verbose,
                          load_from_checkpoint=args.load_from_check)

    print("Loading Elapsed time: ", str(evaluator.load_model_elapsed_time), " sec")
    print("Using ", evaluator.device)

    wer_result = evaluator.run_evaluation(dataset_path=os.path.abspath(args.reference_path),
                       audio_dir=os.path.abspath(args.audio_dir))
    
    save_result(wer_result)

    print(f"Unweighted Average WER: {wer_result['Unweighted Average WER']}")
    print(f"Weighted Average WER: {wer_result['Weighted Average WER']}")


if __name__ == "__main__":
    main()