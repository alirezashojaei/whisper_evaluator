import time

import torch
import whisper
from whisper import Whisper
from pandas import read_json, read_excel, Series

from utils import evaluate_wer_detailed, available_models


class Evaluator:

    def __init__(self,
                 model: Whisper) -> None:
        
        start_loading = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model, self.device)
        end_loading = time.time()
        self.load_model_elapsed_time = end_loading - start_loading

    def run_evaluation(self):

        wer_result = evaluate_wer_detailed(model=self.model,
                                       dataset_path=args.dataset_path,
                                       audio_dir=args.audio_dir,
                                       verbose=args.verbose)

        if args.output_file_path:
            df = read_json(wer_result["json_dataframe"], orient='columns')

            df["average_wer"] = Series(dtype='float64')
            num_rows = len(df)
            last_row_index = num_rows - 1
            df.at[last_row_index, 'average_wer'] = wer_result['Weighted Average WER']

            df.to_excel(args.output_file_path, engine="openpyxl", index=False)

        print(f"Unweighted Average WER: {wer_result['Unweighted Average WER']}")
        print(f"Weighted Average WER: {wer_result['Weighted Average WER']}")