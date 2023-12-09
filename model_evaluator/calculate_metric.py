import time

import torch
import whisper
from whisper import Whisper
from faster_whisper import WhisperModel
from pandas import read_excel, Series
import evaluate
from tqdm import tqdm
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from pathlib import Path


class Evaluator:

    def __init__(self,
                 model: Whisper,
                 method: str,
                 verbose: bool) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0)
        self.method = method
        start_loading = time.time()
        if self.method == "openai-whisper": self.model = whisper.load_model(model, self.device)
        elif self.method == "faster-whisper": self.model = WhisperModel(model, device=self.device, 
                                                                        compute_type="float16", 
                                                                        local_files_only=True)
        else: raise ValueError("'--transcription-method' should be either 'faster-whisper' or 'openai-whisper")
        end_loading = time.time()
        self.load_model_elapsed_time = end_loading - start_loading
        self.verbose = verbose


    def transcribe_openai_whisper(self,
                                  audio_path: str,
                                  language: str = "fa",
                                  ):

        start_transcribe = time()
        result = self.model.transcribe(audio=str(audio_path), language=language)
        end_transcribe = time()
        text = result["text"]
        elapsed_time = end_transcribe - start_transcribe

        return {
            "text": text,
            "elapsed_time": elapsed_time
        }
    

    def transcribe_faster_whisper(self,
                                  audio_path: str,
                                  language: str = "fa",
                                  ):

        segments, _ = self.model.transcribe(str(audio_path), language=language, 
                                      vad_filter=True,
                                      vad_parameters=dict(min_silence_duration_ms=500),
                                      )

        start_transcribe = time.perf_counter()
        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        transcription = transcription.strip()
        end_transcribe = time.perf_counter()
        elapsed_time = end_transcribe - start_transcribe

        return {
            "text": transcription,
            "elapsed_time": elapsed_time
        }


    def run_evaluation(self,
                       dataset_path: str,
                       audio_dir: str):

        reference_texts, recognized_texts = [], []
        evaluator = evaluate.load("wer")
        score_sum = 0
        exl = read_excel(dataset_path, engine="openpyxl")
        exl["id"] = exl["id"].astype("string")
        exl["sentence"] = exl["sentence"].astype("string")
        exl.rename(columns={'sentence': 'text'}, inplace=True)
        exl["pred_text"] = ""
        exl["wer"] = Series(dtype='float64')
        exl["elapsed_time"] = Series(dtype='float64')
        exl["device"] = self.device_name

        for audio_path in tqdm(Path(audio_dir).iterdir()):
            audio_name = audio_path.stem

            if audio_name in exl.id.to_list():
                reference_row = exl[exl["id"] == audio_name]
                reference_text = reference_row["text"].iloc[0]

                if self.method == "openai-whisper": 
                    result = self.transcribe_openai_whisper(audio=str(audio_path), language="fa")

                elif self.method == "faster-whisper": 
                    result = self.transcribe_faster_whisper(audio=str(audio_path), language="fa")

                else: raise ValueError("'--transcription-method' should be either 'faster-whisper' or 'openai-whisper")

                recognized_text = result["text"]
                elapsed_time = result["elapsed_time"]
            else:
                if self.verbose:
                    tqdm.write(f"Sentence not found for {audio_name}")
                continue

            reference_texts.append(reference_text)
            recognized_texts.append(recognized_text)

            score = evaluator.compute(references=[reference_text], predictions=[recognized_text])

            if self.verbose:
                tqdm.write(f"Reference Text: {get_display(reshape(reference_text))}")
                tqdm.write(f"Recognized Text: {get_display(reshape(recognized_text))}")
                tqdm.write(f"Elapsed Time: {elapsed_time}")
                tqdm.write(f"WER: {score}")

            score_sum += score

            exl.at[reference_row.index[0], 'wer'] = score
            exl.at[reference_row.index[0], 'pred_text'] = recognized_text
            exl.at[reference_row.index[0], 'elapsed_time'] = elapsed_time

        weighted_average = evaluator.compute(references=reference_texts, predictions=recognized_texts)

        json_data = exl.to_json(orient='columns')
    
        wer_result = {"Unweighted Average WER": score_sum / len(reference_texts),
                "Weighted Average WER": weighted_average,
                "json_dataframe": json_data}

        return wer_result