import os
import warnings

from time import time
from pathlib import Path
import evaluate
from whisper import Whisper
from whisper import available_models as whisper_models
from pandas import read_excel, Series
from tqdm import tqdm
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from create_data import DataProcessor

warnings.simplefilter("ignore")


def available_models(models_directory: str = None) -> list:
    model_list = whisper_models()

    if (models_directory and 
            os.path.exists(models_directory) and 
            os.path.isdir(models_directory)):
        
        dir_list = os.listdir(models_directory)

        directories = [d for d in dir_list if os.path.isdir(os.path.join(models_directory, d))]

        # Filter only files (not directories) in the "output" directory
        files_with_paths = [os.path.join(models_directory, file).replace("\\", "/") for file in dir_list if
                            os.path.isfile(os.path.join(models_directory, file)) and file.endswith(".pt")]
        
        model_directories = []
        # Iterate through the directories and append the path to bin_directories if a .bin file is found
        for d in directories:
            files = os.listdir(os.path.join(models_directory, d))
            if any(file.endswith('.bin') for file in files):
                model_directories.append(os.path.join(models_directory, d))

        model_list.extend(files_with_paths)
        model_list.extend(model_directories)

    return model_list


def evaluate_wer_detailed(model: Whisper,
                          dataset_path: str = "metric_files/reference.xlsx",
                          audio_dir: str = "metric_files/audio",
                          verbose: bool = False) -> dict:
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

    for audio_path in tqdm(Path(audio_dir).iterdir()):
        audio_name = audio_path.stem

        if audio_name in exl.id.to_list():
            reference_row = exl[exl["id"] == audio_name]
            reference_text = reference_row["text"].iloc[0]
            start_transcribe = time()
            result = model.transcribe(audio=str(audio_path), language="fa")
            end_transcribe = time()
            recognized_text = result["text"]
            elapsed_time = end_transcribe - start_transcribe
        else:
            if verbose:
                tqdm.write(f"Sentence not found for {audio_name}")
            continue

        reference_texts.append(reference_text)
        recognized_texts.append(recognized_text)

        score = evaluator.compute(references=[reference_text], predictions=[recognized_text])

        if verbose:
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
 
    return {"Unweighted Average WER": score_sum / len(reference_texts),
            "Weighted Average WER": weighted_average,
            "json_dataframe": json_data}


def evaluate_wer(model: Whisper,
                 dataset_path: str = "data-files/dev_train_dataset.json",
                 verbose: bool = False) -> dict:

    records = DataProcessor.read_records(dataset_path)

    reference_texts, recognized_texts = [], []
    evaluator = evaluate.load("wer")
    score_sum = 0

    for record in tqdm(records):
        audio_path = Path(record.audio_path)
        reference_text = record.text

        if audio_path.exists():
            start_transcribe = time()
            result = model.transcribe(audio=str(audio_path), language="fa")
            end_transcribe = time()
            recognized_text = result["text"]
            elapsed_time = end_transcribe - start_transcribe
        else:
            if verbose:
                tqdm.write(f"Could not find audio file {audio_path}")
            continue

        reference_texts.append(reference_text)
        recognized_texts.append(recognized_text)

        score = evaluator.compute(references=[reference_text], predictions=[recognized_text])

        if verbose:
            tqdm.write(f"Reference Text: {get_display(reshape(reference_text))}")
            tqdm.write(f"Recognized Text: {get_display(reshape(recognized_text))}")
            tqdm.write(f"Elapsed Time: {elapsed_time}")
            tqdm.write(f"WER: {score}")

        score_sum += score

    weighted_average = evaluator.compute(references=reference_texts, predictions=recognized_texts)

    return {"Unweighted Average WER": score_sum / len(reference_texts),
            "Weighted Average WER": weighted_average}
