import os
import time
import pickle
import torch
import whisper
from whisper import Whisper
from faster_whisper import WhisperModel
from pandas import read_excel, Series
from numpy import isnan
import evaluate
from tqdm import tqdm
from arabic_reshaper import reshape
from bidi.algorithm import get_display
from pathlib import Path


class Evaluator:

    def __init__(self,
                 model: Whisper,
                 method: str,
                 verbose: bool,
                 load_from_checkpoint: bool) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0)
        self.method = method
        start_loading = time.time()
        if self.method == "openai-whisper":
            self.model = whisper.load_model(model, self.device)
        elif self.method == "faster-whisper":
            self.model = WhisperModel(model, device=self.device,
                                      compute_type="float16",
                                      local_files_only=True)
        else:
            raise ValueError("'--transcription-method' should be either 'faster-whisper' or 'openai-whisper")
        end_loading = time.time()
        self.load_model_elapsed_time = end_loading - start_loading
        self.load_from_checkpoint = load_from_checkpoint
        self.verbose = verbose

    def transcribe_openai_whisper(self,
                                  audio_path: str,
                                  language: str = "fa",
                                  ):
        """
        The transcribe_openai_whisper function transcribes an audio file using the OpenAI Whisper model.

        :param self: Represent the instance of the class
        :param audio_path: str: Pass the path of the audio file to be transcribed
        :param language: str: Specify the language of the audio file
        :return: A dictionary with two keys: text and elapsed_time
        """
        start_transcribe = time.perf_counter()
        result = self.model.transcribe(audio=str(audio_path), language=language)
        end_transcribe = time.perf_counter()
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

        """
        The transcribe_faster_whisper function transcribes an audio file using
        the faster-whisper module.
            Args:
                audio_path (str): The path to the audio file to be transcribed.
                language (str): The language of the transcription, defaults to &quot;fa&quot;.

        :return: A dictionary with two keys:
        """
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

    def create_dataset(self, dataset_path):
        """
        The create_dataset function takes a path to an excel file and returns a pandas dataframe with the following columns:
            id - The ID of the sentence in the dataset.
            text - The original sentence.
            pred_text - An empty string, which will be filled by predict_dataset function.
                        This is where we will store our predicted sentences for each row in our dataset.
                        We can then compare these predictions to their corresponding ground truth values
                        (the &quot;text&quot; column).

        :param self: Represent the instance of the class
        :param dataset_path: Specify the path to the dataset
        :return: A pandas dataframe
        """
        exl = read_excel(dataset_path, engine="openpyxl")
        exl["id"] = exl["id"].astype("string")
        exl["sentence"] = exl["sentence"].astype("string")
        exl.rename(columns={'sentence': 'text'}, inplace=True)
        exl["pred_text"] = ""
        exl["wer"] = Series(dtype='float64')
        exl["elapsed_time"] = Series(dtype='float64')
        exl["device"] = self.device_name

        return exl

    def run_evaluation(self,
                       dataset_path: str,
                       audio_dir: str):

        """
        The run_evaluation function takes in a dataset_path and an audio_dir. It then creates a dataframe from the
        dataset path, which is used to store the WER scores for each audio file. The function iterates through all
        the files in the audio directory, transcribes them using either OpenAI's or Faster Transformer's Whisper
        model (depending on what was specified by --transcription-method), and computes their WERs. The results are
        stored in a dictionary that is returned at the end of this function.

        :param self: Bind the method to an object
        :param dataset_path: str: Specify the path to the dataset
        :param audio_dir: str: Specify the directory where the audio files are stored
        :return: A dictionary containing the unweighted and weighted average WERs,
                as well as a json string of the dataframe
        """
        reference_texts, recognized_texts = [], []
        evaluator = evaluate.load("wer")
        score_sum = 0

        if self.load_from_checkpoint:
            if os.path.exists("checkpoint.p"):
                with open("checkpoint.p", "rb") as f:
                    exl = pickle.load(f)
                tqdm.write("Checkpoint loaded. Continuing evaluation ... ")
            else:
                tqdm.write("Checkpoint file not found. Skipping loading from checkpoint ... ")
                exl = self.create_dataset(dataset_path)
        else:
            exl = self.create_dataset(dataset_path)

        for idx, audio_path in tqdm(enumerate(Path(audio_dir).iterdir())):
            audio_name = audio_path.stem

            if audio_name not in exl.id.to_list():
                if self.verbose:
                    tqdm.write(f"Sentence not found for {audio_name}")
                continue

            reference_row = exl[exl["id"] == audio_name]

            if not isnan(reference_row["wer"].iloc[0]):
                continue

            reference_text = reference_row["text"].iloc[0]

            if self.method == "openai-whisper":
                result = self.transcribe_openai_whisper(audio_path=str(audio_path), language="fa")

            elif self.method == "faster-whisper":
                result = self.transcribe_faster_whisper(audio_path=str(audio_path), language="fa")

            else:
                raise ValueError("'--transcription-method' should be either 'faster-whisper' or 'openai-whisper")

            recognized_text = result["text"]
            elapsed_time = result["elapsed_time"]

            reference_texts.append(reference_text)
            recognized_texts.append(recognized_text)

            score = evaluator.compute(references=[reference_text], predictions=[recognized_text])

            if self.verbose:
                tqdm.write(f"File Processed: {reference_row['id'].iloc[0]}")
                tqdm.write(f"Reference Text: {get_display(reshape(reference_text))}")
                tqdm.write(f"Recognized Text: {get_display(reshape(recognized_text))}")
                tqdm.write(f"Elapsed Time: {elapsed_time}")
                tqdm.write(f"WER: {score}")

            score_sum += score

            exl.at[reference_row.index[0], 'wer'] = score
            exl.at[reference_row.index[0], 'pred_text'] = recognized_text
            exl.at[reference_row.index[0], 'elapsed_time'] = elapsed_time
            exl.at[reference_row.index[0], 'device'] = self.device_name

            if idx % 100 == 0 and idx != 0:
                with open("checkpoint.p", "wb") as f:
                    pickle.dump(exl, f)

        weighted_average = evaluator.compute(references=reference_texts, predictions=recognized_texts)

        json_data = exl.to_json(orient='columns')

        wer_result = {"Unweighted Average WER": score_sum / len(reference_texts),
                      "Weighted Average WER": weighted_average,
                      "json_dataframe": json_data}

        return wer_result
