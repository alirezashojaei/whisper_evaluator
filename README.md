# Whisper Model Evaluation Script

## Overview
This script is designed to evaluate the performance of different models transcribing audio files by using Whisper models. It generates Word Error Rates (WER) for given audio files compared to reference transcriptions, and outputs the results in an Excel file.

## Features

- Supports evaluation with OpenAI's Whisper and alternative `faster-whisper` models.
- Provides both unweighted and weighted average WER results.
- Outputs the evaluation results and associated data to an Excel file for easy analysis.
- Prints verbose logs of evaluation results if enabled.

## Prerequisites

Before running the script, ensure the following dependencies are installed:

- Python 3.9+
- `pandas`
- `torch`
- `whisper`
- `faster_whisper` (optional alternative model)
- `evaluate`
- `tqdm`
- `arabic_reshaper` and `bidi algorithm` (for reshaping and reordering Arabic(or Persian) text)
- `openpyxl` (for writing to Excel files)

Install them using pip:

```sh
pip install pandas torch whisper faster_whisper evaluate tqdm arabic_reshaper python-bidi openpyxl
```

## Usage

Run the script from the command line with the desired arguments:

```sh
python run_test.py --model MODEL_NAME --audio-dir PATH_TO_AUDIO_FILES --reference-path PATH_TO_REFERENCE_TEXT --output-dir OUTPUT_FILE_PATH --transcription-method TRANSCRIPTION_METHOD [--verbose]
```

### Arguments description:

`--model` (str): Select the Whisper model to use.
`--audio-dir` (str): Path to the directory containing audio files for transcription.
`--reference-path` (str): Path to the Excel file containing reference texts.
`--output-dir` (str): Path to the output Excel file.
`--transcription-method` (str): Transcribe using ‘openai-whisper’ or the ‘faster-whisper’ package.
`--verbose` (flag): Enable detailed logs of each file’s evaluation result.


Remember to replace placeholders like `MODEL_NAME`, `PATH_TO_AUDIO_FILES`, `PATH_TO_REFERENCE_TEXT`, and `OUTPUT_FILE_PATH` with actual values when running the command.

## Outputs

- Excel file containing text predictions, corresponding reference text, WER for individual files, and average WER for all files

## Note

- This script assumes that the filenames in the audio directory match the IDs in the Excel sheet of recognized text. It is crucial for matching audio files with their corresponding reference transcriptions.