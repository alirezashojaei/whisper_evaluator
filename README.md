# [Work In Progress] whisper-finetuning
This is a collection of scripts that can be used to fine-tune a Whisper model using <strong>time-aligned</strong> transcriptions and audio files.
Although there are already codes available for fine-tuning a Whisper model, such as the one provided by the Hugging Face transformers library (https://huggingface.co/blog/fine-tune-whisper), they only offer a way to fine-tune a model using transcripts <strong>without timestamps</strong>.
This makes it difficult to output timestamps along with the transcriptions.
This repository, however, provides scripts that allow you to fine-tune a Whisper model using time-aligned data, making it possible to output timestamps with the transcriptions.

## Setup
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
To install pytorch, you may need to follow the instructions [here](https://pytorch.org/get-started/locally/) depending on your environment.
You may also need to install [ffmpeg](https://ffmpeg.org/) and [rust](https://www.rust-lang.org/) to run Whisper.
See the [instructions](https://github.com/openai/whisper#setup) in the Whisper repository for more details if you encounter any errors.

### Windows 
You will need to install the `soundfile` python package.
```
pip install soundfile
```
On Windows, you also have to change the `num_workers` parameter in `dataloader.py` (almost at the end of the file) by setting it to 0.

For using the Adam 8bit optimizer with the `bitsandbytes` package, you will need to download pre-built binaries from another repo, since by default it is not supported.
You can grab the .dll from [here](https://github.com/DeXtmL/bitsandbytes-win-prebuilt) or more easily download [this folder](https://github.com/bmaltais/kohya_ss/tree/master/bitsandbytes_windows) with the .dll and .py to patch and copy them using the following commands:
```
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```

## Usage
### 1. Prepare your data (`to_txt.py` and `create_data.py`)
1- You should have a directory containing audio files.

2- Create a .txt file that lists the audio files and their corresponding transcripts. 
Each line should contain the path to an audio file and its transcription, separated by a tab. Or create one with the following command 
from any other datasets:
```
python to_txt.py --reference-file data-files/<path-to-your-xlsx-csv> --output-file-path data-files/data.txt --files-dir audio --if-exists True --format <your-audio-file-format-example-.wave>
```
That command will create a txt file like this:
```
audio/sample1.wav    Hello, this is a test.
audio/sample2.wav    This is another example.
```
3- To generate jsonl files, run the following command. It will create two jsonl files. 
one for validation and one for train with the ratio of 97 by 3. (`train_dataset.json` and `dev_train_dataset.json`):
```
python create_data.py --without-timestamps --data-file data-files/data.txt --language fa --output data-files/train_dataset.json --train-dev-ratio 97-3
```


### 2. Finetune a model (`run_finetuning.py`)
You can finetune a model with the jsonl files generated in the previous step:
```
python run_finetuning.py --train-json data-files/train_dataset.json --dev-json data-files/dev_train_dataset.json --batch-size 2 --train-steps 9 --eval-steps 3 --no-timestamps-training --save-all-checkpoints --model <model-name>
```

Your models will be saved in `output` directory containing `last_model.pt`, `best_model.pt` and steps.

You can use the `--use-adam-8bit` flag to utilize the Adam 8bit optimizer from `bitsandbytes`. This will reduce VRAM usage and allows to train using small multimodal models with 8GB of VRAM.

Use the `--training-steps <number>` flag to specify training epochs.

For all available options, see `python run_finetuning.py --help`.

### 3. Test your model, calculate metric and elapsed times using a dataset (`calculate_metric.py`)
You can transcribe audio files and calculate a metric such as Word Error Rate (WER) using the fine-tuned model by running the command:
```

python calculate_metric.py --verbose --output-file-path metric_files/evaluation.xlsx --model <path-to-finetuned-model>
```
This will use the model to save transcribed text and metric and elapsed time in a xlsx file like `metric_files/evaluation.xlsx`

For all available options, see `python calculate_metric.py --help`.
