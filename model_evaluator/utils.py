import os
import warnings

from whisper import available_models as whisper_models

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
