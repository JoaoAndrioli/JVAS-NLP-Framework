from ollama import pull, list

def checkModelExists(model_name):
    """
    Check if the model exists in the local system.
    """
    try:
        #Check if model is already downloaded
        modelsDownloaded = list()
        for modelL in modelsDownloaded:
            for model in modelL[1]:
                if model_name in model.model:
                    return True
    except:
        pass
    return False

def downloadModel(model_name):
    """
    Download the model if it doesn't exist.
    """
    try:
        #Pull model
        print(f"Downloading model {model_name}")
        pull(model_name)
    except:
        pass

def pull_model(model_name):
    modelAlreadyDownloaded = checkModelExists(model_name)
    if not modelAlreadyDownloaded:
        downloadModel(model_name)

#### pull_model('nomic-embed-text')