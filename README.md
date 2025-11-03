# DeckTor: Your Anki Doctor

## Install
```
git clone https://github.com/maurock/decktor.git
cd decktor
conda env create -f environment.yaml
conda activate decktor
```
Install the torch version that works for you. For example:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Download models
We recommend downloading the models first. This will take a while. 
If you don't download the models now, they will be downloaded the first time you run the app.

To selectively dowload models, check the one you want from `src/models.py` -> `SUPPORTED_MODELS` and type the key, e.g.:
```
decktor download-models --models "Qwen3 32B" 
```
or download all the supported ones:
```
decktor download-models
```

## Run
Simply run:
```
decktor run
```


