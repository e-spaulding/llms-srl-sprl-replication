# How to run

To simplify things, I have preprocessed the data into prompts and stored them in .pkl files. The ontonotes prompt dump is too large for standard Github, so it is zipped. Make sure you unzip it before running the code:

```
cd prompts
unzip onto-data.zip
```

Install the needed libraries in a virtual environment:

```
python3 -m venv /path/to/.venv
source /path/to/.venv/bin/activate
pip install -r requirements.txt
```

Set necessary env variables or make necessary .env file:

```
export WATSONX_APIKEY=.....
export WATSONX_URL=.....
export WATSONX_PROJECT_ID=.....
```

Then run the python script!

```
python run_prompts.py
```

The `experiments` directory will be populated with output from the model. There should be 2 files (for 2 experiments, one for each experiment) in the directory when it is done.
