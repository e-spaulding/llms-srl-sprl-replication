# Main libraries
import torch 
import json 
import logging
import pickle
import pytz
import os
from langchain_ibm import ChatWatsonx
from pydantic import SecretStr
from yaml import load, Loader
from time import process_time
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# # Before importing the transformers library, ensure the models are downloaded and pulled from the correct place
# PATH_TO_CACHE = '/mnt/datastore/elsp2673/.cache/' # this is where the models are downloaded
# import os
# os.environ['TRANSFORMERS_CACHE'] = PATH_TO_CACHE # deprecated
# os.environ['HF_HOME'] = PATH_TO_CACHE # new

load_dotenv()
api_key = SecretStr(os.environ['WATSONX_APIKEY'])
base_url = SecretStr(os.environ['WATSONX_URL'])
project_id = os.environ['WATSONX_PROJECT_ID']

def make_datetime_str():
    # mountain time
    mtn = pytz.timezone('US/Mountain')
    now = datetime.now()
    now = now.astimezone(tz=mtn)
    return now.strftime("%b%d-%I%M%S%p%Z")

config_paths = [
    'configs/Llama-3.1-405B-Instruct-spr1.yaml',
    'configs/Llama-3.1-405B-Instruct-onto.yaml'
]

for config_path in config_paths:
    # Get parameters from config
    with open(config_path, 'r') as f:
        config = load(f, Loader=Loader)

    TODAYS_DATE = make_datetime_str()
    MODEL_NAME = config['model_name']
    PATH_TO_PROMPTS = config['path_to_prompts']
    MAX_LENGTH = config['max_length']

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    chat = ChatWatsonx(
        model_id = MODEL_NAME,
        url = base_url,
        project_id = project_id,
        params = {
            "temperature": 0,
            "max_tokens" : MAX_LENGTH,
            "decoding_method": "greedy"
        },
        apikey = api_key
    )

    # Load prompts
    prompt_contexts_by_id = pickle.load(open(PATH_TO_PROMPTS, 'rb'))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    t_0 = process_time()

    # Initialize config and model
    model_name_for_saving = MODEL_NAME.split('/')[-1]

    results_dict = {id_: {'prompt': prompt_contexts_by_id[id_], 'clean_output': '', 'full_output': ''} for id_ in prompt_contexts_by_id.keys()}
    results_dict['config'] = {}
    results_dict['config']['max_length'] = MAX_LENGTH

    i = 0
    for prompt_id in tqdm(prompt_contexts_by_id.keys(), desc='Running prompts', total=len(prompt_contexts_by_id)):
        prompt = prompt_contexts_by_id[prompt_id]
        
        try:
            model_output = chat.invoke(prompt)
        except Exception as e:
            model_output = str(e)
            logger.error(f'Error with prompt {prompt_id}: {model_output}')

        results_dict[prompt_id]['clean_output'] = model_output.content

        # Save intermediate results

        i += 1
        results_dict['batch_progress'] = i
        with open(f'experiments/{TODAYS_DATE}-{model_name_for_saving}-intermediate.json', 'w') as fp:
            json.dump(results_dict, fp, indent=2)

    t_1 = process_time()
    # save results
    results_dict['total_elapsed_time'] = t_1 - t_0

    with open(f'experiments/{TODAYS_DATE}-{model_name_for_saving}.json', 'w') as fp:
        json.dump(results_dict, fp, indent=2)