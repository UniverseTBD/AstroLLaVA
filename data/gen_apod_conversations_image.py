import os
import json
import time
from openai import OpenAI
from multiprocessing import Pool
from tqdm import tqdm
import importlib
import random
import copy
import argparse

import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # for dat exponential backoff


MAX_TOKENS = 2048 

import json

#We need to make this work
class APODataset:
    def __init__(self, data):
        self.dataset = data
    
    def append(self, entry):
        self.dataset.append(entry)

    def write_dataset(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.dataset, f, sort_keys = True, indent = 4)

    def remove_union(self, other_dataset):
        # Assuming 'date' is the unique key for each entry
        other_ids = {entry['date'] for entry in other_dataset.dataset}
        self.dataset = [entry for entry in self.dataset if entry['date'] not in other_ids]

def load_dataset(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        return APODataset(data)
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
        return APODataset([])  # Return an empty dataset
    except json.JSONDecodeError:
        print(f"The file {input_file} could not be decoded as JSON.")
        return APODataset([])  # Return an empty dataset



class QAGenerator:

    def __init__(self, prompt_file: str, n_processes: int = 4) -> None:
        self.n_processes = n_processes
        module = self.load_module(prompt_file)

        self.prompt = module.PROMPT_CONV
        self.questions = None

    def load_module(self, path: str):
        """
        Allows to load variables from any module.py.
        """
        spec = importlib.util.spec_from_file_location("settings", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def get_explanation(self, entry: dict) -> str:
        """
        Get the explanation from the entry.
        """
        return entry['explanation']

    def get_image(self, entry: dict) -> str:
        """
        Get the image url from the entry.
        """
        return entry['url']

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(16))
    def get_answer_from_gpt(self, entry: dict) -> list:
        """
        Send the content to GPT and return the answer into a question/answer format.
        """
        client = OpenAI(
            #api_type = os.getenv("OPENAI_API_TYPE"),
            base_url = os.getenv("OPENAI_API_BASE"),
            azure_endpoint = os.getenv("OPENAI_API_BASE"),
            api_key = os.getenv("OPENAI_API_KEY"),
            api_version= "2023-12-01-preview",
            #api_locale = os.getenv("OPENAI_API_LOCALE"),
            #api_version = os.getenv("OPENAI_API_VERSION"),
        )

        last_e = None
        explanation = self.get_explanation(entry)
        url = copy.deepcopy(self.get_image(entry))
        content = copy.deepcopy(self.prompt) % explanation

        if explanation is not None:
            while True:
                try:
                    response = client.chat.completions.create(
                        model="gpt4_vision_preview",
                        messages=[
                            {
                                'role': 'system',
                                'content': [
                                    {'type': 'text', 'text': content},
                                    {'type': 'image_url', 'image_url': {'url': url, 'detail': 'low'}},
                                ],
                            },
                        ],
                        temperature=0,
                        max_tokens=MAX_TOKENS,
                    )

                    answer = response.choices[0].message.content

                    question_and_answer = json.loads(answer)

                    obj = {
                        "date": "{}".format(entry['date']),
                        "url": "{}".format(entry['url']),
                        "explanation": "{}".format(entry['explanation']),
                        "conversation": question_and_answer
                    }

                    return obj
                
                except Exception as e:
                    logging.error(e)
                    return None
                
                time.sleep(1)
        
        return None

    def generate(self, input_file: str, output_file: str, recover_from: str = None) -> APODataset:
        """
        Run the generation of questions and answers.
        Use multiprocessing to parallelize the generation of summaries by calling call_api over a list of prompts.
        """

        dataset_input = load_dataset(input_file)
        if recover_from is None:
            # Initialise new dataset
            dataset_output = APODataset([])
        else:
            dataset_output = load_dataset(recover_from)
            dataset_output.write_dataset(output_file)
            dataset_input.remove_union(dataset_output)
        with Pool(self.n_processes) as pool:
            for result in tqdm(pool.imap(self.get_answer_from_gpt, dataset_input.dataset), total=len(dataset_input.dataset), desc="Generating QA"):
                if result is not None:
                    dataset_output.append(result)
                # Write to json every 100 answers
                if len(dataset_output.dataset) % 100 == 0:
                    dataset_output.write_dataset(output_file)
        dataset_output.write_dataset(output_file)

        return dataset_output


def main(args):
    from dotenv import load_dotenv
    load_dotenv()

    qa_generator = QAGenerator(args.prompt_file, args.n_processes)
    qa_generator.generate(args.input_file, args.output_file, args.recover_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, type=str)
    parser.add_argument("--input-file", required=True, type=str)
    parser.add_argument("--output-file", default="apod_conversations_image.json", type=str)
    parser.add_argument("--n_processes", type=int, default=4)
    parser.add_argument("--recover-from", type=str)
    args = parser.parse_args()

    main(args)
