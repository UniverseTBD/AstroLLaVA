import os
import json
import time
import openai
from multiprocessing import Pool
from tqdm import tqdm
import importlib
import random
import copy
import argparse

import logging


MAX_TOKENS = 4097

import json

#We need to make this work
class APDataset:
    def __init__(self, data):
        self.dataset = data
    
    def append(self, entry):
        self.dataset.append(entry)

    def write_dataset(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.dataset, f)

    def remove_union(self, other_dataset):
        # Assuming 'id' is the unique key for each entry
        other_ids = {entry['id'] for entry in other_dataset.dataset}
        self.dataset = [entry for entry in self.dataset if entry['id'] not in other_ids]

def load_dataset(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        return APDataset(data)
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
        return APDataset([])  # Return an empty dataset
    except json.JSONDecodeError:
        print(f"The file {input_file} could not be decoded as JSON.")
        return APDataset([])  # Return an empty dataset



class QAGenerator:

    def __init__(self, prompt_file: str, mode: str, n_processes: int = 4) -> None:
        self.n_processes = n_processes
        module = self.load_module(prompt_file)
        assert mode in ['conv'], "Mode must be either 'conv' or 'desc' or 'both but received {mode}".format(mode=mode)
        self.mode = mode

        self.prompt = module.PROMPT_CONV
        self.questions = None
        #elif mode == 'both':

    def load_module(self, path: str):
        """
        Allows to load variables from any module.py.
        """
        spec = importlib.util.spec_from_file_location("settings", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def concat_conversation(self, entry: dict) -> str:
        """
        Convert the conversation into a single string. Each user is separated with '\n\nUser:'.
        If the conversation is not a string or is too long, respectectively discard or crop the conversation.
        """
        conversation = ""
        for j in range(len(entry['conversations'])):
            message = entry['conversations'][j]['value']
            if isinstance(message, str):
                conversation += "User: " + message + "\n\n"

        # Maximum number of tokens you can send to this model is 2,048 tokens per request.
        # TODO: check the size of the conversation

        return conversation
    
    def get_question(self) -> str:
        if self.mode == 'desc':  # V0
            return self.questions[random.randint(0, len(self.questions) - 1)]
        elif self.mode == 'conv':  # V1
            return None

    def get_answer_from_gpt(self, entry: dict) -> list:
        """
        Send the content to GPT and return the answer into a question/answer format.
        """
        last_e = None
        conversation = self.concat_conversation(entry)
        question = self.get_question()
        if self.mode == "desc":
            content = copy.deepcopy(self.prompt) % (conversation, question)
        elif self.mode == "conv":
            content = copy.deepcopy(self.prompt) % conversation

        if conversation is not None:
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{'role': 'user', 'content': content}],
                        temperature=0,
                    )

                    answer = response['choices'][0]['message'].content

                    if self.mode == "desc":  # V0
                        question_and_answer = [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]
                    elif self.mode == "conv":  # V1
                        question_and_answer = json.loads(answer)

                    obj = {
                        "id": "{}".format(entry['id']),
                        "image": "{}".format(entry['id']) + os.path.splitext(entry['image'])[1],
                        "conversations": question_and_answer
                    }

                    return obj
                
                except openai.error.RateLimitError as e:
                    # While GPT is not responding due to rate limit...
                    if not isinstance(last_e, type(e)):  # To prevent multiple printing of the same error every 1s
                        last_e = e
                        logging.error(e)
                    pass
                except Exception as e:
                    logging.error(e)
                    return None
                
                time.sleep(1)
        
        return None

    def generate(self, input_file: str, output_file: str, n_inputs: int = -1, recover_from: str = None) -> APDataset:
        """
        Run the generation of questions and answers.
        Use multiprocessing to parallelize the generation of summaries by calling call_api over a list of prompts.
        """
        dataset_input = load_dataset(input_file) #Tto be implemented
        if recover_from is None:
            dataset_output = load_dataset(input_file)
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
    openai.api_key = os.getenv(args.openai_api_key)

    qa_generator = QAGenerator(args.prompt_file, args.mode, args.n_processes)
    qa_generator.generate(args.input_file, args.output_file, args.n_inputs, args.recover_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--prompt-file", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--n-inputs", type=int, default=-1)
    parser.add_argument("--n-processes", type=str, default=4)
    parser.add_argument("--recover-from", type=str)
    parser.add_argument("--openai-api-key", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    main(args)
    