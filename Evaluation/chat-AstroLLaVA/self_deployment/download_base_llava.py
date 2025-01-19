from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "liuhaotian/llava-v1.5-7b"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
