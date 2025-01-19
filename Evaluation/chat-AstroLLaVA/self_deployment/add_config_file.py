from transformers import LlavaForConditionalGeneration

base_model_name = "liuhaotian/llava-1.5-7b"
model = LlavaForConditionalGeneration.from_pretrained(base_model_name)

print("Model directory:", model.config.name_or_path)
