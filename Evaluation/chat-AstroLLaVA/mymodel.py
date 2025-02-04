import torch
from PIL import Image
from llava.model.builder import load_pretrained_model  # Import from LLaVA repo
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# Set model name
model_name = "astroLLava_v3"

# Load pretrained model, tokenizer, processor, and config using LLaVA's function
tokenizer, model, processor, model_cfg = load_pretrained_model(
    model_path=model_name,  # Path to the pretrained model
    model_base=None,        # Set to None unless using a base model
    model_name=model_name,  # Model identifier
    torch_dtype=torch.float16,  # Use half precision for efficiency
    device="mps",  # Move to GPU
    load_8bit=False,
    load_4bit=False
)

# Load and process the image
image_path = "JWST_testAstrollava.jpeg"  # Ensure the image is uploaded
image = Image.open(image_path).convert("RGB")  # Convert image to RGB

# Preprocess image using LLaVA's processor
image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
images = image_tensor.unsqueeze(0).half().to("mps") # Add batch dimension and move to GPU

# Define the conversation template (Modify if AstroLLaVA uses a different format)
conv_mode = "llava_v1"  # Ensure this matches AstroLLaVA's conversation format
conv = conv_templates[conv_mode].copy()

# Prepare the prompt with the image token
prompt_template = "Analyze this image and provide insights."
prompt_in = DEFAULT_IMAGE_TOKEN + "\n" + prompt_template
conv.append_message(conv.roles[0], prompt_in)  # Add user prompt
conv.append_message(conv.roles[1], None)  # Model response placeholder

# Tokenize input
input_ids = tokenizer(conv.get_prompt(), return_tensors="pt").input_ids.to("mps")

# Generate response
output_ids = model.generate(input_ids, images=images, max_new_tokens=200, temperature=1.0)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the model's response
print("Generated Response:")
print(response)
