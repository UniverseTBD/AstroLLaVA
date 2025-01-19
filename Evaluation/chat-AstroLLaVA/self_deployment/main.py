import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import requests
import torchvision.transforms as transforms

# Load the base LLaVA model in float16
base_model_path = "local/liuhaotian/llava-v1.5-7b"
print(f"Obtaining Local Model from {base_model_path}")
model = LlavaForConditionalGeneration.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(base_model_path)

# Apply the AstroLLaVA adapter
adapter_model_name = "universeTBD/AstroLLaVA_v2"
print(f"Setting up the adapter {adapter_model_name}")
model = PeftModel.from_pretrained(model, adapter_model_name)

# Custom image preprocessing
def preprocess_image(image, target_size=(224, 224)):
    # Resize image to match model's expected input
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()  # Remove normalization
    ])
    
    # Apply transformation
    processed_image = transform(image).unsqueeze(0)
    return processed_image

# Conversion function to handle normalized tensors
def tensor_to_pil(tensor):
    # Denormalize and convert tensor to PIL Image
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.clamp(0, 1)  # Clamp values to [0, 1]
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

# Load and preprocess the image
print("Loading the image")
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Preprocess image manually
processed_image = preprocess_image(image)

# Prepare input using processor with text
print("Preparing the input")
inputs = processor(
    images=processed_image, 
    text="Describe this image in detail", 
    return_tensors="pt", 
    padding=True
)

# Use the appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate output
print("Generating output")
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print the generated text
generated_text = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)

# Optional: Convert processed image back to PIL for viewing
# pil_image = tensor_to_pil(processed_image)
# pil_image.show()
