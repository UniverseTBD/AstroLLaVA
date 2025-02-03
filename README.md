# AstroLLaVA 🚀 🌋 🦙

AstroLLaVA is a visual language model for astronomy that enables interaction with astronomical imagery through natural dialogue. Built on the LLaVA architecture, it has been fine-tuned on a dataset of ~30k astronomical images with captions and question-answer pairs.

## Quickstart instructions

- Check out the HF repo and space to try out the model (TODO LINK)

## Dataset

Our training dataset consists of approximately 30,000 image-caption pairs from:

- NASA's Astronomy Picture of the Day (APOD): ~10k images
- European Southern Observatory (ESO): ~15k images  
- Hubble Space Telescope Archive: ~5k images

Each image is paired with expert-written captions and synthetic question-answer pairs generated using GPT-4.

## Installation

```bash
# Clone the repository
git clone https://github.com/universeTBD/AstroLLaVA
cd AstroLLaVA

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from astrollava import AstroLLaVA

# Initialize model
model = AstroLLaVA.from_pretrained("universeTBD/AstroLLaVA")

# Load an image
image = load_image("galaxy.jpg")

# Ask questions about the image
response = model.generate_response(image, "What interesting features do you see in this galaxy?")
print(response)
```

## How to finetune your own `*LLaVA`

TODO

## Model Architecture

AstroLLaVA combines:
- CLIP ViT-L/14 vision encoder (pretrained at 336px resolution)
- LLaMA 3 7.0B language model
- Custom projection layers for bridging visual and language domains

The model is trained in two stages:
1. Training of visual-language projection layers
2. End-to-end instruction tuning with astronomical QA pairs

## Resources

- [Training Dataset](https://doi.org/10.57967/hf/4236)
- [Model Weights](https://hf.co/universeTBD/AstroLLaVA)
- [Discord Community](https://discord.gg/rwNEjfhT)

## Environmental Impact

Training was performed on the ITER Teide HPC cluster using 100% renewable energy. 
Total energy usage was approximately 5kWh on 4xA100-40G GPUs.

## Contributing

We welcome contributions! Join our Discord community to collaborate on improving AstroLLaVA.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
