import torch
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from itertools import islice
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class Galaxy10Evaluator:
    def __init__(self, model_path, model_base=None, batch_size=8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer using original LLaVa code
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name
        )
        
        # Determine conversation mode
        if "llama-2" in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Galaxy class mapping with key features
        self.class_mapping = {
            0: {
                "name": "Disturbed Galaxies",
                "features": ["irregular shape", "asymmetric", "disturbed structure", "unusual patterns", "deformed"]
            },
            1: {
                "name": "Merging Galaxies",
                "features": ["multiple cores", "interaction", "tidal tails", "bridges", "overlapping"]
            },
            2: {
                "name": "Round Smooth Galaxies",
                "features": ["circular", "elliptical", "smooth", "regular", "uniform brightness"]
            },
            3: {
                "name": "In-between Round Smooth Galaxies",
                "features": ["somewhat round", "partially smooth", "mild irregularity", "intermediate shape"]
            },
            4: {
                "name": "Cigar Shaped Smooth Galaxies",
                "features": ["elongated", "cigar-shaped", "smooth", "uniform", "linear structure"]
            },
            5: {
                "name": "Barred Spiral Galaxies",
                "features": ["bar structure", "spiral arms", "central bar"]
            },
            6: {
                "name": "Unbarred Tight Spiral Galaxies",
                "features": ["tight spiral arms", "no bar", "well-defined arms", "compact structure"]
            },
            7: {
                "name": "Unbarred Loose Spiral Galaxies",
                "features": ["loose spiral arms", "no bar", "open structure", "widely spaced arms"]
            },
            8: {
                "name": "Edge-on Galaxies without Bulge",
                "features": ["edge-on", "thin disk", "no central bulge", "linear structure"]
            },
            9: {
                "name": "Edge-on Galaxies with Bulge",
                "features": ["edge-on", "central bulge", "thick center", "disk structure"]
            }
        }

    def evaluate_description(self, description, true_class):
        """
        Evaluate a single description against the true class using multiple metrics
        """
        true_class_info = self.class_mapping[true_class]
        true_class_name = true_class_info["name"]
        true_features = true_class_info["features"]
        class_text = f"{true_class_name}. Features: {', '.join(true_features)}"
        
        # 1. Feature Match Score
        feature_count = 0
        for feature in true_features:
            if feature.lower() in description.lower():
                feature_count += 1
        feature_score = feature_count / len(true_features)
        
        # 2. Class Name Recognition
        class_mentioned = true_class_name.lower() in description.lower()
        
        # 3. Semantic Similarity
        desc_embedding = self.sentence_model.encode([description])
        class_embedding = self.sentence_model.encode([class_text])
        semantic_score = cosine_similarity(desc_embedding, class_embedding)[0][0]
        
        # 4. Confidence Analysis
        uncertainty_phrases = ["possibly", "maybe", "might be", "could be", "uncertain"]
        confidence_score = 1.0
        for phrase in uncertainty_phrases:
            if phrase in description.lower():
                confidence_score *= 0.8
        
        return {
            "feature_score": feature_score,
            "class_mentioned": class_mentioned,
            "semantic_score": semantic_score,
            "confidence_score": confidence_score
        }

    def prepare_prompt(self, custom_prompt=None):
        """Prepare the conversation prompt"""
        default_prompt = "Describe the following galaxy image in detail. What type of galaxy is it and what are its key features?"
        prompt_text = custom_prompt if custom_prompt else default_prompt
        
        # Set up conversation
        conv = conv_templates[self.conv_mode].copy()
        
        # Add image token and prompt
        if self.model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN
            
        qs = image_token + "\n" + prompt_text
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate_description(self, image, prompt):
        """Generate description for a single image"""
        # Process image
        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Prepare input ids
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=200,
                use_cache=True,
            )
            
        # Decode output
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def evaluate_dataset(self, custom_prompt=None):
        """Evaluate the model on the test dataset"""
        dataset = load_dataset("matthieulel/galaxy10_decals", split="test", streaming=True)
        data_iterator = iter(dataset)
        
        results = []
        prompt = self.prepare_prompt(custom_prompt)
        
        # Process the streaming dataset
        with tqdm(desc="Processing images") as pbar:
            while True:
                try:
                    batch = list(islice(data_iterator, self.batch_size))
                    if not batch:
                        break
                except StopIteration:
                    break
                
                for item in batch:
                    description = self.generate_description(item['image'], prompt)
                    results.append({
                        'true_class': self.class_mapping[item['label']]["name"],
                        'description': description
                    })
                
                pbar.update(len(batch))
        
        results_df = pd.DataFrame(results)
        metrics = self.calculate_aggregate_metrics(results_df)
        
        return results_df, metrics

    def calculate_aggregate_metrics(self, results_df):
        """Calculate aggregate metrics across all evaluations"""
        all_metrics = []
        
        for _, row in results_df.iterrows():
            true_class = row['true_class']
            description = row['description']
            
            # Find the class index from the name
            class_index = None
            for idx, class_info in self.class_mapping.items():
                if class_info["name"] == true_class:
                    class_index = idx
                    break
            
            if class_index is not None:
                metrics = self.evaluate_description(description, class_index)
                all_metrics.append(metrics)
        
        # Calculate average scores
        avg_metrics = {
            "avg_feature_score": np.mean([m["feature_score"] for m in all_metrics]),
            "class_mention_rate": np.mean([m["class_mentioned"] for m in all_metrics]),
            "avg_semantic_score": np.mean([m["semantic_score"] for m in all_metrics]),
            "avg_confidence_score": np.mean([m["confidence_score"] for m in all_metrics])
        }
        
        # Calculate composite score
        composite_score = (
            0.25 * avg_metrics["avg_feature_score"] +
            0.25 * avg_metrics["class_mention_rate"] +
            0.25 * avg_metrics["avg_semantic_score"] +
            0.25 * avg_metrics["avg_confidence_score"]
        )
        
        avg_metrics["composite_score"] = composite_score
        return avg_metrics

    def save_results(self, results_df, metrics, output_path="llava_galaxy_results"):
        """Save evaluation results and metrics to files"""
        # Save descriptions
        results_df.to_csv(f"{output_path}_descriptions.csv", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_path}_metrics.csv", index=False)
        
        print(f"Results and metrics saved to {output_path}_descriptions.csv and {output_path}_metrics.csv")

    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'image_processor'):
            del self.image_processor
        if hasattr(self, 'sentence_model'):
            self.sentence_model.cpu()
            del self.sentence_model
        torch.cuda.empty_cache()

def main():
    model_paths = [
        "UniverseTBD/AstroLLaVA_v3",
    ]
    
    for model_path in model_paths:
        evaluator = Galaxy10Evaluator(
            model_path=model_path,
            model_base=None,  # Set if using a separate base model
            batch_size=8
        )
        
        custom_prompt = "Describe the following galaxy image in detail. What type of galaxy is it and what are its key features?"
        
        # Run evaluation
        results_df, metrics = evaluator.evaluate_dataset(custom_prompt=custom_prompt)
        
        # Save results
        model_name = model_path.split("/")[-1]
        evaluator.save_results(results_df, metrics, output_path=f'results/{model_name}')
        
        # Print metrics
        print(f"\nEvaluation Metrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
        
        evaluator.cleanup()

if __name__ == "__main__":
    main()