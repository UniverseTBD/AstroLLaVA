import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer
from llava.model import *
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

class Galaxy10Evaluator:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", batch_size=8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        if hasattr(model, 'tie_weights'):
            model.tie_weights()

        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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

    def calculate_aggregate_metrics(self, results_df):
        """
        Calculate aggregate metrics across all evaluations
        """
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
        
        # Calculate composite score (weighted average)
        composite_score = (
            0.25 * avg_metrics["avg_feature_score"] +
            0.25 * avg_metrics["class_mention_rate"] +
            0.25 * avg_metrics["avg_semantic_score"] +
            0.25 * avg_metrics["avg_confidence_score"]
        )
        
        avg_metrics["composite_score"] = composite_score
        return avg_metrics

    def evaluate_dataset(self, custom_prompt=None):
        """Evaluate the model on the test dataset"""
        dataset = load_dataset("matthieulel/galaxy10_decals", split="test", streaming=True)
        data_iterator = iter(dataset)
        
        results = []
        batch = []
        
        # Process the streaming dataset in batches
        with tqdm(desc="Processing images") as pbar:
            while True:
                # Collect a batch
                try:
                    batch = list(islice(data_iterator, self.batch_size))
                    if not batch:
                        break
                except StopIteration:
                    break
                
                batch_images = [item['image'] for item in batch]
                batch_labels = [item['label'] for item in batch]
                
                inputs = self.prepare_batch(batch_images, custom_prompt)
                descriptions = self.generate_descriptions(inputs)
                
                for label, desc in zip(batch_labels, descriptions):
                    results.append({
                        'true_class': self.class_mapping[label]["name"],
                        'description': desc
                    })
                
                pbar.update(len(batch))
        
        results_df = pd.DataFrame(results)
        metrics = self.calculate_aggregate_metrics(results_df)
        
        return results_df, metrics

    # def prepare_batch(self, batch_images, custom_prompt=None):
    #     """Prepare a batch of images for processing"""
    #     default_prompt = "Describe the following galaxy image in detail. What type of galaxy is it and what are its key features?"
    #     prompt_text = custom_prompt if custom_prompt else default_prompt
    #     
    #     # Prepare conversations
    #     conversations = [
    #         [{
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt_text},
    #                 {"type": "image"},
    #             ],
    #         }] for _ in range(len(batch_images))
    #     ]
    #     
    #     # Apply chat template
    #     prompts = [
    #         self.processor.apply_chat_template(conv, add_generation_prompt=True)
    #         for conv in conversations
    #     ]
    #     
    #     # Process inputs
    #     inputs = self.processor(
    #         text=prompts,
    #         images=batch_images,
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(self.model.device, torch.float16)
    #     
    #     return inputs

    def prepare_batch(self, batch_images, custom_prompt=None):
        """Prepare a batch of images for processing using raw tokenizer and image processor."""
        default_prompt = "Describe the following galaxy image in detail. What type of galaxy is it and what are its key features?"
        prompt_text = custom_prompt if custom_prompt else default_prompt

        # Prepare conversations
        conversations = [
            [{
                "role": "user",
                "content": f"{prompt_text} <image>",  # Combine text and image placeholder
            }] for _ in range(len(batch_images))
        ]

        # Apply chat template using the tokenizer (returns token IDs)
        prompt_ids = [
            self.tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]

        # Debugging: Print prompt IDs to verify their format
        print("Prompt IDs:", prompt_ids)

        # Convert token IDs to tensors
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).to(self.model.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.model.device)

        # Process images
        image_inputs = self.image_processor(
            batch_images,
            return_tensors="pt"
        )

        # Combine inputs and move to device/dtype
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": image_inputs["pixel_values"].to(self.model.device, dtype=torch.float16),
        }

        return inputs
    
    def generate_descriptions(self, inputs):
        """Generate descriptions for a batch of processed inputs using raw tokenizer."""
        with torch.no_grad():
            # Generate text using the model
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode responses using the tokenizer
        responses = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        
        # Clean up responses
        cleaned_responses = [
            response.split("ASSISTANT:")[-1].strip()
            for response in responses
        ]
        
        return cleaned_responses

    def generate_descriptions(self, inputs):
        """Generate descriptions for a batch of processed inputs"""
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode responses
        responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        
        # Clean up responses
        cleaned_responses = [
            response.split("ASSISTANT:")[-1].strip()
            for response in responses
        ]
        
        return cleaned_responses

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
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'sentence_model'):
            self.sentence_model.cpu()
            del self.sentence_model
        torch.cuda.empty_cache()

def main():
    for model_name in ["UniverseTBD/AstroLLaVA_v3"]:
        evaluator = Galaxy10Evaluator(model_name=model_name, batch_size=8)
    
        custom_prompt = "Describe the following galaxy image in detail. What type of galaxy is it and what are its key features?"
        
        # Run evaluation on full test set
        results_df, metrics = evaluator.evaluate_dataset(custom_prompt=custom_prompt)
        
        # Save results
        evaluator.save_results(results_df, metrics, output_path=f'results/{model_name.split("/")[-1]}')
        
        # Print metrics
        print("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")
        evaluator.cleanup()

if __name__ == "__main__":
    main()