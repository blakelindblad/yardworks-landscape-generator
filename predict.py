# predict.py
import torch
from PIL import Image
import numpy as np
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline
from cog import BasePredictor, Path as CogPath

class Predictor(BasePredictor):
    def setup(self):
        """Load the SAM and Stable Diffusion Inpainting models into memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load SAM for yard detection
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

        # Load Stable Diffusion Inpainting for landscape regeneration
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=False
        ).to(self.device)

    def predict(self, image: CogPath, prompt: str = "A lush garden with colorful flower beds, a small fountain, and neatly trimmed bushes") -> CogPath:
        """
        Detect the yard area (including current fixtures like trees or pathways) using SAM, exclude the house or structures,
        and regenerate a new landscape based on user input, blending it seamlessly with the existing image.

        Args:
            image (cog.Path): Path to the input image containing a yard and a house/structure.
            prompt (str): User-provided text prompt describing the desired new landscape for the yard.

        Returns:
            cog.Path: Path to the generated image with the new landscape blended into the yard area.
        """
        # Load the input image
        input_image = Image.open(str(image)).convert("RGB")
        input_image_np = np.array(input_image)

        # Use SAM to segment the image with a prompt to focus on the yard
        inputs = self.sam_processor(input_image, prompt="segment the yard area including all fixtures like trees and pathways, exclude the house or any structures", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        # Get the segmentation masks
        masks = outputs.pred_masks.squeeze().cpu().numpy()

        # Select the mask corresponding to the yard (excluding the house/structures)
        mask_sizes = [mask.sum() for mask in masks]
        yard_mask_idx = np.argmax(mask_sizes)  # Largest mask is likely the yard with fixtures
        yard_mask = masks[yard_mask_idx]

        # Create a binary mask for the yard (1 for yard and fixtures, 0 for house/structures)
        yard_mask = (yard_mask > 0).astype(np.uint8) * 255
        yard_mask = Image.fromarray(yard_mask).resize(input_image.size)
        yard_mask_np = np.array(yard_mask)  # Convert mask to numpy array for inpainting

        # Convert input image to numpy array for inpainting
        input_image_np = np.array(input_image)

        # Use Stable Diffusion Inpainting to regenerate the landscape in the yard area
        with torch.autocast(self.device):
            generated_image = self.pipe(
                prompt=prompt,
                image=input_image_np,  # Pass as numpy array
                mask_image=yard_mask_np,  # Pass as numpy array
                strength=0.6,  # Lower strength for better blending
                guidance_scale=7.5,
                num_inference_steps=50,
                eta=0.1  # Adjust eta for smoother blending
            ).images[0]

        # Save the generated image
        output_path = "/tmp/blended_yard.png"
        generated_image.save(output_path)

        return CogPath(output_path)
