# predict.py
import torch
from PIL import Image, ImageFilter
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

        # Load a higher-quality Stable Diffusion Inpainting model
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=False
        ).to(self.device)

    def create_yard_mask(self, input_image: Image.Image) -> Image.Image:
        """
        Step 1: Use SAM to create a mask for the yard area, excluding the house or structures.

        Args:
            input_image (PIL.Image.Image): The input image to segment.

        Returns:
            PIL.Image.Image: Binary mask of the yard area.
        """
        # Convert image to numpy array for SAM
        input_image_np = np.array(input_image)

        # Use SAM to segment the image with a refined prompt
        inputs = self.sam_processor(
            input_image,
            prompt="segment the open yard area with grass, plants, trees, and pathways, exclude the house, buildings, or any man-made structures",
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        # Get the segmentation masks
        masks = outputs.pred_masks.squeeze().cpu().numpy()

        # Select the mask corresponding to the yard (excluding the house/structures)
        mask_sizes = [mask.sum() for mask in masks]
        yard_mask_idx = np.argmax(mask_sizes)  # Largest mask is likely the yard
        yard_mask = masks[yard_mask_idx]

        # Create a binary mask for the yard (1 for yard, 0 for house/structures)
        yard_mask = (yard_mask > 0).astype(np.uint8) * 255
        return Image.fromarray(yard_mask).resize(input_image.size)

    def regenerate_yard(self, input_image: Image.Image, yard_mask: Image.Image, prompt: str) -> Image.Image:
        """
        Step 2: Use Stable Diffusion Inpainting to regenerate the yard area based on the user prompt.

        Args:
            input_image (PIL.Image.Image): The original input image.
            yard_mask (PIL.Image.Image): Binary mask of the yard area.
            prompt (str): User-provided text prompt for the new landscape.

        Returns:
            PIL.Image.Image: Generated image with the new yard landscape.
        """
        # Convert image and mask to numpy arrays for inpainting
        input_image_np = np.array(input_image)
        yard_mask_np = np.array(yard_mask)

        # Use Stable Diffusion Inpainting to regenerate the landscape in the yard area
        with torch.autocast(self.device):
            generated_image = self.pipe(
                prompt=prompt,
                image=input_image_np,
                mask_image=yard_mask_np,
                strength=1.0,  # Full replacement of the yard area
                guidance_scale=9.0,  # Higher guidance for better prompt adherence
                num_inference_steps=100,  # More steps for higher quality
                eta=0.0  # No blending noise
            ).images[0]

        # Apply post-processing to enhance the output
        generated_image = generated_image.filter(ImageFilter.SHARPEN)  # Sharpen for clarity
        return generated_image

    def predict(self, image: CogPath, prompt: str = "A lush garden with colorful flower beds, a small fountain, and neatly trimmed bushes") -> CogPath:
        """
        Main prediction function: Detect the yard area with SAM and regenerate it with Stable Diffusion Inpainting.

        Args:
            image (cog.Path): Path to the input image containing a yard and a house/structure.
            prompt (str): User-provided text prompt describing the desired new landscape for the yard.

        Returns:
            cog.Path: Path to the generated image with the new landscape in the yard area.
        """
        # Load the input image
        input_image = Image.open(str(image)).convert("RGB")

        # Step 1: Create the yard mask using SAM
        yard_mask = self.create_yard_mask(input_image)

        # Step 2: Regenerate the yard area with Stable Diffusion Inpainting
        generated_image = self.regenerate_yard(input_image, yard_mask, prompt)

        # Save the generated image
        output_path = "/tmp/high_quality_yard.png"
        generated_image.save(output_path)

        return CogPath(output_path)
