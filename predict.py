from cog import BasePredictor, Input, Path
from PIL import Image
import io
import numpy as np
import torch
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline

class Predictor(BasePredictor):
    def setup(self):
        """Load the models during setup."""
        try:
            print("Loading SAM processor...")
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-tiny")
            print("Loading SAM model...")
            self.model = SamModel.from_pretrained("facebook/sam-vit-tiny")
            print("SAM models loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM models: {str(e)}")

        try:
            print("Loading Stable Diffusion inpainting model...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float32,
            )
            print("Moving Stable Diffusion model to GPU...")
            self.pipe.to("cuda")
            print("Stable Diffusion model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Stable Diffusion model: {str(e)}")

    def predict(
        self,
        image: Path = Input(description="Upload a house photo"),
        prompt: str = Input(description="Describe the new front yard", default="a modern lawn with colorful flowers"),
        negative_prompt: str = Input(description="What to avoid in the new front yard", default="blurry, low quality"),
    ) -> Path:
        """Run the prediction: segment the house and inpaint the yard."""
        # Load the image
        img = Image.open(image).convert("RGB")

        # Step 1: Segment the house using SAM
        mask, mask_error = self.segment_house(img)
        if mask is None:
            raise ValueError(f"Segmentation failed: {mask_error}")

        # Step 2: Inpaint the yard using Stable Diffusion
        result, inpaint_error = self.generate_front_yard(img, mask, prompt, negative_prompt)
        if result is None:
            raise ValueError(f"Inpainting failed: {inpaint_error}")

        # Save the result to a temporary file
        output_path = "/tmp/output.png"
        result.save(output_path)

        return Path(output_path)

    def segment_house(self, image):
        try:
            # Prepare image for SAM
            inputs = self.processor(image, return_tensors="pt").to("cuda")

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get the predicted segmentation masks
            predicted_masks = outputs.pred_masks  # Shape: (batch_size, num_masks, height, width)
            predicted_masks = predicted_masks[0]  # Shape: (num_masks, height, width)
            predicted_mask = predicted_masks[0]  # Shape: (height, width)
            predicted_mask = predicted_mask.cpu().numpy()

            # Convert to binary mask (white for house, black for yard)
            mask_array = (predicted_mask > 0.5).astype(np.uint8) * 255

            # Post-process: Ensure the house is white and yard is black
            height, width = mask_array.shape
            center_region = mask_array[height//4:3*height//4, width//4:3*width//4]
            if np.mean(center_region) < 128:  # If the center (house) is mostly black, invert
                mask_array = 255 - mask_array

            return Image.fromarray(mask_array), "Segmentation successful"
        except Exception as e:
            return None, f"Error in segment_house: {str(e)}"

    def generate_front_yard(self, image, mask, prompt, negative_prompt):
        try:
            if mask.mode != "L":
                mask = mask.convert("L")

            # Run inpainting
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            return result, "Inpainting successful"
        except Exception as e:
            return None, f"Error in generate_front_yard: {str(e)}"
