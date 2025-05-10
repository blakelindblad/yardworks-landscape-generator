import torch
import requests
from pathlib import Path
from PIL import Image
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusionInpaintPipeline

class Predictor:
    def setup(self):
        try:
            print("Loading SAM processor...")
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            print("Loading SAM model...")
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
            print("SAM models loaded successfully")

            # Move SAM model to CPU to free GPU memory
            self.sam_model.to("cpu")
            torch.cuda.empty_cache()

            print("Loading Stable Diffusion inpainting model...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                use_auth_token=False,
            )
            print("Moving Stable Diffusion model to GPU...")
            self.pipe = self.pipe.to("cuda")
            print("Stable Diffusion model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def segment_yard(self, image):
        # Move SAM model back to GPU for inference
        self.sam_model.to("cuda")
        # Simplified segmentation (replace with your actual logic)
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        outputs = self.sam_model(**inputs)
        mask = outputs.pred_masks  # Process mask as needed
        # Move SAM model back to CPU and clear memory
        self.sam_model.to("cpu")
        torch.cuda.empty_cache()
        return mask

    def predict(self, image: Path, prompt: str, negative_prompt: str) -> str:
        """
        Redesign the front yard of a house in the input image based on the prompt.

        Args:
            image (Path): An image of a house with a front yard to redesign (upload an image file or provide a URL).
            prompt (str): A description of the desired front yard design (e.g., "a modern lawn with colorful flowers").
            negative_prompt (str): What to avoid in the design (e.g., "blurry, low quality").

        Returns:
            str: Path to the generated image with the redesigned front yard.
        """
        # Load image
        # If image is a URL (from API) or a local path (from UI upload), handle accordingly
        if str(image).startswith(("http://", "https://")):
            input_image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
        else:
            input_image = Image.open(image).convert("RGB")
        input_image = input_image.resize((512, 512))

        # Segment yard
        print("Segmenting yard with SAM...")
        mask = self.segment_yard(input_image)
        print("Yard segmented successfully")

        # Generate new yard
        print("Generating new yard with Stable Diffusion...")
        output_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        print("New yard generated successfully")

        # Save output
        output_path = "output.png"
        output_image.save(output_path)
        return output_path
