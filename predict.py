# predict.py
import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Path as CogPath

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=False
        ).to(self.device)

    def predict(self, image: CogPath, prompt: str = "A beautiful landscape with mountains and a river") -> CogPath:
        """
        Generate a landscape image based on an input image and a text prompt.

        Args:
            image (cog.Path): Path to the input image file.
            prompt (str): Text prompt to guide the landscape generation.

        Returns:
            cog.Path: Path to the generated landscape image.
        """
        # Load the input image (cog.Path can be used like a string or pathlib.Path)
        init_image = self.pipe.image_processor.preprocess(str(image))

        # Generate the landscape image
        with torch.autocast(self.device):
            output = self.pipe(
                prompt,
                init_image=init_image,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images[0]

        # Save the generated image to a temporary file
        output_path = "/tmp/generated_landscape.png"
        output.save(output_path)

        # Return the path to the generated image as a cog.Path
        return CogPath(output_path)
