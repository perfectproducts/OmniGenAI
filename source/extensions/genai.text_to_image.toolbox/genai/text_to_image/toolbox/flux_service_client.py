import requests
import json
from typing import Optional


class FluxServiceClient:
    """Client for interacting with the Flux service API"""

    def __init__(self, host: str = "localhost", port: int = 8011):
        """Initialize the client with host and port

        Args:
            host (str): Hostname of the service
            port (int): Port number of the service
        """
        self.base_url = f"http://{host}:{port}"

    def get_status(self) -> dict:
        """Get current service status

        Returns:
            dict: Status response
        """
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

    def get_health(self) -> dict:
        """Get service health status

        Returns:
            dict: Health status response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_ready(self) -> dict:
        """Get service readiness status

        Returns:
            dict: Readiness status response
        """
        response = requests.get(f"{self.base_url}/ready")
        response.raise_for_status()
        return response.json()

    def get_startup(self) -> dict:
        """Get service startup status

        Returns:
            dict: Startup status response
        """
        response = requests.get(f"{self.base_url}/startup")
        response.raise_for_status()
        return response.json()

    def get_asyncapi_docs(self, app_name: str) -> dict:
        """Get async API documentation

        Args:
            app_name (str): Name of the app

        Returns:
            dict: API documentation
        """
        response = requests.get(f"{self.base_url}/asyncapi/docs", params={"app_name": app_name})
        response.raise_for_status()
        return response.json()

    def get_asyncapi_schema(self, app_name: str) -> dict:
        """Get async API schema

        Args:
            app_name (str): Name of the app

        Returns:
            dict: API schema
        """
        response = requests.get(f"{self.base_url}/asyncapi/schema", params={"app_name": app_name})
        response.raise_for_status()
        return response.json()

    def generate_image(self,
                      prompt: str = "a beautiful image of a cat",
                      height: int = 1024,
                      width: int = 1024,
                      inference_steps: int = 8,
                      guidance_scale: float = 3.5,
                      seed: int = 0) -> bytes:
        """Generate an image from text prompt

        Args:
            prompt (str): Text prompt for image generation
            height (int): Height of generated image
            width (int): Width of generated image
            inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale value
            seed (Optional[int]): Random seed for generation

        Returns:
            bytes: Generated image data
        """
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }

        response = requests.post(f"{self.base_url}/text_to_image", json=payload)
        # dump the request and response to the console
        print(f"Request: {payload}")
        print(f"Response: {response.content}")
        response.raise_for_status()
        return response.content
