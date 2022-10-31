"""Hugging Face client."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

from manifest.clients.client import Client

logger = logging.getLogger(__name__)

# User param -> (client param, default value)
DIFFUSER_PARAMS = {
    "num_inference_steps": ("num_inference_steps", 50),
    "height": ("height", 512),
    "width": ("width", 512),
    "num_images_per_prompt": ("num_images_per_prompt", 1),
    "guidance_scale": ("guidance_scale", 7.5),
    "eta": ("eta", 0.0),
}


class DiffuserClient(Client):
    """Diffuser client."""

    def connect(
        self,
        connection_str: Optional[str] = None,
        client_args: Dict[str, Any] = {},
    ) -> None:
        """
        Connect to the Diffuser url.

        Arsg:
            connection_str: connection string.
            client_args: client arguments.
        """
        self.host = connection_str.rstrip("/")
        for key in DIFFUSER_PARAMS:
            setattr(self, key, client_args.pop(key, DIFFUSER_PARAMS[key][1]))
        self.model_params = self.get_model_params()

    def close(self) -> None:
        """Close the client."""
        pass

    def get_model_params(self) -> Dict:
        """
        Get model params.

        By getting model params from the server, we can add to request
        and make sure cache keys are unique to model.

        Returns:
            model params.
        """
        res = requests.post(self.host + "/params")
        return res.json()

    def get_model_inputs(self) -> List:
        """
        Get allowable model inputs.

        Returns:
            model inputs.
        """
        return list(DIFFUSER_PARAMS.keys())

    def get_request(
        self, query: str, request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function.

        Args:
            query: query string.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        request_params = {"prompt": query}
        for key in DIFFUSER_PARAMS:
            request_params[DIFFUSER_PARAMS[key][0]] = request_args.pop(
                key, getattr(self, key)
            )
        request_params.update(self.model_params)

        def _run_completion() -> Dict:
            post_str = self.host + "/completions"
            res = requests.post(post_str, json=request_params)
            result = res.json()
            # Convert array to np.array
            for choice in result["choices"]:
                choice["array"] = np.array(choice["array"])
            return result

        return _run_completion, request_params

    def get_choice_logit_request(
        self, query: str, gold_choices: List[str], request_args: Dict[str, Any] = {}
    ) -> Tuple[Callable[[], Dict], Dict]:
        """
        Get request string function for choosing max choices.

        Args:
            query: query string.
            gold_choices: choices for model to choose from via max logits.

        Returns:
            request function that takes no input.
            request parameters as dict.
        """
        raise NotImplementedError("Diffusers does not support choice logit request.")
