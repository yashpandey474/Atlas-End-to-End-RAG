from typer import prompt

from llm.config import LLMGenerationConfig
from llm.llm import LLM
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
logger = logging.getLogger(__name__)

class HuggingFaceLLM(LLM):

    def __init__(
        self,
        model_name: str,
        device: str = "cpu"
    ):
        logger.info(f"Loading hugging face model: {model_name}")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # trust_remote_code=True
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )

        # if default padding token is not configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Successfully loaded %s", model_name)

    def generate(
        self,
        prompt: str,
        generation_config: LLMGenerationConfig
    ) -> str:

        # tokenize the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt" # output tensor objects
        )

        # put the inputs on the same device as model
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        logger.info("PROMPT LENGTH:", len(prompt))
        logger.info("PROMPT:", repr(prompt[:500]))
        logger.info("INPUT IDS SHAPE:", inputs["input_ids"].shape)
        logger.info("INPUT IDS NUMEL:", inputs["input_ids"].numel())

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                do_sample=generation_config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()