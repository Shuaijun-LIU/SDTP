import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper:
    """
    Unified wrapper for Baseline & SDTP model.

    - setup 阶段：只打印信息（real_load=False）

    - 推理阶段：加载 HuggingFace 模型并执行 generate

    """

    def __init__(
        self,
        model_name: str,
        pruning_module_path: str = None,
        mode: str = "baseline",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device: str = None,
    ):
        self.model_name = model_name
        self.pruning_module_path = pruning_module_path
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, real_load: bool = False):
        print(f"[Init] Preparing model loading: {self.model_name}")
        if self.pruning_module_path:
            print(f"[Init] Pruning module: {self.pruning_module_path}")
        print(f"[Init] Mode: {self.mode}")

        if not real_load:
            print("[Init] Model loading is disabled in setup stage.")
            return

        print("[Init] >>> Real model loading START <<<")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Init] >>> Real model loading DONE <<<")

    def infer(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model(real_load=True) before infer()."
            )

        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]

        out = self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return out.strip()
