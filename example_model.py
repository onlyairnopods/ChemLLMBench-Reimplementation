import torch
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

class Model:
    def __init__(self, model_name_or_id: str, debug_mode: bool = False, **kwargs):
        self.model_name_or_id = model_name_or_id
        self.debug_mode = debug_mode
        
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)

        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=5,
            num_return_sequences=5,
            **kwargs,
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
        self.model.eval()

    def __call__(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        generated_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        response = [i[len(prompt):].strip() for i in generated_text]

        if self.debug_mode:
            print(f"Top response: \n{generated_text[0]}")

        return response