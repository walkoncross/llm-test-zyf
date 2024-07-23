import transformers
import torch

# model_id = "meta-llama/Meta-Llama-3.1-8B"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="mps")

outputs = pipeline("Hey how are you doing today?")
print(outputs)
print(outputs[0]["generated_text"])
