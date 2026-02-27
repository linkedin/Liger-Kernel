import os

from transformers import AutoModelForCausalLM

MODEL_DIR = "/shared/public/elr-models/Qwen/Qwen3-8B/9c925d64d72725edaf899c6cb9c377fd0709d9c5"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
model.eval()

print(model)


from liger_kernel.transformers import AutoLigerKernelForCausalLM

model2 = AutoLigerKernelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
model2.eval()

print(model2)

# # Print "layers" (transformer blocks)
# layers = None
# for attr_path in [
#     ("model", "layers"),  # Qwen/LLaMA-like
#     ("model", "decoder", "layers"),
#     ("transformer", "h"),  # GPT-2-like
#     ("gpt_neox", "layers"),
# ]:
#     obj = model
#     ok = True
#     for a in attr_path:
#         if not hasattr(obj, a):
#             ok = False
#             break
#         obj = getattr(obj, a)
#     if ok:
#         layers = obj
#         container_name = ".".join(attr_path)
#         break

# if layers is None:
#     raise RuntimeError("Couldn't find a layers container on this model.")

# print(f"{container_name} count = {len(layers)}")
# for i, layer in enumerate(layers):
#     print(f"{container_name}[{i}] -> {layer.__class__.__name__}")
