import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


MODEL_DIR = "/shared/public/elr-models/Qwen/Qwen3-8B/9c925d64d72725edaf899c6cb9c377fd0709d9c5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoLigerKernelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="bfloat16").cuda()

def hook(mod, inp, out):
    x = inp[0]
    print("MLP x:", tuple(x.shape), x.dtype, x.device, "intermediate:", mod.intermediate_size)

# print(model)

for m in model.modules():
    print("module:", m)
    print("module type:", type(m))
    print("module name:", m.__class__.__name__)
    if m.__class__.__name__ == "LigerSwiGLUMLP":
        print("LigerSwiGLUMLP found")
        m.register_forward_hook(hook)
        break
    print("--------------------------------")
    if isinstance(m, LigerSwiGLUMLP):
        m.register_forward_hook(hook)
        break

inputs = tokenizer("hello", return_tensors="pt").to("cuda")
with torch.inference_mode():
    _ = model(**inputs)          # triggers hook
    # or: _ = model.generate(**inputs, max_new_tokens=1)