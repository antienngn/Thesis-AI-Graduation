import torch, os
p = "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned"
files = [f for f in os.listdir(p) if f.endswith(".bin") or f.endswith(".safetensors")]
print("files:", files)
# Load & inspect keys
from safetensors.torch import load_file
sd = load_file(os.path.join(p, files[0])) if files[0].endswith(".safetensors") \
     else torch.load(os.path.join(p, files[0]), map_location="cpu")
print([k for k in sd.keys() if any(x in k for x in ["score", "classifier", "lm_head"])])