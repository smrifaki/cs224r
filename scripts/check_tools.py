import gymnasium
import modal
import timm
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("timm", timm.__version__)
print("gymnasium", gymnasium.__version__)
print("modal", modal.__version__)

m = timm.create_model("vit_small_patch16_224", pretrained=True)
print("vit params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
