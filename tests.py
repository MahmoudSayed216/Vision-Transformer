from ViT.vision_transformer import VisionTransformer
import torch

model = VisionTransformer("s")

# print(model)
for name, layer in model.named_children():
    print(name, "->", layer)

model.eval()

tensor = torch.randn(size=(10, 3, 224, 224))

print(model(tensor))