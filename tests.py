from ViT.vision_transformer import VisionTransformer
import torch

model = VisionTransformer("s")

print(model)


# model.eval()

# tensor = torch.randn(size=(10, 3, 224, 224))

# print(model(tensor))