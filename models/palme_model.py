import torch
from palme import PalmE
from torchvision.utils import save_image

# dati di esempio
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = PalmE()
output = model(img, caption)

# salva il primo batch come immagine
save_image(output[0], "output.png")
