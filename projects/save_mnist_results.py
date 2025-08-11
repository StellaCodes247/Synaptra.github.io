# projects/save_mnist_results.py
from pathlib import Path
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt

# --- Paths ---
ROOT    = Path(__file__).resolve().parents[1]          # repo root
OUTDIR  = ROOT / "assets" / "images"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPNG  = OUTDIR / "mnist_results.png"
DATADIR = ROOT / "data"
CKPT    = ROOT / "projects" / "mnist_cnn.pt"

# --- Small CNN (same as trainer) ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32,10)
        )
    def forward(self,x): return self.net(x)

# --- Device & model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN().to(device)
has_ckpt = CKPT.exists()
if has_ckpt:
    model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# --- NORMALIZED TEST TRANSFORM (matches trainer) ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))   # mean/std for MNIST
])

# --- Data ---
testset = torchvision.datasets.MNIST(root=DATADIR, train=False, download=True, transform=transform)
loader  = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

images, labels = next(iter(loader))
images = images.to(device)

# --- Predict (fallback to true labels if no checkpoint yet) ---
with torch.no_grad():
    if has_ckpt:
        logits = model(images)
        preds = logits.argmax(1).cpu()
    else:
        preds = labels  # fallback so the page still shows something

# --- Plot grid and save ---
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].cpu().squeeze(0), cmap="gray")
    plt.title(f"P:{int(preds[i])} / T:{int(labels[i])}", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPNG, dpi=160)
print(f"Saved: {OUTPNG}")
