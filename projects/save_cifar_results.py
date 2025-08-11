from pathlib import Path
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "data"
OUTDIR = ROOT / "assets" / "images"; OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPNG = OUTDIR / "cifar_results.png"
CKPT = ROOT / "projects" / "cifar_cnn.pt"

classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

class SmallCIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCIFARCNN().to(device)
if CKPT.exists():
    model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
])
testset = torchvision.datasets.CIFAR10(DATADIR, train=False, download=True, transform=transform_test)
loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

images, labels = next(iter(loader))
with torch.no_grad():
    preds = model(images.to(device)).argmax(1).cpu()

# unnormalize for display
mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
std  = torch.tensor([0.247,0.243,0.261]).view(3,1,1)
vis = images * std + mean

plt.figure(figsize=(10,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(vis[i].permute(1,2,0).clamp(0,1))
    plt.title(f"P:{classes[preds[i]]} / T:{classes[labels[i]]}", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPNG, dpi=160)
print(f"Saved: {OUTPNG}")
