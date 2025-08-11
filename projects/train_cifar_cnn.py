from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "data"
CKPT = ROOT / "projects" / "cifar_cnn.pt"

class SmallCIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),      # 32x16x16
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),     # 64x8x8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCIFARCNN().to(device)
opt = optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()

transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
])
trainset = torchvision.datasets.CIFAR10(DATADIR, train=True, download=True, transform=transform_train)
loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

EPOCHS = 8
model.train()
for e in range(EPOCHS):
    total=correct=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward(); opt.step()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    print(f"Epoch {e+1}: acc={correct/total:.3f}")

CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), CKPT)
print(f"Saved checkpoint: {CKPT}")
