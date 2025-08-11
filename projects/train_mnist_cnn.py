# projects/train_mnist_cnn.py
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T

print(">>> trainer started")

ROOT    = Path(__file__).resolve().parents[1]
DATADIR = ROOT / "data"
CKPT    = ROOT / "projects" / "mnist_cnn.pt"

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32,10)
        )
    def forward(self,x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model = SmallCNN().to(device)

# ✅ NORMALIZED INPUTS (this is “number 1”)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))  # mean, std for MNIST
])

trainset = torchvision.datasets.MNIST(root=DATADIR, train=True, download=True, transform=transform)
loader   = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

opt = optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()

# Train longer so accuracy climbs
EPOCHS = 8
model.train()
for epoch in range(EPOCHS):
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward(); opt.step()
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
    print(f"Epoch {epoch+1}: acc={correct/total:.3f}")

CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), CKPT)
print(f"Saved checkpoint: {CKPT}")
