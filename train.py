import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List


class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has a learnable gate.
    The gate controls whether the weight stays active or gets pruned.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.constant_(self.gate_scores, 1.0)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def get_sparsity(self, threshold: float = 0.05) -> float:
        gates = self.get_gates()
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return 100.0 * pruned / total if total > 0 else 0.0


class SelfPruningNetwork(nn.Module):
    """
    Feed-forward network for CIFAR-10 using prunable linear layers.
    """

    def __init__(self):
        super().__init__()

        self.prunable_layers: List[PrunableLinear] = []

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.prunable_layers.append(self.fc1)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.prunable_layers.append(self.fc2)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.prunable_layers.append(self.fc3)

        self.fc4 = PrunableLinear(128, 10)
        self.prunable_layers.append(self.fc4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers:
            loss = loss + layer.get_gates().mean()
        return loss

    def get_total_sparsity(self, threshold: float = 0.05) -> float:
        total_pruned = 0
        total_weights = 0

        for layer in self.prunable_layers:
            gates = layer.get_gates()
            total_pruned += (gates < threshold).sum().item()
            total_weights += gates.numel()

        return 100.0 * total_pruned / total_weights if total_weights > 0 else 0.0

    def get_all_gates(self) -> torch.Tensor:
        all_gates = []
        for layer in self.prunable_layers:
            all_gates.append(layer.get_gates().detach().cpu().flatten())
        return torch.cat(all_gates)


def get_optimizer(model: SelfPruningNetwork, lr: float = 1e-3):
    gate_params = [layer.gate_scores for layer in model.prunable_layers]
    other_params = []

    for layer in model.prunable_layers:
        other_params.append(layer.weight)
        if layer.bias is not None:
            other_params.append(layer.bias)

    return optim.Adam(
        [
            {"params": gate_params, "lr": lr * 10},
            {"params": other_params, "lr": lr},
        ]
    )


def get_data_loaders(batch_size: int = 128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device, lambda_sparse):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        cls_loss = F.cross_entropy(out, y)
        sparse_loss = model.get_sparsity_loss()
        loss = cls_loss + lambda_sparse * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def run_lambda_experiments():
    lambdas = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    for i, lambda_sparse in enumerate(lambdas):
        print(f"\n--- Experiment {i+1}/5: λ={lambda_sparse} ---")

        train_loader, test_loader = get_data_loaders(batch_size=128)
        model = SelfPruningNetwork().to(device)
        optimizer = get_optimizer(model, lr=3e-4)

        for epoch in range(20):
            loss = train_epoch(model, train_loader, optimizer, device, lambda_sparse)

        final_acc = evaluate(model, test_loader, device)
        final_sparsity = model.get_total_sparsity()
        final_gates = model.get_all_gates()

        print(f"λ={lambda_sparse:.0e}: Acc={final_acc:.1f}% | Sparsity={final_sparsity:.1f}%")

        results.append({
            "lambda": lambda_sparse,
            "acc": final_acc,
            "sparsity": final_sparsity,
            "gates": final_gates.numpy(),
            "model": model
        })

    return results


def plot_results(results):
    best_idx = max(range(len(results)), key=lambda i: results[i]["acc"])
    best = results[best_idx]

    print("\nRESULTS SUMMARY")
    print("| Lambda | Test Accuracy | Sparsity Level (%) |")
    print("|---|---:|---:|")
    for r in results:
        print(f"| {r['lambda']:.0e} | {r['acc']:.1f}% | {r['sparsity']:.1f}% |")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    gates = best["gates"]
    plt.hist(gates, bins=100, density=True, alpha=0.8, color="green", edgecolor="black")
    plt.axvline(0.05, color="red", linestyle="--", label="Threshold = 0.05")
    plt.xlabel("Gate Value")
    plt.ylabel("Density")
    plt.title(f"Best Model Gate Distribution\nλ={best['lambda']:.0e}")
    plt.legend()

    plt.subplot(1, 3, 2)
    sparsities = [r["sparsity"] for r in results]
    accs = [r["acc"] for r in results]
    plt.scatter(sparsities, accs, s=120, color="blue", edgecolors="black")
    for r in results:
        plt.annotate(f"λ={r['lambda']:.0e}", (r["sparsity"], r["acc"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Sparsity")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    log_lambdas = np.log10([r["lambda"] for r in results])
    plt.plot(log_lambdas, accs, "o-", label="Accuracy")
    plt.plot(log_lambdas, sparsities, "s-", label="Sparsity")
    plt.xlabel("log10(Lambda)")
    plt.ylabel("Value")
    plt.title("Effect of Lambda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("self_pruning_report.png", dpi=300, bbox_inches="tight")
    plt.show()

    return best


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    results = run_lambda_experiments()
    best = plot_results(results)

    print(f"\nBest Lambda: {best['lambda']:.0e}")
    print(f"Best Accuracy: {best['acc']:.1f}%")
    print(f"Best Sparsity: {best['sparsity']:.1f}%")
    print("Saved plot: self_pruning_report.png")


if __name__ == "__main__":
    main()
