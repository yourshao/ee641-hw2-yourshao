import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# 直接使用作业方提供的统一评测口径
from provided.metrics import font_consistency_score, mode_coverage_score


def _ensure_dirs():
    vis_dir = Path("results/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


@torch.no_grad()
def _to01(x_m11: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return torch.clamp((x_m11 + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def _grid_and_save(imgs01: torch.Tensor, path: str, nrow: int = 10):
    """imgs01: [N,1,28,28] in [0,1]"""
    grid = make_grid(imgs01, nrow=nrow, padding=2)
    save_image(grid, path)


@torch.no_grad()
def interpolation_experiment(generator, device):
    """
    Interpolate between latent codes to generate smooth transitions.

    """
    device = torch.device(device)
    generator = generator.to(device).eval()
    vis_dir = _ensure_dirs()

    z_dim = getattr(generator, "z_dim", 100)

    steps = 12
    z_start = torch.randn(z_dim, device=device)
    z_end   = torch.randn(z_dim, device=device)
    zs = torch.stack([(1-t)*z_start + t*z_end for t in torch.linspace(0, 1, steps)], dim=0)  # [S, z_dim]
    imgs_m11 = generator(zs)         # [-1,1]
    imgs01 = _to01(imgs_m11)         # [0,1]
    _grid_and_save(imgs01, str(vis_dir / "interpolation_line.png"), nrow=steps)

    # ---- 2----
    rows, cols = 8, 12
    seqs = []
    for _ in range(rows):
        z0 = torch.randn(z_dim, device=device)
        z1 = torch.randn(z_dim, device=device)
        seq = torch.stack([(1-t)*z0 + t*z1 for t in torch.linspace(0, 1, cols)], dim=0)  # [cols, z_dim]
        seqs.append(seq)
    Z = torch.cat(seqs, dim=0)           # [rows*cols, z_dim]
    imgs = _to01(generator(Z))
    _grid_and_save(imgs, str(vis_dir / "interpolation_grid.png"), nrow=cols)

    print("[interpolation_experiment] Saved:",
          vis_dir / "interpolation_line.png",
          "and", vis_dir / "interpolation_grid.png")


@torch.no_grad()
def style_consistency_experiment(conditional_generator, device):
    """
    Test if conditional GAN maintains style across letters.
    """
    assert getattr(conditional_generator, "conditional", False), \
        "style_consistency_experiment 需要 conditional Generator(conditional=True)."

    device = torch.device(device)
    G = conditional_generator.to(device).eval()
    vis_dir = _ensure_dirs()

    z_dim = getattr(G, "z_dim", 100)
    num_classes = getattr(G, "num_classes", 26)
    K = 4

    def one_hot(c, C=num_classes, device=device):
        v = torch.zeros(C, device=device)
        v[c] = 1.0
        return v

    bucket_for_metric = {c: [] for c in range(num_classes)}

    all_rows = []
    for k in range(K):
        z = torch.randn(1, z_dim, device=device).repeat(num_classes, 1)  # [26, z_dim]
        cls = torch.stack([one_hot(c) for c in range(num_classes)], dim=0)  # [26, C]
        imgs_m11 = G(z, class_label=cls)     # [-1,1]
        imgs01 = _to01(imgs_m11)             # [26,1,28,28]
        all_rows.append(imgs01)


        for c in range(num_classes):
            bucket_for_metric[c].append(imgs01[c:c+1])  # 保持 4D

        _grid_and_save(imgs01, str(vis_dir / f"style_row_z{k}.png"), nrow=num_classes)

    if all_rows:
        big = torch.cat(all_rows, dim=0)  # [K*26, 1, 28, 28]
        _grid_and_save(big, str(vis_dir / "style_consistency_grid.png"), nrow=num_classes)

    # font_consistency_score 期望：Dict[letter_id] -> List[Tensor]（ [1,1,28,28]）
    consistency = font_consistency_score(bucket_for_metric, n_samples=10)
    print(f"[style_consistency_experiment] font_consistency_score = {consistency:.4f}")
    print("[style_consistency_experiment] Saved grid:",
          vis_dir / "style_consistency_grid.png")


@torch.no_grad()
def mode_recovery_experiment(generator_checkpoints):
    """
    Analyze how mode collapse progresses and potentially recovers.。
    """
    vis_dir = _ensure_dirs()

    if isinstance(generator_checkpoints, dict):
        items = list(generator_checkpoints.items())
    else:
        items = list(generator_checkpoints)

    if items and not isinstance(items[0], (tuple, list)):
        items = [(str(i), g) for i, g in enumerate(items)]

    coverages = []
    labels = []
    bar_paths = []

    for label, G in items:
        G = G.eval()
        device = next(G.parameters()).device
        z_dim = getattr(G, "z_dim", 100)

        N = 2000
        z = torch.randn(N, z_dim, device=device)
        imgs01 = _to01(G(z))  # [N,1,28,28] in [0,1]

        result = mode_coverage_score(imgs01.cpu(), classifier_fn=None)
        cov = float(result["coverage_score"])
        letter_counts = result["letter_counts"]  # dict: int->count

        coverages.append(cov)
        labels.append(str(label))

        xs = list(range(26))
        ys = [letter_counts.get(i, 0) for i in xs]
        plt.figure(figsize=(9, 3))
        plt.bar(xs, ys)
        plt.xticks(xs, [chr(65+i) for i in xs], rotation=0)
        plt.title(f"Letter distribution @ {label} (coverage={cov:.2f})")
        outp = vis_dir / f"mode_hist_{label}.png"
        plt.tight_layout()
        plt.savefig(outp, dpi=140)
        plt.close()
        bar_paths.append(outp)

    plt.figure(figsize=(8, 4))
    xs = np.arange(len(labels))
    plt.plot(xs, coverages, marker='o')
    plt.xticks(xs, labels, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.xlabel("Checkpoint")
    plt.ylabel("Mode Coverage (0..1)")
    plt.title("Mode coverage over checkpoints")
    out_line = vis_dir / "mode_recovery_curve.png"
    plt.tight_layout()
    plt.savefig(out_line, dpi=150)
    plt.close()

    print("[mode_recovery_experiment] Saved:",
          out_line, "and per-checkpoint histograms:", ", ".join(map(str, bar_paths)))
