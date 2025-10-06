import torch
from torch.utils.data import DataLoader
from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from analyze_latent import visualize_latent_hierarchy, interpolate_styles, measure_disentanglement, controllable_generation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_ds = DrumPatternDataset('../data/drums', split='val')
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = HierarchicalDrumVAE().to(device)
model.load_state_dict(torch.load('results/best_model.pth', map_location=device))
model.eval()

visualize_latent_hierarchy(model, val_loader, device=device)
p1,_,_ = val_ds[0]; p2,_,_ = val_ds[50]
interpolate_styles(model, p1, p2, n_steps=8, device=device)
print(measure_disentanglement(model, val_loader, device=device))
controllable_generation(model, [0,1,2,3,4], device=device)
