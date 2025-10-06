import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def train_gan_with_fix(generator, discriminator, data_loader,
                       num_epochs=100, fix_type='feature_matching'):
    """
    Train GAN with mode collapse mitigation techniques.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        fix_type: Stabilization method ('feature_matching', 'unrolled', 'minibatch')

    Returns:
        dict: Training history with metrics
    """
    device = next(generator.parameters()).device
    generator.train()
    discriminator.train()

    # Optimizers for main nets
    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    history = {"g_loss": [], "d_loss": []}

    # --------------------------- feature_matching ---------------------------
    if fix_type == 'feature_matching':
        # Feature matching: Match statistics of intermediate layers
        # instead of just final discriminator output
        def feature_matching_loss(real_images01, fake_images_m11, discriminator):
            """
            Extract intermediate features from discriminator
            Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
            Use discriminator.features (before final classifier)

            real_images01: [B,1,28,28] in [0,1]
            fake_images_m11: [B,1,28,28] in [-1,1]
            """
            real_m11 = real_images01 * 2 - 1
            with torch.no_grad():
                f_real = discriminator.features(real_m11)        # [B,C,H,W]
                mu_real = f_real.mean(dim=[0, 2, 3])             # [C]
            f_fake = discriminator.features(fake_images_m11)     # grad -> G
            mu_fake = f_fake.mean(dim=[0, 2, 3])                 # [C]
            return F.mse_loss(mu_fake, mu_real)

    # --------------------------- unrolled ---------------------------
    elif fix_type == 'unrolled':
        # Unrolled GANs: Look ahead k discriminator updates
        def unrolled_discriminator(discriminator, real_data, fake_data, k=5):
            """
            k-step unrolled discriminator:
            - Create temporary discriminator copy
            - Update it k times using real/fake
            - Return the updated copy for generator loss
            """
            tempD = copy.deepcopy(discriminator).to(device).train()
            for p_src, p_dst in zip(discriminator.parameters(), tempD.parameters()):
                p_dst.data.copy_(p_src.data)

            temp_opt = torch.optim.Adam(tempD.parameters(), lr=2e-4, betas=(0.5, 0.999))

            # 2) k times tempD
            ones  = torch.ones(real_data.size(0), 1, device=device)
            zeros = torch.zeros(real_data.size(0), 1, device=device)

            # real01 = real_data.detach()
            # fake01 = fake_data.detach()

            real01 = real_data.detach()
            fake01 = ((fake_data.detach() + 1) / 2).clamp(0, 1)

            for _ in range(k):
                temp_opt.zero_grad(set_to_none=True)
                real_logit = tempD(real01)
                fake_logit = tempD(fake01)

                real_prob = torch.sigmoid(real_logit)
                fake_prob = torch.sigmoid(fake_logit)

                loss_real = bce(real_prob, ones)
                loss_fake = bce(fake_prob, zeros)
                (loss_real + loss_fake).backward()
                temp_opt.step()

            tempD.eval()
            return tempD

    # --------------------------- minibatch discrimination ---------------------------
    elif fix_type == 'minibatch':
        # Minibatch discrimination: Let discriminator see batch statistics
        class MinibatchDiscrimination(nn.Module):
            """
            Compute L1 (or L2) distances in an embedding space across batch samples
            and summarize them as features to append to per-sample features.
            """
            def __init__(self, in_features, out_features=50, kernel_dims=5, p=1):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.kernel_dims = kernel_dims
                self.p = p
                self.T = nn.Linear(in_features, out_features * kernel_dims, bias=False)

            def forward(self, x):
                # x: [B, F]
                B = x.size(0)
                M = self.T(x).view(B, self.out_features, self.kernel_dims)  # [B, O, K]
                # pairwise distances
                M1 = M.unsqueeze(0)  # [1, B, O, K]
                M2 = M.unsqueeze(1)  # [B, 1, O, K]
                if self.p == 1:
                    L = torch.abs(M1 - M2).sum(dim=3)     # [B, B, O]
                else:
                    L = torch.square(M1 - M2).sum(dim=3).sqrt()
                o_b = torch.exp(-L).sum(dim=1)           # [B, O]
                return o_b

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28, device=device) * 0.0
            f = discriminator.features(dummy * 2 - 1)  # [1,C,H,W]
            feat_dim = int(f.numel())

        md = MinibatchDiscrimination(in_features=feat_dim, out_features=50, kernel_dims=5).to(device)
        aux_clf = nn.Linear(feat_dim + 50, 1).to(device)
        aux_params = list(md.parameters()) + list(aux_clf.parameters())
        aux_opt = torch.optim.Adam(aux_params, lr=2e-4, betas=(0.5, 0.999))

        aux_w = 1.0

    # ------------------------------- Training loop (shared) -------------------------------
    for epoch in range(num_epochs):
        for real_images01, _ in data_loader:
            real_images01 = real_images01.to(device)              # [B,1,28,28] in [0,1]
            B = real_images01.size(0)
            real_y = torch.ones(B, 1, device=device)
            fake_y = torch.zeros(B, 1, device=device)

            # ====== Train D ======
            generator.eval()
            discriminator.train()

            z = torch.randn(B, generator.z_dim, device=device)
            with torch.no_grad():
                fake_images_m11 = generator(z)                    # [-1,1]

            d_opt.zero_grad(set_to_none=True)
            aux_opt.zero_grad(set_to_none=True)   # 这是我加的 可能可以替换到166行
            # main branch
            d_real_logit = discriminator(real_images01)
            d_fake_logit = discriminator((fake_images_m11 + 1) / 2)
            d_real_prob = torch.sigmoid(d_real_logit)
            d_fake_prob = torch.sigmoid(d_fake_logit)
            d_loss_real = bce(d_real_prob, real_y)
            d_loss_fake = bce(d_fake_prob, fake_y)
            d_loss = d_loss_real + d_loss_fake

            if fix_type == 'minibatch':
  # 这是我加的 可能可以替换到166行
                # 通过 D.features 取特征（需要输入 [-1,1]）
                f_real = discriminator.features(real_images01 * 2 - 1).view(B, -1)
                f_fake = discriminator.features(fake_images_m11).view(B, -1)

                o_real = md(f_real)
                o_fake = md(f_fake)
                feat_real_cat = torch.cat([f_real, o_real], dim=1)
                feat_fake_cat = torch.cat([f_fake, o_fake], dim=1)

                aux_real_logit = aux_clf(feat_real_cat)
                aux_fake_logit = aux_clf(feat_fake_cat)
                aux_real_prob = torch.sigmoid(aux_real_logit)
                aux_fake_prob = torch.sigmoid(aux_fake_logit)
                aux_loss = bce(aux_real_prob, real_y) + bce(aux_fake_prob, fake_y)

                (d_loss + aux_w * aux_loss).backward()
                d_opt.step()
                aux_opt.step()
            else:

                d_loss.backward()
                d_opt.step()

            # ====== Train G ======
            generator.train()
            discriminator.eval()

            z = torch.randn(B, generator.z_dim, device=device)
            fake_images_m11 = generator(z)  # [-1,1]

            if fix_type == 'feature_matching':
                g_adv_prob = torch.sigmoid(discriminator((fake_images_m11 + 1) / 2))
                g_adv = bce(g_adv_prob, real_y)
                fm = feature_matching_loss(real_images01, fake_images_m11, discriminator)
                g_loss = g_adv + 10.0 * fm

            elif fix_type == 'unrolled':
                tempD = unrolled_discriminator(discriminator, real_images01, fake_images_m11, k=5)
                g_prob = torch.sigmoid(tempD((fake_images_m11 + 1) / 2))
                g_loss = bce(g_prob, real_y)

            elif fix_type == 'minibatch':
                g_adv_prob = torch.sigmoid(discriminator((fake_images_m11 + 1) / 2))
                g_adv = bce(g_adv_prob, real_y)

                f_fake = discriminator.features(fake_images_m11).view(B, -1)
                o_fake = md(f_fake)
                aux_fake_logit = aux_clf(torch.cat([f_fake, o_fake], dim=1))
                aux_fake_prob = torch.sigmoid(aux_fake_logit)
                aux_g = bce(aux_fake_prob, real_y)
                g_loss = g_adv + aux_w * aux_g

            else:
                # Fallback
                g_prob = torch.sigmoid(discriminator((fake_images_m11 + 1) / 2))
                g_loss = bce(g_prob, real_y)

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            history["g_loss"].append(float(g_loss.detach().cpu()))
            history["d_loss"].append(float(d_loss.detach().cpu()))

    print(history)

    return history
