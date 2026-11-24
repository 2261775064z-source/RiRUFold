import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RiRUFold', 'RiRFold']


def image_penalty(image):
    """Per-image initial penalty factor: 5 * std(vec(X))."""
    return 5 * image.flatten(start_dim=1).std(dim=1).view(-1, 1, 1, 1)


class RiRUFold(nn.Module):
    """Sparse Reweighted Mask Tensor Unfolding Network (RiRUFold) for ISTD."""

    def __init__(self, stage_num=5, slayers=6, llayers=3, mlayers=3, channel=32, mode='train'):
        super(RiRUFold, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(
                DecompositionModule(slayers=slayers, llayers=llayers, mlayers=mlayers, channel=channel)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros_like(D)
        y1 = torch.zeros_like(D)
        y2 = torch.zeros_like(D)
        y3 = torch.zeros_like(D)
        W = torch.zeros_like(D)
        mu = image_penalty(D)

        for i in range(self.stage_num):
            D, T, y1, y2, y3, W = self.decos[i](D, T, y1, y2, y3, W, mu)
        if self.mode == 'train':
            return D, T
        return T


class DecompositionModule(nn.Module):
    """Single unfolded stage integrating LR_G, SR_G, DV_G, and RW_G."""

    def __init__(self, slayers=6, llayers=3, mlayers=3, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(base_ch=channel, num_blocks=llayers)
        self.sparse = SparseModule(base_ch=channel, num_experts=4)
        self.stepgate = AdaptiveStepGate(channel=1)
        self.weight = WeightUpdateNetwork(in_channels=2, hidden_dim=16)

    def forward(self, D, T, y1, y2, y3, W, mu):
        B1, B2, B3 = self.lowrank(D, mu, y1, y2, y3, T)
        self.last_B1 = B1
        T = self.sparse(D, mu, y1, y2, y3, B1, B2, B3, T, W)
        y1, y2, y3 = self.stepgate(D, T, y1, y2, y3, B1, B2, B3)
        W = self.weight(D, T)
        return D, T, y1, y2, y3, W


class LowrankModule(nn.Module):
    """Low-Rank Separation Module (LR_G): Shared Encoder (SE) + Branch Adapters (BA)."""

    def __init__(self, base_ch=32, num_blocks=3):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            MultiScaleFeatureExtractor(base_ch),
            *[ResidualSEBlock(base_ch) for _ in range(num_blocks)]
        )
        self.branch_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_ch, 1, 1),
                SpatialAttention()
            ) for _ in range(3)
        ])

    def process_branch(self, x, branch_adapter):
        shared_feat = self.shared_encoder(x)
        branch_feat = branch_adapter(shared_feat)
        return x + branch_feat

    def forward(self, D, mu, y1, y2, y3, T):
        B1 = self.process_branch(D + mu * y1 - T, self.branch_adapters[0])
        B2 = self.process_branch(D + mu * y2 - T, self.branch_adapters[1])
        B3 = self.process_branch(D + mu * y3 - T, self.branch_adapters[2])
        return B1, B2, B3


class SparseModule(nn.Module):
    """Sparse Target Extraction Module (SR_G): Gating Weight Generator + Dynamic Expert Conv."""

    def __init__(self, base_ch=32, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.base_ch = base_ch

        self.input_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, num_experts * base_ch * base_ch * 3 * 3)
        )
        self.expert_weights = nn.Parameter(torch.ones(num_experts))
        self.fusion = nn.Sequential(
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.epsilon = nn.Parameter(torch.tensor(0.01))

    def dynamic_conv(self, x):
        B, C, H, W = x.shape
        dynamic_weights = self.weight_generator(x)
        kernels = dynamic_weights.view(B, self.num_experts, C, C, 3, 3)
        combined_kernel = torch.einsum(
            'beoihw,e->boihw',
            kernels,
            torch.softmax(self.expert_weights, dim=0)
        )
        combined_kernel = combined_kernel.reshape(B * C, C, 3, 3)
        x_reshaped = x.view(1, B * C, H, W)
        conv_feat = F.conv2d(x_reshaped, combined_kernel, padding=1, groups=B)
        return conv_feat.view(B, C, H, W)

    def forward(self, D, mu, y1, y2, y3, B1, B2, B3, T, W):
        x = self.input_conv(W * T)
        conv_feat = self.dynamic_conv(x)
        attn = self.fusion(conv_feat)
        T_update = D + mu * (y1 + y2 + y3) / 3 - (B1 + B2 + B3) / 3
        T = T + T_update - self.epsilon * attn * T
        return T


class AdaptiveStepGate(nn.Module):
    """Dual Variable Update Module (DV_G) enforcing residual decomposition consistency."""

    def __init__(self, channel=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, F_obs, T, y1, y2, y3, B1, B2, B3):
        residual1 = F_obs - B1 - T
        alpha1 = self.conv(residual1)
        y1 = y1 + alpha1 * residual1

        residual2 = F_obs - B2 - T
        alpha2 = self.conv(residual2)
        y2 = y2 + alpha2 * residual2

        residual3 = F_obs - B3 - T
        alpha3 = self.conv(residual3)
        y3 = y3 + alpha3 * residual3
        return y1, y2, y3


class WeightUpdateNetwork(nn.Module):
    """Weight Generation Module (RW_G) coupling local structure W_LS and sparse W_SE."""

    def __init__(self, in_channels=2, hidden_dim=16):
        super().__init__()
        self.local_struct_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.sparsity_net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, F_obs, T):
        with torch.no_grad():
            grad_x = F_obs[:, :, 1:, :] - F_obs[:, :, :-1, :]
            grad_y = F_obs[:, :, :, 1:] - F_obs[:, :, :, :-1]
            grad_x = F.pad(grad_x, (0, 0, 0, 1), mode='replicate')
            grad_y = F.pad(grad_y, (0, 1, 0, 0), mode='replicate')
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        struct_input = torch.cat([F_obs, grad_mag], dim=1)
        W_local = self.local_struct_net(struct_input)

        T_abs = torch.abs(T)
        W_sparse = 1.0 / (self.sparsity_net(T_abs) + 1e-6)
        return W_local * W_sparse


class ResidualSEBlock(nn.Module):
    """Residual Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x * self.se(x)
        return residual + x


class MultiScaleFeatureExtractor(nn.Module):
    """Parallel multi-scale feature extraction (1, 1/2, 1/4) with bilinear alignment."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        _, _, H, W = x1.shape
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(H, W), mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([x1, x2, x3], dim=1))


class SpatialAttention(nn.Module):
    """Channel-pooled spatial attention module."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.conv(combined)


RiRFold = RiRUFold
