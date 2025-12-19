class DynamicInceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=in_channels)
        ])
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        
        # Dynamic Kernel Weights
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )
        
    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))

class MSDIM(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(min_ch, ks, ks * 3 + 2))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        x = self.conv_1x1(x_group)
        return x

class DIDC(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = MSDIM(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class C3k_MSDIM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DIDC(c_) for _ in range(n)))

class C3k2_MSDIM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSDIM(self.c, self.c, 2, shortcut, g) if c3k else DIDC(self.c) for _ in range(n))