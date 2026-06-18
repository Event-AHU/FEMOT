import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftn, ifftn



class FFTFusion(nn.Module):
    def __init__(self, resnet_dim=256, complex=True):
        super().__init__()
        self.resnet_dim = resnet_dim
        self.cat_dim = resnet_dim * 2

        self.rgb_ln = nn.LayerNorm(self.resnet_dim)
        self.event_ln = nn.LayerNorm(self.resnet_dim)
        self.rgb_event_11_conv = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_depthwise = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=3, stride=1, padding=1, groups=self.cat_dim)
        
        # 振幅和相位处理模块（代替原来的实部虚部融合）
        self.amp_processor = nn.Sequential(
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        )
        
        self.phase_processor = nn.Sequential(
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        )
        
        self.re_depthwise_11_conv = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_amp_11_conv = nn.Conv2d(self.cat_dim, self.resnet_dim, kernel_size=1)
        self.re_phase_11_conv = nn.Conv2d(self.cat_dim, self.resnet_dim, kernel_size=1)
        self.re_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.re_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)

        # rgb branch - 振幅和相位处理
        self.r_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.r_depthwise = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=3, stride=1, padding=1, groups=self.resnet_dim)
        self.r_depthwise_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.r_amp_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.r_phase_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.r_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.r_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.r_preprocessBack = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        
        # event branch - 振幅和相位处理
        self.e_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.e_depthwise = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=3, stride=1, padding=1, groups=self.resnet_dim)
        self.e_depthwise_11conv = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        self.e_amp_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.e_phase_processor = nn.Sequential(
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        )
        self.e_conv1_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.e_conv2_fft = nn.Conv2d(self.cat_dim, self.cat_dim, kernel_size=1)
        self.e_preprocessBack = nn.Conv2d(self.resnet_dim, self.resnet_dim, kernel_size=1)
        
    def _make_multiply_and_softmax(self, vis, event):
        # 用于振幅和相位特征融合的函数
        # Normalize features to avoid numerical instability
        vis = F.normalize(vis, dim=1)
        event = F.normalize(event, dim=1)

        # Flatten and multiply
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = event.view(event.size(0), event.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)

        # Apply softmax
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))

        # Residual connection
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, rgb, event):
        B, C, H, W = rgb.shape
        
        # Layer normalization
        ori_rgb = rgb.clone()
        ori_event = event.clone()
        rgb = self.rgb_ln(rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        event = self.event_ln(event.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # RGB-EVENT fusion path
        re = torch.cat([rgb, event], dim=1).contiguous()
        re = self.rgb_event_11_conv(re)
        re = self.re_depthwise(re)
        re = self.re_depthwise_11_conv(re).contiguous()
        
        # FFT Transform using rfft2
        rgb_event_freq = torch.fft.rfft2(re, norm='backward')
        
        # Extract amplitude and phase
        re_amp = torch.abs(rgb_event_freq)
        re_phase = torch.angle(rgb_event_freq)
        
        # Process amplitude and phase
        re_amp = self.amp_processor(re_amp)
        re_phase = self.phase_processor(re_phase)
        
        # Adaptive fusion of amplitude and phase
        re_amp = self._make_multiply_and_softmax(re_amp, re_amp)  # Self attention for regularization
        re_phase = self._make_multiply_and_softmax(re_phase, re_phase)  # Keep phase as is
        re_amp = self.re_amp_11_conv(re_amp)
        re_phase = self.re_phase_11_conv(re_phase)
        
        # Reconstruct complex tensor
        filter_real = re_amp * torch.cos(re_phase)
        filter_imag = re_amp * torch.sin(re_phase)
        f = torch.cat((filter_real, filter_imag), dim=1)
        f = F.relu(self.re_conv1_fft(f))
        f = self.re_conv2_fft(f).float()
        filter_real, filter_imag = torch.chunk(f, 2, dim=1)
        filter_freq = torch.complex(filter_real, filter_imag)
        
        # Inverse FFT
        enhanced_filter = torch.fft.irfft2(filter_freq, s=(H, W), norm='backward')
        
        # Convert filter to complex for multiplication with rgb/event frequency
        filter_amp = torch.abs(filter_freq)
        filter_phase = torch.angle(filter_freq)

        # RGB branch
        rgb = self.r_11conv(rgb)
        rgb = self.r_depthwise(rgb)
        rgb = self.r_depthwise_11conv(rgb).contiguous()
        
        # FFT for RGB
        rgb_freq = torch.fft.rfft2(rgb, norm='backward')
        rgb_amp = torch.abs(rgb_freq)
        rgb_phase = torch.angle(rgb_freq)
        
        # Amplitude and phase processing for RGB
        rgb_amp_processed = self.r_amp_processor(rgb_amp)
        rgb_phase_processed = self.r_phase_processor(rgb_phase)
        
        # Fusion with filter in frequency domain
        # Amplitude multiplication, phase addition (complex multiplication)
        fused_rgb_amp = self._make_multiply_and_softmax(rgb_amp_processed, filter_amp)
        fused_rgb_phase = self._make_multiply_and_softmax(rgb_phase_processed, filter_phase)
        
        # Reconstruct complex tensor
        rgb_real = fused_rgb_amp * torch.cos(fused_rgb_phase)
        rgb_imag = fused_rgb_amp * torch.sin(fused_rgb_phase)
        f = torch.cat((rgb_real, rgb_imag), dim=1)
        f = F.relu(self.r_conv1_fft(f))
        f = self.r_conv2_fft(f).float()
        rgb_real, rgb_imag = torch.chunk(f, 2, dim=1)
        rgb_freq_fused = torch.complex(rgb_real, rgb_imag)
        
        # Inverse FFT
        rgb_out = torch.fft.irfft2(rgb_freq_fused, s=(H, W), norm='backward')
        rgb_out = self.r_preprocessBack(rgb_out)
        rgb_out = rgb_out + ori_rgb
        
        # EVENT branch (similar to RGB branch)
        event_spatial = self.e_11conv(event)
        event_spatial = self.e_depthwise(event_spatial)
        event_spatial = self.e_depthwise_11conv(event_spatial).contiguous()
        
        # FFT for EVENT
        event_freq = torch.fft.rfft2(event_spatial, norm='backward')
        event_amp = torch.abs(event_freq)
        event_phase = torch.angle(event_freq)
        
        # Amplitude and phase processing for EVENT
        event_amp_processed = self.e_amp_processor(event_amp)
        event_phase_processed = self.e_phase_processor(event_phase)
        
        # Fusion with filter in frequency domain
        fused_event_amp = self._make_multiply_and_softmax(event_amp_processed, filter_amp)
        fused_event_phase = self._make_multiply_and_softmax(event_phase_processed, filter_phase)
        
        
        # Reconstruct complex tensor
        event_real = fused_event_amp * torch.cos(fused_event_phase)
        event_imag = fused_event_amp * torch.sin(fused_event_phase)
        f = torch.cat((event_real, event_imag), dim=1)
        f = F.relu(self.e_conv1_fft(f))
        f = self.e_conv2_fft(f).float()
        event_real, event_imag = torch.chunk(f, 2, dim=1)
        event_freq_fused = torch.complex(event_real, event_imag)
        
        # Inverse FFT
        event_out = torch.fft.irfft2(event_freq_fused, s=(H, W), norm='backward')
        event_out = self.e_preprocessBack(event_out)
        event_out = event_out + ori_event
        
        fused = rgb_out + event_out + enhanced_filter
        return fused
    
    def forward_srcs(self, rgb_srcs, event_srcs):
        final_scrs = []
        for rgb, event in zip(rgb_srcs, event_srcs):
            fused = self.forward(rgb, event)
            final_scrs.append(fused)
        return final_scrs

if __name__ == '__main__':
    resnet_dim = 256

    # # model = myNet(resnet_dim=resnet_dim, H=H, W=W, complex=True)
    # model = myNetV4(resnet_dim=resnet_dim)  # Max -> 2, 098, 180 2M; 仅RGB+ conv comb = 4M; 两个 conv comb = 10M
    # model = Ablation_CrossAtnn(in_dim=256, embed_dim=256)
    model = FFTFusion()
    H, W = 25, 32
    rgb = torch.randn(2, resnet_dim, H, W)
    event = torch.randn(2, resnet_dim, H, W)
    enhanced_rgb = model(rgb, event)

    output = model(rgb, event)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
