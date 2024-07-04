import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, num_channels):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, f):
        pooled = self.pool(f)
        conv1 = F.relu(self.conv1(pooled))
        conv2 = F.relu(self.conv2(conv1))
        attention = F.softmax(conv2.squeeze(-1).squeeze(-1), dim=1)
        return attention.view_as(pooled) * f

class MultiStageAttention(nn.Module):
    """
    Multi-stage Attention for Knowledge Distillation with two attention modules,
    one for student and one for teacher.
    """
    def __init__(self, s_shapes, t_shapes):
        super(MultiStageAttention, self).__init__()
        self.s_attention = nn.ModuleList([Attention(ch) for ch in s_shapes])
        self.t_attention = nn.ModuleList([Attention(ch) for ch in t_shapes])
        self.p = 2
        
    def forward(self, g_s, g_t):
        total_loss = 0.0
        for s_att, t_att, f_s, f_t in zip(self.s_attention, self.t_attention, g_s, g_t):
            if f_s.dim() == 2 and f_t.dim() == 2:  # Check if the tensors are 1D (i.e., no spatial dimensions)
                # Directly apply attention without spatial processing
                att_s = s_att(f_s.unsqueeze(-1).unsqueeze(-1))  # Fake spatial dimension to fit attention module
                att_t = t_att(f_t.unsqueeze(-1).unsqueeze(-1))
                att_s = att_s.squeeze(-1).squeeze(-1)
                att_t = att_t.squeeze(-1).squeeze(-1)
            else:
                # Handle regular 4D tensors
                s_H, s_W = f_s.size(2), f_s.size(3)
                t_H, t_W = f_t.size(2), f_t.size(3)
                if s_H != t_H or s_W != t_W:
                    target_H = min(s_H, t_H)
                    target_W = min(s_W, t_W)
                    f_s = F.adaptive_avg_pool2d(f_s, (target_H, target_W))
                    f_t = F.adaptive_avg_pool2d(f_t, (target_H, target_W))

                att_s = s_att(f_s)
                att_t = t_att(f_t)

            att_s = F.normalize(att_s.pow(self.p).mean(1).view(att_s.size(0), -1))
            att_t = F.normalize(att_t.pow(self.p).mean(1).view(att_t.size(0), -1))

            total_loss += (att_s - att_t).pow(2).mean()
        return total_loss



if __name__ == "__main__":
    # Define example feature maps for student and teacher
    f_s = [torch.randn(2, 16, 16, 16), torch.randn(2, 16, 32, 32), torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8), torch.randn(2, 64, 4, 4)]
    f_t = [torch.randn(2, 8, 32, 32), torch.randn(2, 8, 32, 32), torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8), torch.randn(2, 128, 4, 4)]

    # Automatically determine channel shapes
    s_shapes = [t.size(1) for t in f_s]
    t_shapes = [t.size(1) for t in f_t]

    # Initialize the MultiStageAttention model with determined shapes
    loss_fn = MultiStageAttention(s_shapes=s_shapes, t_shapes=t_shapes)
    loss = loss_fn(f_s, f_t)
    print("Loss:", loss.item())
