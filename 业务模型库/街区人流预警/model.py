import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        

    def forward(self, x):#[B,T,N,F]
        
        x = x.permute(0, 3, 2, 1)#[B,F,N,T]
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.permute(0, 3, 2, 1)#[B,T,N,F]


class MultiAdjGraphConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_adj, bias=True,dropout=0.0):
        super().__init__()
        self.theta = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=bias)
            for _ in range(num_adj)
        ])
        self.act=nn.ReLU()
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, A_list):
        # x: (B, T, N, F)
        B, T, N, F_in = x.shape
        out = 0

        for k, A in enumerate(A_list):
            # 图聚合
            x_agg = torch.einsum('ij,btjf->btif', A, x)
            x_k = self.theta[k](x_agg)
            out = out + x_k
        out = self.act(out)
        out = self.dropout(out)


        return out



class STGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_adj,
                 kernel_size=3,
                 dropout=0.0):
        super().__init__()
        self.temp1 = TemporalConv(in_channels, hidden_channels, kernel_size, dropout)
        self.spatial = MultiAdjGraphConv(hidden_channels, hidden_channels, num_adj)
        self.temp2 = TemporalConv(hidden_channels, out_channels, kernel_size, dropout)
        self.norm = nn.LayerNorm(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x, A_sets):
        out = self.temp1(x)
        out = self.spatial(out, A_sets)
        out = self.temp2(out)
        out = self.norm(out)
        out = out.permute(0, 3, 2, 1)#[B,F,N,T]
        out = out + self.residual(x.permute(0,3,2,1))
        return out.permute(0,3,2,1)


class STGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_channels,
                 hidden_channels,
                 num_adj,
                 T_in=6,
                 num_blocks=2,
                 dropout=0.0):
        super().__init__()

        self.blocks = nn.ModuleList([
            STGCNBlock(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                hidden_channels,
                num_adj,
                dropout=dropout
            )
            for i in range(num_blocks)
        ])


        self.time_proj = nn.Conv2d(
            hidden_channels,
            1,                  # 输出 = 密度
            kernel_size=(1, T_in)
        )


    def forward(self, x, A_sets):
    # x: (B, T=6, N, F)

        for block in self.blocks:
            x = block(x, A_sets)   # (B, 6, N, hidden)

    # ===== 时间压缩 =====
        x = x.permute(0, 3, 2, 1)   # (B, hidden, N, 6)
        x = self.time_proj(x)       # (B, 1, N, 1)
        x = x.squeeze(-1)           # (B, 1, N)


        return x                    # 预测的密度

