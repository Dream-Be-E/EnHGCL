
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class FeatureTransform(nn.Module):
    """Project node features."""
    def __init__(self, in_dim: int, out_dim: int, node_type: str):
        super(FeatureTransform, self).__init__()
        self.node_type = node_type
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

class MultiHopDiffusionAttentionGNN(nn.Module):

    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.3, 
                 num_hops: int = 3, alpha: float = 0.5):
    
        super(MultiHopDiffusionAttentionGNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_hops = num_hops
        self.alpha = alpha
        theta = torch.tensor(
            [alpha * ((1 - alpha) ** k) for k in range(num_hops + 1)],
            dtype=torch.float32,
        )
        theta = theta / theta.sum().clamp(min=1e-12)
        self.register_buffer("diffusion_theta", theta)
        assert out_dim % num_heads == 0, "out_dim必须能被num_heads整除"
        
        
        # 为每个头创建独立的参数
        self.W_attn = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(num_heads)
        ])  # W^(l) for each head
        
        self.u_attn = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.head_dim * 2) * 0.01) for _ in range(num_heads)
        ])  
        

        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.input_layer_norm = nn.LayerNorm(in_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def compute_edge_attention(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
 
        num_nodes = features.size(0)
        
        # 计算每个头的注意力分数
        Q_0_list = []
        for h in range(self.num_heads):
            # 1. 投影特征: W^(l) Z_i^(l)
            W_h = self.W_attn[h]  # [in_dim, head_dim]
            Z_proj = W_h(features)  # [N+M, head_dim]
            
      
            # 使用广播: [N+M, 1, head_dim] 和 [1, N+M, head_dim]
            Z_i = Z_proj.unsqueeze(1).expand(num_nodes, num_nodes, self.head_dim)  # [N+M, N+M, head_dim]
            Z_j = Z_proj.unsqueeze(0).expand(num_nodes, num_nodes, self.head_dim)  # [N+M, N+M, head_dim]
            
            # 拼接: [N+M, N+M, head_dim * 2]
            concat_features = torch.cat([Z_i, Z_j], dim=-1)
            
          
            concat_features = torch.tanh(concat_features)  # [N+M, N+M, head_dim * 2]
            
         
            u_h = self.u_attn[h]  # [1, head_dim * 2]
            
    
            # [N+M, N+M, head_dim * 2] * [1, 1, head_dim * 2] -> [N+M, N+M]
            attn_scores = (concat_features * u_h.unsqueeze(0)).sum(dim=-1)  # [N+M, N+M]
            
            # 5. 应用 LeakyReLU
            attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)  # [N+M, N+M]
            
            Q_0_list.append(attn_scores)
        
        # 堆叠所有头的注意力 [N+M, N+M, num_heads]
        Q_0 = torch.stack(Q_0_list, dim=-1)
        
        # 应用邻接矩阵mask（只保留有边的位置）
        edge_weight = adj.clamp(min=0.0)
        mask = edge_weight <= 0
        log_weight_bias = torch.log(edge_weight.clamp(min=1e-6)).unsqueeze(-1)
        Q_0 = Q_0 + log_weight_bias
        Q_0 = Q_0.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        # 行归一化（softmax）
        Q_0 = F.softmax(Q_0, dim=1)  # [N+M, N+M, num_heads]
        # 安全保护：防止零度节点导致NaN（全-inf行softmax后为NaN）
        Q_0 = Q_0.nan_to_num(0.0)
        
        return Q_0
    
    def diffusion_attention(self, Q_0: torch.Tensor) -> torch.Tensor:

        num_nodes = Q_0.size(0)
        device = Q_0.device
        
        # 初始化扩散注意力矩阵
        Q_diffused = torch.zeros_like(Q_0)
        
        # 计算注意力衰减权重 θ_k = α(1-α)^k
        theta = self.diffusion_theta.to(device=device, dtype=Q_0.dtype)
        
        # 归一化θ（确保和为1）
        
        # 迭代计算 (Q^(0))^k
        Q_power = torch.eye(num_nodes, device=device, dtype=Q_0.dtype).unsqueeze(-1).expand(-1, -1, self.num_heads)  # Q^0 = I
        
        for k in range(self.num_hops + 1):
            # Q_diffused += θ_k * Q^k
            Q_diffused += theta[k] * Q_power
            
            # 计算下一个幂次：Q^(k+1) = Q^k @ Q^(0)
            if k < self.num_hops:
                # 对每个头分别计算矩阵乘法
                Q_power = torch.bmm(
                    Q_power.permute(2, 0, 1).contiguous(),
                    Q_0.permute(2, 0, 1).contiguous(),
                ).permute(1, 2, 0)
                # 安全保护：钳制极端值防止数值溢出
                Q_power = Q_power.clamp(-1e6, 1e6)
        Q_diffused = Q_diffused.clamp(min=0.0)
        Q_diffused = Q_diffused / Q_diffused.sum(dim=1, keepdim=True).clamp(min=1e-12)
        
        return Q_diffused
    
    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        num_nodes = features.size(0)
        
        # 构建完整的邻接矩阵 [N+M, N+M]（用于二部图）
        N, M = adj.size()
        full_adj = torch.zeros(
            num_nodes,
            num_nodes,
            device=adj.device,
            dtype=adj.dtype,
        )
        full_adj[:N, N:] = adj  # 上半部分：N->M
        full_adj[N:, :N] = adj.t()  # 下半部分：M->N

        # 1. 计算1-hop边注意力 Q^(0)
        Q_0 = self.compute_edge_attention(features, full_adj)  # [N+M, N+M, num_heads]
        
        # 2. 通过扩散得到多跳注意力 Q^(h)
        Q_diffused = self.diffusion_attention(Q_0)  # [N+M, N+M, num_heads]
        
        # 3. 使用扩散注意力聚合邻居信息
        # 投影特征到多头空间
        normalized_features = self.input_layer_norm(features)
        v_proj = self.W_v(normalized_features).view(num_nodes, self.num_heads, self.head_dim)  # [N+M, num_heads, head_dim]
        
        # 对每个头进行聚合
        aggregated_list = []
        for h in range(self.num_heads):
            # 使用扩散注意力聚合
            aggregated_h = torch.matmul(Q_diffused[:, :, h], v_proj[:, h, :])  # [N+M, head_dim]
            aggregated_list.append(aggregated_h)
        
        # 拼接所有头
        aggregated = torch.cat(aggregated_list, dim=-1)  # [N+M, out_dim]
        
        # 输出投影
        output = self.W_o(aggregated)  # [N+M, out_dim]
        
        # Eq. 20 uses the diffused attention over LN(H), without an extra self-loop residual.
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output

class PathEmbeddingFusion(nn.Module):
    
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: 嵌入维度
        """
        super(PathEmbeddingFusion, self).__init__()
        self.embed_dim = embed_dim
        # PDF Eq. 22-24: path score from average cosine similarity, then softmax.

    def _path_informativeness(
        self,
        normalized_embed: torch.Tensor,
        hetero_ref_embed: torch.Tensor
    ) -> torch.Tensor:
        return self._average_cosine_score(normalized_embed, hetero_ref_embed)
        
    def _average_cosine_score(
        self,
        embed: torch.Tensor,
        hetero_ref_embed: torch.Tensor
    ) -> torch.Tensor:
        if hetero_ref_embed is None or hetero_ref_embed.numel() == 0:
            raise ValueError("PathEmbeddingFusion requires a non-empty hetero_ref_embed.")
        embed_norm = F.normalize(embed, p=2, dim=1)
        ref_norm = F.normalize(hetero_ref_embed, p=2, dim=1)
        return torch.mm(embed_norm, ref_norm.t()).mean(dim=1)

    def forward(
        self,
        embeddings: List[torch.Tensor],
        hetero_refs: List[torch.Tensor]
    ) -> torch.Tensor:
  
        if len(embeddings) == 0:
            raise ValueError("PathEmbeddingFusion requires at least one path embedding.")
        if hetero_refs is None or len(hetero_refs) != len(embeddings):
            raise ValueError("PathEmbeddingFusion requires one hetero reference per path embedding.")

        scores = []
        for path_idx, embed in enumerate(embeddings):
            scores.append(self._average_cosine_score(embed, hetero_refs[path_idx]).unsqueeze(1))

        path_scores = torch.cat(scores, dim=1)
        weights = F.softmax(path_scores, dim=1)

        fused = torch.zeros_like(embeddings[0])
        for path_idx, embed in enumerate(embeddings):
            fused = fused + weights[:, path_idx:path_idx + 1] * embed

        return fused

class ThreeLayerSemanticAttentionNetwork(nn.Module):
    def __init__(
        self,
        mirna_dim: int,
        lncrna_dim: int,
        disease_dim: int,
        unified_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_hops: int = 2,
        alpha: float = 0.7,
        use_feature_transform: bool = True,
    ):
        super().__init__()
        del unified_dim

        self.num_gnn_layers = int(num_gnn_layers)
        if self.num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1.")

        self.use_feature_transform = use_feature_transform
        if use_feature_transform:
            self.mirna_transform = FeatureTransform(mirna_dim, hidden_dim, 'miRNA')
            self.lncrna_transform = FeatureTransform(lncrna_dim, hidden_dim, 'lncRNA')
            self.disease_transform = FeatureTransform(disease_dim, hidden_dim, 'disease')
        else:
            self.mirna_transform = nn.Linear(mirna_dim, hidden_dim)
            self.lncrna_transform = nn.Linear(lncrna_dim, hidden_dim)
            self.disease_transform = nn.Linear(disease_dim, hidden_dim)

        self.gnn_md_layers = self._build_gnn_stack(
            hidden_dim, out_dim, num_heads, dropout, num_hops, alpha
        )
        self.gnn_ml_layers = self._build_gnn_stack(
            hidden_dim, out_dim, num_heads, dropout, num_hops, alpha
        )
        self.gnn_ld_layers = self._build_gnn_stack(
            hidden_dim, out_dim, num_heads, dropout, num_hops, alpha
        )

        self.mirna_fusion = PathEmbeddingFusion(out_dim)
        self.lncrna_fusion = PathEmbeddingFusion(out_dim)
        self.disease_fusion = PathEmbeddingFusion(out_dim)
        self.mirna_skip_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.lncrna_skip_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.disease_skip_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.mirna_out_norm = nn.LayerNorm(out_dim)
        self.lncrna_out_norm = nn.LayerNorm(out_dim)
        self.disease_out_norm = nn.LayerNorm(out_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.gnn_inter_dropout = nn.Dropout(dropout)
        self.residual_ratio = 0.0

    def _build_gnn_stack(
        self,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        num_hops: int,
        alpha: float,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        for layer_idx in range(self.num_gnn_layers):
            layer_out_dim = out_dim if layer_idx == self.num_gnn_layers - 1 else hidden_dim
            layers.append(
                MultiHopDiffusionAttentionGNN(
                    hidden_dim, layer_out_dim, num_heads, dropout, num_hops, alpha
                )
            )
        return layers

    def _run_gnn_stack(
        self,
        layers: nn.ModuleList,
        adj: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        h = features
        for layer_idx, layer in enumerate(layers):
            h = layer(adj, h)
            if layer_idx < len(layers) - 1:
                h = self.gnn_inter_dropout(F.relu(h))
        return h

    def forward(
        self,
        mirna_features: torch.Tensor,
        lncrna_features: torch.Tensor,
        disease_features: torch.Tensor,
        mirna_disease_adj: torch.Tensor,
        mirna_lncrna_adj: torch.Tensor,
        lncrna_disease_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_mirna = mirna_features.size(0)
        num_lncrna = lncrna_features.size(0)
        num_disease = disease_features.size(0)

        if self.use_feature_transform:
            mirna_h = self.mirna_transform(mirna_features)
            lncrna_h = self.lncrna_transform(lncrna_features)
            disease_h = self.disease_transform(disease_features)
        else:
            mirna_h = self.mirna_transform(mirna_features)
            lncrna_h = self.lncrna_transform(lncrna_features)
            disease_h = self.disease_transform(disease_features)

        md_features = torch.cat([mirna_h, disease_h], dim=0)
        md_h = self._run_gnn_stack(self.gnn_md_layers, mirna_disease_adj, md_features)
        mirna_md_embed = md_h[:num_mirna]
        disease_md_embed = md_h[num_mirna:]

        ml_features = torch.cat([mirna_h, lncrna_h], dim=0)
        ml_h = self._run_gnn_stack(self.gnn_ml_layers, mirna_lncrna_adj, ml_features)
        mirna_ml_embed = ml_h[:num_mirna]
        lncrna_ml_embed = ml_h[num_mirna:]

        ld_features = torch.cat([lncrna_h, disease_h], dim=0)
        ld_h = self._run_gnn_stack(self.gnn_ld_layers, lncrna_disease_adj, ld_features)
        lncrna_ld_embed = ld_h[:num_lncrna]
        disease_ld_embed = ld_h[num_lncrna:]

        # 按路径使用异类型节点作为相似度参考：
        # miRNA: md路径参考disease，ml路径参考lncRNA
        # lncRNA: ml路径参考miRNA，ld路径参考disease
        # disease: md路径参考miRNA，ld路径参考lncRNA
        mirna_fused = self.mirna_fusion(
            [mirna_md_embed, mirna_ml_embed],
            hetero_refs=[disease_md_embed, lncrna_ml_embed]
        )
        lncrna_fused = self.lncrna_fusion(
            [lncrna_ml_embed, lncrna_ld_embed],
            hetero_refs=[mirna_ml_embed, disease_ld_embed]
        )
        disease_fused = self.disease_fusion(
            [disease_md_embed, disease_ld_embed],
            hetero_refs=[mirna_md_embed, lncrna_ld_embed]
        )

        mirna_skip = self.mirna_skip_proj(mirna_h)
        lncrna_skip = self.lncrna_skip_proj(lncrna_h)
        disease_skip = self.disease_skip_proj(disease_h)

        residual_ratio = float(self.residual_ratio)
        mirna_final = (1.0 - residual_ratio) * mirna_fused + residual_ratio * mirna_skip
        lncrna_final = (1.0 - residual_ratio) * lncrna_fused + residual_ratio * lncrna_skip
        disease_final = (1.0 - residual_ratio) * disease_fused + residual_ratio * disease_skip

        mirna_final = self.output_dropout(self.mirna_out_norm(mirna_final))
        lncrna_final = self.output_dropout(self.lncrna_out_norm(lncrna_final))
        disease_final = self.output_dropout(self.disease_out_norm(disease_final))

        return mirna_final, lncrna_final, disease_final


ThreeLayerHeteroGNN = ThreeLayerSemanticAttentionNetwork
