import torch
import torch.nn as nn
import torch.nn.functional as F


class HIN2VecWithAttentionAndSimilarity(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        num_relations,
        input_matrix,
        similarity_matrix,
        num_heads=8,
        dropout=0.2,
    ):
        """
            Relation-aware HIN2Vec with the similarity prior from Eq. 28.
            - input_matrix: retained for shape validation and compatibility with
              callers that build the full heterogeneous block matrix A.
            - similarity_matrix: stores the prior similarity s_ij.
        """
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        self.num_relations = int(num_relations)

        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)#节点嵌入和关系嵌入都是可学习嵌入，
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.node_layer_norm = nn.LayerNorm(embedding_dim)
        self.relation_layer_norm = nn.LayerNorm(embedding_dim)#两个归一化层

        bias = torch.zeros(num_relations, num_heads)
        bias += 0.01 * torch.randn_like(bias)
        self.relation_attention_bias = nn.Parameter(bias)#每个头一个偏置向量

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        self.relation_query = nn.Linear(embedding_dim, embedding_dim)
        self.relation_key = nn.Linear(embedding_dim, embedding_dim)
        self.relation_value = nn.Linear(embedding_dim, embedding_dim)

        self.attention_output = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        normalized_input_matrix = self._row_normalize_input_matrix(input_matrix)
        similarity_matrix = torch.as_tensor(similarity_matrix, dtype=torch.float32)
        if similarity_matrix.shape != normalized_input_matrix.shape:
            raise ValueError("input_matrix and similarity_matrix must have the same shape")
        self.register_buffer("input_matrix", normalized_input_matrix)#保存归一化矩阵
        self.register_buffer("similarity_matrix", similarity_matrix)

        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self._init_weights()

    def _row_normalize_input_matrix(self, matrix):
        matrix = torch.as_tensor(matrix, dtype=torch.float32)
        if matrix.dim() != 2:
            raise ValueError("input_matrix must be a 2D tensor or array")
        if matrix.size(0) != matrix.size(1):
            raise ValueError("input_matrix must be square")#检验是二维方阵

        row_sum = matrix.sum(dim=1, keepdim=True)
        safe_row_sum = row_sum.clamp(min=1e-12)
        normalized = matrix / safe_row_sum#初始矩阵一开始会做行归一化

        zero_row_mask = row_sum <= 1e-12#对全0行的处理
        if bool(zero_row_mask.any()):
            normalized = normalized.masked_fill(zero_row_mask.expand_as(normalized), 0.0)
        return normalized

    def _init_weights(self):#初始化权重
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def _normalize_node_representation(self, x):#节点表示向量，先层归一化，后l2归一化
        x = self.node_layer_norm(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_node_input_rows(self, node_indices):#获取节点索引对应的行
        node_indices = node_indices.long()
        return self.input_matrix.index_select(0, node_indices)

    def forward(self, node_i, relation, node_j, group_ids=None):
        node_i = node_i.long()
        node_j = node_j.long()
        relation = relation.long()

        node_i_emb = self.node_embeddings(node_i)
        node_j_emb = self.node_embeddings(node_j)
        relation_emb = self.relation_embeddings(relation)

        node_i_emb = self._normalize_node_representation(node_i_emb)
        node_j_emb = self._normalize_node_representation(node_j_emb)
        relation_emb = self.relation_layer_norm(relation_emb)

        s_ij = self.get_similarity(node_i, node_j)

        if group_ids is None:
            group_ids = relation * self.input_matrix.size(0) + node_i

        pair_context = self.compute_relation_aware_multihead_context(
            node_i_emb,
            node_j_emb,
            relation_emb,
            relation,
            s_ij,
            group_ids=group_ids,
        )#返回shape[batch_size, embedding_dim]

        attention_pair_score = torch.sum(pair_context * node_j_emb, dim=-1)
        pair_score = attention_pair_score
        pred_prob = torch.sigmoid(pair_score)#转成概率

        return pred_prob, self.get_lambda().detach()

    def _candidate_group_softmax(self, scores, value, group_ids=None):
        batch_size = scores.size(0)
        if group_ids is None:
            inverse = torch.arange(batch_size, device=scores.device)
            attn_weights = torch.ones_like(scores)
            num_groups = batch_size
        else:
            group_ids = group_ids.to(scores.device).long()
            _, inverse = torch.unique(group_ids, sorted=False, return_inverse=True)
            num_groups = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
            attn_weights = torch.empty_like(scores)
            for group_idx in range(num_groups):
                mask = inverse == group_idx
                attn_weights[mask] = F.softmax(scores[mask], dim=0)

        #attn_weights = self.dropout(attn_weights)
        weighted_value = attn_weights.unsqueeze(-1) * value
        group_context = torch.zeros(
            num_groups,
            self.num_heads,
            self.head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        group_context.index_add_(0, inverse, weighted_value)
        return group_context[inverse]#返回[batch_size, num_heads]

    def compute_relation_aware_multihead_context(
        self,
        node_i_emb,
        node_j_emb,
        relation_emb,
        relation_idx,
        s_ij,
        group_ids=None,
    ):
        batch_size = node_i_emb.size(0)

        query = self.query_proj(node_i_emb).view(
            batch_size, self.num_heads, self.head_dim
        )
        key = self.key_proj(node_j_emb).view(
            batch_size, self.num_heads, self.head_dim
        )
        value = self.value_proj(node_j_emb).view(
            batch_size, self.num_heads, self.head_dim
        )

        rel_query = self.relation_query(relation_emb).view(
            batch_size, self.num_heads, self.head_dim
        )
        rel_key = self.relation_key(relation_emb).view(
            batch_size, self.num_heads, self.head_dim
        )
        rel_value = self.relation_value(relation_emb).view(
            batch_size, self.num_heads, self.head_dim
        )

        query = query + rel_query
        key = key + rel_key
        value = value + rel_value

        scores = torch.sum(query * key, dim=-1) / (self.head_dim ** 0.5)
        relation_specific_bias = self.relation_attention_bias[relation_idx]
        scores = scores + relation_specific_bias
        scores = scores + self.get_lambda() * s_ij.unsqueeze(-1)

        candidate_context = self._candidate_group_softmax(
            scores,
            value,
            group_ids=group_ids,
        )

        context = candidate_context.reshape(batch_size, -1)
        output = self.attention_output(context)
        return output

    def get_similarity(self, node_i, node_j):
        source_rows = self.similarity_matrix.index_select(0, node_i.long())
        return source_rows.gather(1, node_j.long().unsqueeze(1)).squeeze(1)

    def get_lambda(self):
        return self.beta

    def get_node_embeddings(self):
        embeddings = self.node_embeddings.weight.detach()
        embeddings = self._normalize_node_representation(embeddings)
        return embeddings

    def get_relation_embeddings(self):
        embeddings = self.relation_embeddings.weight.detach()
        embeddings = self.relation_layer_norm(embeddings)
        return embeddings
