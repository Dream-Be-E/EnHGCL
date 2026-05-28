# -*- coding: utf-8 -*-
from dataclasses import dataclass
import torch
from compare_net import (
    build_similarity_feature_embeddings,
    build_weighted_augmented_network,
    pretrain_source_graph_augmentor,
)
from train_augmentation_utils import (
    _augment_similarity_matrix,
    _build_augmented_node_info,
    _build_augmented_semantic_graph,
    _build_full_hin2vec_input_matrix,
    _build_full_similarity_matrix,
    _extract_augmented_positive_samples,
    _remap_samples_to_node_info,
)
from train_hin2vec import prepare_hin2vec_data_for_fold
from utils import build_task_samples_from_positive_samples


@dataclass
class AugmentedGraphContext:
    source_graph_stats: dict#保存源图预训练得到的边权、节点重要性等统计信息；
    network_aug: object#增强后的网络；
    augmentation_plan: dict#增强方案，比如新增了哪些节点、软删除了哪些节点；
    augmentation_config: dict#
    aug_node_info_dict: dict#增强后的节点信息；
    aug_node_names: dict#
    aug_name_to_idx: dict#
    aug_idx_to_name: dict#
    aug_IM: object##增强后的miRNA、lncRNA、disease 相似性矩阵；
    aug_IL: object#
    aug_ID: object#
    task_train_samples_aug: dict##增强图上的训练样本；
    task_val_samples_aug_semantic: dict#映射到增强图索引体系下的验证样本；
    aug_graph: dict#
    aug_mirna_features: torch.Tensor#
    aug_lncrna_features: torch.Tensor#
    aug_disease_features: torch.Tensor#
    aug_train_data: object#增强图对应的 HIN2Vec 训练数据；
    aug_input_matrix_full: torch.Tensor#完整输入矩阵
    aug_similarity_matrix_full: torch.Tensor#完整相似性矩阵。


def _print_augmentation_summary(augmentation_plan, augmentation_config, node_info_dict):
    added_m = len(augmentation_plan['added_nodes']['miRNA'])
    added_l = len(augmentation_plan['added_nodes']['lncRNA'])
    added_d = len(augmentation_plan['added_nodes']['disease'])
    soft_deleted_total = (
        len(augmentation_plan['soft_deleted_nodes']['miRNA']) +
        len(augmentation_plan['soft_deleted_nodes']['lncRNA']) +
        len(augmentation_plan['soft_deleted_nodes']['disease'])
    )
    print(
        " 增强图信息: "
        f"新增节点(miRNA/lncRNA/disease)="
        f"{added_m}/{added_l}/{added_d}, "
        f"软删除节点="
        f"{soft_deleted_total}"
    )

    base_counts = [
        ('miRNA', node_info_dict['n_mirna']),
        ('lncRNA', node_info_dict['n_lncrna']),
        ('disease', node_info_dict['n_disease']),
    ]
    for node_type, base_count in base_counts:
        added_count = len(augmentation_plan['added_nodes'][node_type])
        soft_count = len(augmentation_plan['soft_deleted_nodes'][node_type])
        add_ratio_actual = added_count / max(base_count, 1)
        soft_ratio_actual = soft_count / max(base_count, 1)
        print(
            f"    [{node_type}] 新增比例(实际/配置)="
            f"{add_ratio_actual:.3f}/{augmentation_config['add_node_ratio']:.3f}, "
            f"软删除比例(实际/配置)="
            f"{soft_ratio_actual:.3f}/{augmentation_config['drop_node_ratio']:.3f}"
        )
        if add_ratio_actual > augmentation_config['add_node_ratio'] * 2.5:
            print(
                f"    警告：{node_type} 新增比例可能偏高 "
                f"({add_ratio_actual:.3f} 对比配置 {augmentation_config['add_node_ratio']:.3f})"
            )


def build_augmented_graph_context(
    *,
    network_train,
    final_emb,
    node_names,
    similarity_features,
    node_info_dict,
    IM,
    IL,
    ID,
    task_val_samples_semantic,
    tasks,
    relation_to_idx,
    random_seed,
    fold,
    contrastive_epochs,
    source_edge_pretrain_epochs,
    source_edge_pretrain_batch_size,
    source_edge_pretrain_learning_rate,
    negative_ratio_hin2vec,
    device,
    add_node_ratio=0.015,
    drop_node_ratio=0.05,
    edge_prune_ratio=0.04,
):
    contrastive_edge_feature_embeddings = build_similarity_feature_embeddings(
        similarity_features,
        device=device,
    )#将相似度特征转换为可使用的特征嵌入
    source_graph_stats = pretrain_source_graph_augmentor(
        network=network_train,#源图网络
        initial_embeddings=final_emb,#源图节点初始嵌入
        node_names=node_names,#源图节点名称
        edge_feature_embeddings=contrastive_edge_feature_embeddings,#源图边特征嵌入
        num_epochs=source_edge_pretrain_epochs,#预训练迭代次数
        batch_size=source_edge_pretrain_batch_size,#预训练批次大小
        aug_lr=source_edge_pretrain_learning_rate,#预训练学习率
        device=device,
        lambda_sparse=0.000002,
    )#源图初始边预训练，决定那些边适合增强

    augmentation_config = {
        'add_node_ratio': float(add_node_ratio),
        'drop_node_ratio': float(drop_node_ratio),
        'edge_prune_ratio': float(edge_prune_ratio),
    }
    print(
        "增强配置："
        f"加节点比例={augmentation_config['add_node_ratio']:.3f},"
        f"软删除节点比例={augmentation_config['drop_node_ratio']:.3f},"
        f"剪边比例={augmentation_config['edge_prune_ratio']:.3f}"
    )
    network_aug, augmentation_plan = build_weighted_augmented_network(
        network=network_train,
        node_names=node_names,
        source_graph_stats=source_graph_stats,
        initial_embeddings=final_emb,
        random_seed=int(random_seed + fold * 31),
        add_node_ratio=augmentation_config['add_node_ratio'],
        drop_node_ratio=augmentation_config['drop_node_ratio'],
        edge_prune_ratio=augmentation_config['edge_prune_ratio'],
    )#根据预训练对原始网络进行增强
    _print_augmentation_summary(augmentation_plan, augmentation_config, node_info_dict)

    aug_node_info_dict = _build_augmented_node_info(node_info_dict, augmentation_plan)#增强图的节点边，索引信息
    aug_node_names = {
        'miRNA': aug_node_info_dict['mirna_names'],
        'lncRNA': aug_node_info_dict['lncrna_names'],
        'disease': aug_node_info_dict['disease_names'],
    }
    aug_name_to_idx = {}
    aug_name_to_idx.update(aug_node_info_dict['disease_name_to_idx'])
    aug_name_to_idx.update(aug_node_info_dict['mirna_name_to_idx'])
    aug_name_to_idx.update(aug_node_info_dict['lncrna_name_to_idx'])
    aug_idx_to_name = {idx: name for name, idx in aug_name_to_idx.items()}

    aug_IM = _augment_similarity_matrix(
        IM,
        node_info_dict['mirna_names'],
        augmentation_plan['added_nodes']['miRNA'],
        augmentation_plan['added_node_anchors'],
    )
    aug_IL = _augment_similarity_matrix(
        IL,
        node_info_dict['lncrna_names'],
        augmentation_plan['added_nodes']['lncRNA'],
        augmentation_plan['added_node_anchors'],
    )
    aug_ID = _augment_similarity_matrix(
        ID,
        node_info_dict['disease_names'],
        augmentation_plan['added_nodes']['disease'],
        augmentation_plan['added_node_anchors'],
    )

    aug_positive_samples = _extract_augmented_positive_samples(
        network_aug,
        aug_node_info_dict,
        min_node_active_weight=0.2,
        min_effective_edge_weight=0.05,
        verbose=True,
    )#从增强图中提取正样本
    task_train_samples_aug = {}
    for task in tasks:#为每个任务构建训练以及验证样本
        task_train_samples_aug[task], _ = build_task_samples_from_positive_samples(
            node_info_dict=aug_node_info_dict,
            task=task,
            positive_samples=aug_positive_samples[task],
            known_positive_samples=aug_positive_samples[task],
            negative_ratio=1.0,
            random_seed=random_seed,
            verbose=False,
            log_prefix=f"[第 {fold} 折] 增强训练",
        )
    task_val_samples_aug_semantic = {
        task: _remap_samples_to_node_info(
            task_val_samples_semantic[task],
            src_node_info_dict=node_info_dict,
            dst_node_info_dict=aug_node_info_dict,
        )
        for task in tasks
    }

    aug_graph = _build_augmented_semantic_graph(network_aug, aug_node_info_dict, device)#构建增强语义图
    aug_mirna_features = torch.as_tensor(aug_IM, dtype=torch.float32, device=device)
    aug_lncrna_features = torch.as_tensor(aug_IL, dtype=torch.float32, device=device)
    aug_disease_features = torch.as_tensor(aug_ID, dtype=torch.float32, device=device)

    aug_train_data = prepare_hin2vec_data_for_fold(
        network=network_aug,
        task_train_samples=task_train_samples_aug,
        node_to_idx=aug_name_to_idx,
        relation_to_idx=relation_to_idx,
        negative_ratio=negative_ratio_hin2vec,
        sim_threshold=0.0,
        sim_negative_lower_bound=0.0,
        full_network=network_aug,
    )#构建语义注意力训练样本

    aug_input_matrix_full = _build_full_hin2vec_input_matrix(
        network_aug,
        aug_node_info_dict,
        aug_IM,
        aug_IL,
        aug_ID,
    )
    aug_similarity_matrix_full = _build_full_similarity_matrix(
        aug_node_info_dict,
        aug_IM,
        aug_IL,
        aug_ID,
    )

    return AugmentedGraphContext(
        source_graph_stats=source_graph_stats,
        network_aug=network_aug,
        augmentation_plan=augmentation_plan,
        augmentation_config=augmentation_config,
        aug_node_info_dict=aug_node_info_dict,
        aug_node_names=aug_node_names,
        aug_name_to_idx=aug_name_to_idx,
        aug_idx_to_name=aug_idx_to_name,
        aug_IM=aug_IM,
        aug_IL=aug_IL,
        aug_ID=aug_ID,
        task_train_samples_aug=task_train_samples_aug,
        task_val_samples_aug_semantic=task_val_samples_aug_semantic,
        aug_graph=aug_graph,
        aug_mirna_features=aug_mirna_features,
        aug_lncrna_features=aug_lncrna_features,
        aug_disease_features=aug_disease_features,
        aug_train_data=aug_train_data,
        aug_input_matrix_full=aug_input_matrix_full,
        aug_similarity_matrix_full=aug_similarity_matrix_full,
    )
