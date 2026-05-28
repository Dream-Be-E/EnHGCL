import os
import numpy as np
import networkx as nx
import pandas as pd


class MLDHG:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.G = nx.Graph()
        self.mirna_nodes = set()
        self.lncrna_nodes = set()
        self.disease_nodes = set()
        self.mirna_sim = None
        self.lncrna_sim = None
        self.disease_sim = None
        self.mirna_disease_assoc = None
        self.mirna_lncrna_assoc = None
        self.lncrna_disease_assoc = None
        self.mirna_names = []
        self.lncrna_names = []
        self.disease_names = []
        self.node_info = None
        self.edge_info = None
        self.association_blocks = None
        self.association_block_matrix = None
        self.similarity_block_matrix = None
        self.input_matrix = None
        self.association_edge_indices = {
            'md': [],
            'ml': [],
            'ld': [],
        }

    def load_similarity_matrices(self):
        self.mirna_sim = pd.read_csv(os.path.join(self.data_dir, 'IM_integrated_with_names.csv'), index_col=0)
        self.lncrna_sim = pd.read_csv(os.path.join(self.data_dir, 'IL_integrated_with_names.csv'), index_col=0)
        self.disease_sim = pd.read_csv(os.path.join(self.data_dir, 'ID_integrated_with_names.csv'), index_col=0)
        self.mirna_names = self.mirna_sim.index.tolist()
        self.lncrna_names = self.lncrna_sim.index.tolist()
        self.disease_names = self.disease_sim.index.tolist()

        self.mirna_sim = self._align_matrix(self.mirna_sim, self.mirna_names, self.mirna_names)
        self.lncrna_sim = self._align_matrix(self.lncrna_sim, self.lncrna_names, self.lncrna_names)
        self.disease_sim = self._align_matrix(self.disease_sim, self.disease_names, self.disease_names)

    def _align_matrix(self, matrix_df, row_names, col_names):
        aligned = matrix_df.reindex(index=row_names, columns=col_names).fillna(0.0)
        return aligned.astype(np.float32)

    def _load_association_matrix(self, relative_path, row_names, col_names):
        matrix_df = pd.read_csv(os.path.join(self.data_dir, relative_path), index_col=0)

        row_set = set(str(name) for name in row_names)
        col_set = set(str(name) for name in col_names)
        direct_score = (
            len(set(matrix_df.index.astype(str)) & row_set) +
            len(set(matrix_df.columns.astype(str)) & col_set)
        )
        transpose_score = (
            len(set(matrix_df.index.astype(str)) & col_set) +
            len(set(matrix_df.columns.astype(str)) & row_set)
        )
        if transpose_score > direct_score:
            matrix_df = matrix_df.T

        matrix_df.index = matrix_df.index.astype(str)
        matrix_df.columns = matrix_df.columns.astype(str)
        return self._align_matrix(matrix_df, list(map(str, row_names)), list(map(str, col_names)))

    def load_association_matrices(self):
        self.mirna_disease_assoc = self._load_association_matrix(
            'interaction/miRNA_disease_interaction.csv',
            self.mirna_names,
            self.disease_names,
        )
        self.mirna_lncrna_assoc = self._load_association_matrix(
            'interaction/miRNA_lncRNA_interaction.csv',
            self.mirna_names,
            self.lncrna_names,
        )
        self.lncrna_disease_assoc = self._load_association_matrix(
            'interaction/disease_lncrna_interaction.csv',
            self.lncrna_names,
            self.disease_names,
        )

    def get_ordered_node_names(self):
        return list(self.mirna_names) + list(self.lncrna_names) + list(self.disease_names)

    def add_nodes(self):
        for mirna in self.mirna_names:
            self.G.add_node(mirna, node_type='miRNA', layer=[1, 2])
            self.mirna_nodes.add(mirna)

        for lncrna in self.lncrna_names:
            self.G.add_node(lncrna, node_type='lncRNA', layer=[2, 3])
            self.lncrna_nodes.add(lncrna)

        for disease in self.disease_names:
            self.G.add_node(disease, node_type='disease', layer=[1, 3])
            self.disease_nodes.add(disease)

    def add_association_edges(self):
        self.association_edge_indices['md'] = []
        for mirna in self.mirna_disease_assoc.index:
            for disease in self.mirna_disease_assoc.columns:
                weight = float(self.mirna_disease_assoc.loc[mirna, disease])
                if weight > 0 and mirna in self.mirna_nodes and disease in self.disease_nodes:
                    self.G.add_edge(mirna, disease, edge_type='miRNA-disease', layer=1, weight=weight)
                    self.association_edge_indices['md'].append((mirna, disease))

        self.association_edge_indices['ml'] = []
        for mirna in self.mirna_lncrna_assoc.index:
            for lncrna in self.mirna_lncrna_assoc.columns:
                weight = float(self.mirna_lncrna_assoc.loc[mirna, lncrna])
                if weight > 0 and mirna in self.mirna_nodes and lncrna in self.lncrna_nodes:
                    self.G.add_edge(mirna, lncrna, edge_type='miRNA-lncRNA', layer=2, weight=weight)
                    self.association_edge_indices['ml'].append((mirna, lncrna))

        self.association_edge_indices['ld'] = []
        for lncrna in self.lncrna_disease_assoc.index:
            for disease in self.lncrna_disease_assoc.columns:
                weight = float(self.lncrna_disease_assoc.loc[lncrna, disease])
                if weight > 0 and lncrna in self.lncrna_nodes and disease in self.disease_nodes:
                    self.G.add_edge(lncrna, disease, edge_type='lncRNA-disease', layer=3, weight=weight)
                    self.association_edge_indices['ld'].append((lncrna, disease))

    def add_similarity_edges_threshold(self, threshold=0.5):
        mirna_list = list(self.mirna_nodes)
        for i, mirna1 in enumerate(mirna_list):
            for mirna2 in mirna_list[i + 1:]:
                sim = self.mirna_sim.loc[mirna1, mirna2]
                if sim > threshold:
                    self.G.add_edge(mirna1, mirna2, edge_type='miRNA-similarity', layer=1, weight=float(sim))

        lncrna_list = list(self.lncrna_nodes)
        for i, lncrna1 in enumerate(lncrna_list):
            for lncrna2 in lncrna_list[i + 1:]:
                sim = self.lncrna_sim.loc[lncrna1, lncrna2]
                if sim > threshold:
                    self.G.add_edge(lncrna1, lncrna2, edge_type='lncRNA-similarity', layer=2, weight=float(sim))

        disease_list = list(self.disease_nodes)
        for i, disease1 in enumerate(disease_list):
            for disease2 in disease_list[i + 1:]:
                sim = self.disease_sim.loc[disease1, disease2]
                if sim > threshold:
                    self.G.add_edge(disease1, disease2, edge_type='disease-similarity', layer=3, weight=float(sim))

    def _build_node_info(self):
        n_mirna = len(self.mirna_names)
        n_lncrna = len(self.lncrna_names)
        n_disease = len(self.disease_names)
        ordered_node_names = self.get_ordered_node_names()
        return {
            'n_disease': n_disease,
            'n_mirna': n_mirna,
            'n_lncrna': n_lncrna,
            'disease_names': list(self.disease_names),
            'mirna_names': list(self.mirna_names),
            'lncrna_names': list(self.lncrna_names),
            'ordered_node_names': ordered_node_names,
            'node_type_ranges': {
                'miRNA': (0, n_mirna),
                'lncRNA': (n_mirna, n_mirna + n_lncrna),
                'disease': (n_mirna + n_lncrna, n_mirna + n_lncrna + n_disease),
            },
            'mirna_name_to_idx': {name: i for i, name in enumerate(self.mirna_names)},
            'lncrna_name_to_idx': {
                name: i + n_mirna
                for i, name in enumerate(self.lncrna_names)
            },
            'disease_name_to_idx': {
                name: i + n_mirna + n_lncrna
                for i, name in enumerate(self.disease_names)
            },
        }

    def _build_edge_info(self):
        md_count = 0
        ml_count = 0
        ld_count = 0
        mm_sim_count = 0
        ll_sim_count = 0
        dd_sim_count = 0
        for _, _, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', '')
            if edge_type == 'miRNA-disease':
                md_count += 1
            elif edge_type == 'miRNA-lncRNA':
                ml_count += 1
            elif edge_type == 'lncRNA-disease':
                ld_count += 1
            elif edge_type == 'miRNA-similarity':
                mm_sim_count += 1
            elif edge_type == 'lncRNA-similarity':
                ll_sim_count += 1
            elif edge_type == 'disease-similarity':
                dd_sim_count += 1

        return {
            'edge_count': {
                'md': md_count,
                'ml': ml_count,
                'ld': ld_count,
                'mm_sim': mm_sim_count,
                'll_sim': ll_sim_count,
                'dd_sim': dd_sim_count,
            },
            'edge_types': {
                0: 'miRNA-disease',
                1: 'miRNA-lncRNA',
                2: 'lncRNA-disease',
                3: 'miRNA-similarity',
                4: 'lncRNA-similarity',
                5: 'disease-similarity',
            },
            'md_edge_indices': list(self.association_edge_indices['md']),
            'ml_edge_indices': list(self.association_edge_indices['ml']),
            'ld_edge_indices': list(self.association_edge_indices['ld']),
        }

    def _build_association_blocks_from_graph(self):
        md = pd.DataFrame(0.0, index=self.mirna_names, columns=self.disease_names, dtype=np.float32)
        ml = pd.DataFrame(0.0, index=self.mirna_names, columns=self.lncrna_names, dtype=np.float32)
        ld = pd.DataFrame(0.0, index=self.lncrna_names, columns=self.disease_names, dtype=np.float32)

        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', '')
            weight = float(data.get('weight', 1.0))

            if edge_type == 'miRNA-disease':
                if u in md.index and v in md.columns:
                    md.loc[u, v] = max(float(md.loc[u, v]), weight)
                elif v in md.index and u in md.columns:
                    md.loc[v, u] = max(float(md.loc[v, u]), weight)
            elif edge_type == 'miRNA-lncRNA':
                if u in ml.index and v in ml.columns:
                    ml.loc[u, v] = max(float(ml.loc[u, v]), weight)
                elif v in ml.index and u in ml.columns:
                    ml.loc[v, u] = max(float(ml.loc[v, u]), weight)
            elif edge_type == 'lncRNA-disease':
                if u in ld.index and v in ld.columns:
                    ld.loc[u, v] = max(float(ld.loc[u, v]), weight)
                elif v in ld.index and u in ld.columns:
                    ld.loc[v, u] = max(float(ld.loc[v, u]), weight)

        return {
            'miRNA_disease': md,
            'miRNA_lncRNA': ml,
            'lncRNA_disease': ld,
        }

    def rebuild_input_matrices(self):
        if len(self.mirna_names) == 0 and len(self.lncrna_names) == 0 and len(self.disease_names) == 0:
            self.association_blocks = None
            self.association_block_matrix = None
            self.similarity_block_matrix = None
            self.input_matrix = None
            return None

        ordered_node_names = self.get_ordered_node_names()
        total_nodes = len(ordered_node_names)
        n_mirna = len(self.mirna_names)
        n_lncrna = len(self.lncrna_names)
        n_disease = len(self.disease_names)
        lnc_start = n_mirna
        disease_start = n_mirna + n_lncrna

        similarity_block = pd.DataFrame(
            np.zeros((total_nodes, total_nodes), dtype=np.float32),
            index=ordered_node_names,
            columns=ordered_node_names,
        )
        if self.mirna_sim is not None:
            similarity_block.iloc[:n_mirna, :n_mirna] = self.mirna_sim.values
        if self.lncrna_sim is not None:
            similarity_block.iloc[lnc_start:lnc_start + n_lncrna, lnc_start:lnc_start + n_lncrna] = self.lncrna_sim.values
        if self.disease_sim is not None:
            similarity_block.iloc[disease_start:disease_start + n_disease, disease_start:disease_start + n_disease] = self.disease_sim.values

        association_blocks = self._build_association_blocks_from_graph()
        md = association_blocks['miRNA_disease']
        ml = association_blocks['miRNA_lncRNA']
        ld = association_blocks['lncRNA_disease']

        association_block = pd.DataFrame(
            np.zeros((total_nodes, total_nodes), dtype=np.float32),
            index=ordered_node_names,
            columns=ordered_node_names,
        )
        association_block.iloc[:n_mirna, lnc_start:lnc_start + n_lncrna] = ml.values
        association_block.iloc[lnc_start:lnc_start + n_lncrna, :n_mirna] = ml.values.T
        association_block.iloc[:n_mirna, disease_start:disease_start + n_disease] = md.values
        association_block.iloc[disease_start:disease_start + n_disease, :n_mirna] = md.values.T
        association_block.iloc[lnc_start:lnc_start + n_lncrna, disease_start:disease_start + n_disease] = ld.values
        association_block.iloc[disease_start:disease_start + n_disease, lnc_start:lnc_start + n_lncrna] = ld.values.T

        self.association_blocks = association_blocks
        self.association_block_matrix = association_block
        self.similarity_block_matrix = similarity_block
        self.input_matrix = (association_block + similarity_block).astype(np.float32)
        return self.input_matrix

    def get_input_matrix(self, as_numpy=False):
        if self.input_matrix is None:
            self.rebuild_input_matrices()
        if self.input_matrix is None:
            return None
        if as_numpy:
            return self.input_matrix.values.astype(np.float32, copy=True)
        return self.input_matrix.copy()

    def refresh_metadata(self):
        self.node_info = self._build_node_info()
        self.edge_info = self._build_edge_info()
        self.rebuild_input_matrices()

    def build_graph(self, add_similarity=True, sim_threshold=0.0):
        self.G.clear()
        self.mirna_nodes.clear()
        self.lncrna_nodes.clear()
        self.disease_nodes.clear()
        self.load_similarity_matrices()
        self.load_association_matrices()
        self.add_nodes()
        self.add_association_edges()
        if add_similarity:
            self.add_similarity_edges_threshold(threshold=sim_threshold)
        self.refresh_metadata()

    def build_network(self, add_similarity=True, sim_threshold=0.0):
        self.build_graph(add_similarity=add_similarity, sim_threshold=sim_threshold)

    def save_network(self, output_path):
        import pickle

        with open(output_path, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"\n网络已保存到: {output_path}")

    def export_edgelist(self, output_path):
        edges_data = []
        for u, v, data in self.G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'edge_type': data.get('edge_type', 'unknown'),
                'layer': data.get('layer', 0),
                'weight': data.get('weight', 1.0),
            })

        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(output_path, index=False)
        print(f"\n边列表已保存到: {output_path}")

    def extract_positive_task_samples(self, task):
        if self.node_info is None:
            self.refresh_metadata()

        positive_samples = []
        existing_edges = set()
        mirna_name_to_idx = self.node_info['mirna_name_to_idx']
        lncrna_name_to_idx = self.node_info['lncrna_name_to_idx']
        disease_name_to_idx = self.node_info['disease_name_to_idx']

        for u, v, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', '')
            if task == 'miRNA_disease' and edge_type == 'miRNA-disease':
                if u in mirna_name_to_idx and v in disease_name_to_idx:
                    sample = (mirna_name_to_idx[u], disease_name_to_idx[v], 1)
                elif v in mirna_name_to_idx and u in disease_name_to_idx:
                    sample = (mirna_name_to_idx[v], disease_name_to_idx[u], 1)
                else:
                    continue
            elif task == 'miRNA_lncRNA' and edge_type == 'miRNA-lncRNA':
                if u in mirna_name_to_idx and v in lncrna_name_to_idx:
                    sample = (mirna_name_to_idx[u], lncrna_name_to_idx[v], 1)
                elif v in mirna_name_to_idx and u in lncrna_name_to_idx:
                    sample = (mirna_name_to_idx[v], lncrna_name_to_idx[u], 1)
                else:
                    continue
            elif task == 'lncRNA_disease' and edge_type == 'lncRNA-disease':
                if u in lncrna_name_to_idx and v in disease_name_to_idx:
                    sample = (lncrna_name_to_idx[u], disease_name_to_idx[v], 1)
                elif v in lncrna_name_to_idx and u in disease_name_to_idx:
                    sample = (lncrna_name_to_idx[v], disease_name_to_idx[u], 1)
                else:
                    continue
            else:
                continue

            if sample[:2] not in existing_edges:
                positive_samples.append(list(sample))
                existing_edges.add(sample[:2])

        if len(positive_samples) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(positive_samples, dtype=np.int64)


def build_mldhg(data_dir, add_similarity=True, sim_threshold=0.0):
    mldhg = MLDHG(data_dir=data_dir)
    mldhg.build_graph(add_similarity=add_similarity, sim_threshold=sim_threshold)
    return mldhg
