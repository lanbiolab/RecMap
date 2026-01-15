# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class RECMAP(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RECMAP, self).__init__(config, dataset)

        self.build_item_graph = True
        self.feat_bipartite_adj = None
        self.feat_bipartite_adj_built = False

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k_mm = config['knn_k_mm']
        self.knn_k_up = config['knn_k_up']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_mm_layers = config['n_mm_layers']
        self.n_up_layers = config['n_up_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.up_image_weight = config['up_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']

        self.gamma = config['gamma']
        self.mu = config['mu']
        self.num_clusters = config['num_clusters']
        self.tau = config['tau']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.item_main_cluster = nn.Linear(self.embedding_dim, self.num_clusters)
        self.user_main_cluster = nn.Linear(self.embedding_dim, self.num_clusters)
        if self.v_feat is not None:
            self.item_image_cluster = nn.Linear(self.feat_embed_dim, self.num_clusters)
        if self.t_feat is not None:
            self.item_text_cluster = nn.Linear(self.feat_embed_dim, self.num_clusters)
        if self.p_feat is not None:
            self.user_portrait_cluster = nn.Linear(self.feat_embed_dim, self.num_clusters)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_uurec_{}_{}.pt'.format(self.knn_k_mm, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        if self.p_feat is not None:
            self.portrait_embedding = nn.Embedding.from_pretrained(self.p_feat, freeze=False)
            self.portrait_trs = nn.Linear(self.p_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                v_indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.knn_k_mm)
                self.mm_adj = image_adj
            if self.t_feat is not None:
                t_indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach(), self.knn_k_mm)
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

    def get_knn_adj_mat(self, mm_embeddings, knn_k):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))

        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        self.build_item_graph = True
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def build_feat_bipartite_adj(self):
        if not self.feat_bipartite_adj_built and self.p_feat is not None and (
                self.v_feat is not None or self.t_feat is not None):

            p_feat = self.portrait_trs(self.portrait_embedding.weight).detach()
            p_feat = F.normalize(p_feat, p=2, dim=1)

            S_combined = None

            if self.v_feat is not None:
                v_feat = self.image_trs(self.image_embedding.weight).detach()
                v_feat = F.normalize(v_feat, p=2, dim=1)
                S_v = torch.mm(p_feat, v_feat.t())

                if S_combined is None:
                    S_combined = self.up_image_weight * S_v
                else:
                    S_combined += self.up_image_weight * S_v

            if self.t_feat is not None:
                t_feat = self.text_trs(self.text_embedding.weight).detach()
                t_feat = F.normalize(t_feat, p=2, dim=1)
                S_t = torch.mm(p_feat, t_feat.t())

                if S_combined is None:
                    S_combined = (1.0 - self.up_image_weight) * S_t
                else:
                    S_combined += (1.0 - self.up_image_weight) * S_t

            _, topk_indices = torch.topk(S_combined, k=self.knn_k_up, dim=1)

            users = torch.arange(self.n_users, device=self.device).view(-1, 1).repeat(1, self.knn_k_up)
            items = topk_indices.view(-1)

            edges_up = torch.stack([users.flatten(), items + self.n_users])
            edges_down = torch.stack([items + self.n_users, users.flatten()])
            all_edges = torch.cat([edges_up, edges_down], dim=1)

            values = torch.ones(all_edges.size(1), device=self.device)
            adj = torch.sparse.FloatTensor(all_edges, values,
                                           torch.Size([self.n_users + self.n_items,
                                                       self.n_users + self.n_items]))

            deg = torch.sparse.sum(adj, dim=1).to_dense()
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

            norm_values = deg_inv_sqrt[all_edges[0]] * deg_inv_sqrt[all_edges[1]]

            self.feat_bipartite_adj_built = True

            return torch.sparse.FloatTensor(all_edges, norm_values, torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])).coalesce()

    def forward(self, adj, build_item_graph = False):
        if build_item_graph:
            if self.v_feat is not None:
                v_indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.knn_k_mm)
                self.mm_adj = image_adj
            if self.t_feat is not None:
                t_indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach(), self.knn_k_mm)
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            if self.p_feat is not None and not self.feat_bipartite_adj_built:
                self.feat_bipartite_adj = self.build_feat_bipartite_adj()

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        if self.mm_adj is not None:
            h_mm = self.item_id_embedding.weight
            for i in range(self.n_mm_layers):
                h_mm = torch.sparse.mm(self.mm_adj, h_mm)

        if self.feat_bipartite_adj is not None:
            feat_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            for _ in range(self.n_up_layers):
                feat_embeddings = torch.sparse.mm(self.feat_bipartite_adj.to(self.device), feat_embeddings)
            feat_u, feat_i = torch.split(feat_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings + feat_u, i_g_embeddings + feat_i + h_mm

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def cluster_contrastive_loss(self, z1, z2):
        # z1, z2: (batch_size, num_clusters)
        batch_size = z1.size(0)
        sim = torch.mm(z1, z2.T) / self.tau
        labels = torch.arange(batch_size).to(z1.device)
        return F.cross_entropy(sim, labels)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj, self.build_item_graph)
        self.build_item_graph = False

        mc_loss =  0.0
        cc_loss = 0.0

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        pos_items = interaction[1].squeeze()
        main_item_cluster = F.softmax(self.item_main_cluster(ia_embeddings), dim=-1)
        main_item_cluster_batch = main_item_cluster[pos_items]
        main_user_cluster = F.softmax(self.user_main_cluster(ua_embeddings), dim=-1)
        main_user_cluster_batch = main_user_cluster[pos_items]

        mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mc_loss += self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
            text_cluster = F.softmax(self.item_text_cluster(text_feats), dim=-1)
            cc_loss += self.cluster_contrastive_loss(main_item_cluster_batch, text_cluster)
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mc_loss += self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
            image_cluster = F.softmax(self.item_image_cluster(image_feats), dim=-1)
            cc_loss += self.cluster_contrastive_loss(main_item_cluster_batch, image_cluster)
        if self.p_feat is not None:
            portrait_feats = self.portrait_trs(self.portrait_embedding.weight)
            p_cluster = F.softmax(self.user_portrait_cluster(portrait_feats), dim=-1)
            cc_loss += self.cluster_contrastive_loss(main_user_cluster_batch, p_cluster)

        if self.v_feat is not None and self.t_feat is not None and self.p_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            portrait_feats = self.portrait_trs(self.portrait_embedding.weight)
            mc_loss += (self.bpr_loss(portrait_feats[users], text_feats[pos_items], text_feats[neg_items]) +
                        self.bpr_loss(portrait_feats[users], image_feats[pos_items], image_feats[neg_items]))
            cc_loss /= 3

        total_loss = mf_loss + self.gamma * mc_loss + self.mu * cc_loss

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

