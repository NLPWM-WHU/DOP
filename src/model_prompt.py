#最终模型
import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch_sparse import masked_select_nnz
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import random
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 正则
class KGPrompt(nn.Module):
    def __init__(
            self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
            n_entity, num_users, adj_mat, topk, gama,itemid_set
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_entity = n_entity

        self.adj_mat = self._convert_sp_mat_to_sp_tensor(adj_mat).to(device)
        self.adj_mat = self.adj_mat.coalesce()
        self.adj_mat_t = self._convert_sp_mat_to_sp_tensor(adj_mat.T).to(device)
        self.adj_mat_t = self.adj_mat_t.coalesce()

        self.num_users = num_users

        self.node_embeds = nn.Parameter(torch.empty(n_entity, hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)

        self.token_proj = nn.Linear(token_hidden_size, hidden_size)

        # 新增用户表示
        self.user_embedding = nn.Parameter(torch.empty(num_users, hidden_size))
        user_stdv = math.sqrt(6.0 / (self.user_embedding.size(-2) + self.user_embedding.size(-1)))
        self.user_embedding.data.uniform_(-user_stdv, user_stdv)
        nn.init.zeros_(self.user_embedding[0])

        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, self.n_layer * n_block * hidden_size)

        self.sim_w1 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.sim_w2 = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.normal_(self.sim_w1.data)
        nn.init.normal_(self.sim_w2.data)

        self.topk = topk
        self.gama = gama

        self.f = nn.Sigmoid()
        self.p = nn.PReLU()

        self.trans = nn.Linear(self.hidden_size, self.hidden_size)
        self.trans_t = nn.Linear(self.hidden_size, self.hidden_size)

        self.trans_s = nn.Linear(self.hidden_size, 1)

        self.meta_dim = self.hidden_size // 2
        self.meta_decoder = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.meta_dim), torch.nn.ReLU(),
                                                torch.nn.Linear(self.meta_dim, self.hidden_size * self.hidden_size))

        self.itemid_set=itemid_set

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def nor_sparse_matrix(self, sparse_matrix):
        # 稀疏矩阵归一化
        sum_matrix = torch.sparse.sum(sparse_matrix, 1).to_dense()

        # 将matrix中的value替换
        idx = sparse_matrix.coalesce().indices()
        data = sparse_matrix.coalesce().values() / sum_matrix[idx[0]]

        coo_a = torch.sparse.FloatTensor(idx, data, sparse_matrix.shape)
        return coo_a


    def model_gcn(self, user_item_final_matrix, item_user_final_matrix, _user_emb, _item_emb, all_token_embeds):
        # gcn 4层
        user_emb = [_user_emb]
        item_emb = [_item_emb]

        user_token_emb = torch.mul(self.p(self.trans_t(all_token_embeds)), all_token_embeds)

        user_emb_layer1 = torch.sparse.mm(user_item_final_matrix, _item_emb) + \
                          torch.mul(torch.sparse.mm(user_item_final_matrix, _item_emb), user_token_emb)
        user_emb.append(user_emb_layer1)
        item_emb_layer1 = torch.sparse.mm(item_user_final_matrix, _user_emb) + \
                          torch.mul(torch.sparse.mm(item_user_final_matrix, user_token_emb), _item_emb)
        item_emb.append(item_emb_layer1)

        user_emb_layer2 = torch.sparse.mm(user_item_final_matrix, item_emb_layer1) + \
                          torch.mul(torch.sparse.mm(user_item_final_matrix, item_emb_layer1), user_token_emb)
        user_emb.append(user_emb_layer2)
        item_emb_layer2 = torch.sparse.mm(item_user_final_matrix, user_emb_layer1) + \
                          torch.mul(torch.sparse.mm(item_user_final_matrix, user_token_emb), item_emb_layer1)
        item_emb.append(item_emb_layer2)

        all_user_emb = torch.stack(user_emb, dim=1)
        final_user_emb = torch.mean(all_user_emb, dim=1)

        all_item_emb = torch.stack(item_emb, dim=1)
        final_item_emb = torch.mean(all_item_emb, dim=1)
        return final_user_emb, final_item_emb

    def forward(self, userid=None, systemid=None, entity_ids=None, r_list=None, token_embeds=None, token_speakers=None,
                attention_mask=None, user_convs_embeds=None, rec_labels=None,
                output_entity=False, use_rec_prefix=False, use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj(token_embeds)  # [bs,len,hs]

            unique_system_ids = list(Counter(systemid.cpu().numpy()).keys())
            sys_embeds = self.user_embedding[unique_system_ids]

            reshape_token_embeds = []
            for id in unique_system_ids:
                ones_mask = torch.ones_like(systemid).to(device)
                zeros_mask = torch.zeros_like(systemid).to(device)
                mask_mat = torch.where(systemid == id, ones_mask, zeros_mask)  # [bs]
                reshape_token_embeds.append(torch.sum(torch.mul(token_embeds, mask_mat.unsqueeze(1)), dim=0) / torch.nonzero(mask_mat).shape[0])
            reshape_token_embeds = torch.stack(reshape_token_embeds, dim=0)

            filter_token_embeds = torch.mul(self.p(self.trans(reshape_token_embeds)), reshape_token_embeds)  # 自过滤
            token_mapping = self.meta_decoder(filter_token_embeds).view(-1, self.hidden_size,self.hidden_size)  # (batch_size, hidden_size, hidden_size)
            enhanced_sys_embeds = torch.bmm(sys_embeds.unsqueeze(1), token_mapping).squeeze(1)  # (batch_size,n,hidden_size)

            all_token_embeds = self.user_embedding.clone()
            all_token_embeds[unique_system_ids] = enhanced_sys_embeds

            token_embeds_edge = torch.zeros_like(self.user_embedding).to(device)
            token_embeds_edge[unique_system_ids] = reshape_token_embeds

            #用文本对表示分别过滤
            user_token_rep = torch.matmul(all_token_embeds, self.sim_w1)#self.user_embedding
            item_rep = torch.matmul(self.node_embeds[self.itemid_set], self.sim_w2)
            user_token_emb1 = torch.nn.functional.normalize(user_token_rep, p=2, dim=1)
            item_emb1 = torch.nn.functional.normalize(item_rep, p=2, dim=1)
            sim_matrix = self.f(torch.matmul(user_token_emb1, item_emb1.T)).to(device) #self.f(

            user_token_topk = torch.topk(sim_matrix, self.topk)
            user_topk_values = torch.reshape(user_token_topk.values, [-1])
            user_topk_columns_id = torch.reshape(user_token_topk.indices, [-1,1])  # bs*topk
            user_topk_columns = torch.LongTensor(self.itemid_set)[user_topk_columns_id].to(device)

            user_all_rows = torch.from_numpy(np.reshape(np.arange(self.num_users), [-1, 1])).to(device)
            user_topk_rows = torch.reshape(user_all_rows.repeat(1, self.topk), [-1, 1])
            user_topk_indexs = torch.cat([user_topk_rows, user_topk_columns], 1).T  # torch.Size([2, 320])
            user_item_add_adj = torch.sparse_coo_tensor(user_topk_indexs, user_topk_values,[self.num_users, self.n_entity])

            item_topk_indexs = torch.cat([user_topk_columns, user_topk_rows], 1).T
            item_user_add_adj = torch.sparse_coo_tensor(item_topk_indexs, user_topk_values,[self.n_entity, self.num_users])

            user_item_final_adj_mat = self.adj_mat+user_item_add_adj
            item_user_final_adj_mat = self.adj_mat_t+item_user_add_adj

            norm_adj_mat = self.nor_sparse_matrix(user_item_final_adj_mat)
            norm_adj_mat_t = self.nor_sparse_matrix(item_user_final_adj_mat)

            all_user_embeds, all_entity_embeds = self.model_gcn(norm_adj_mat,
                                                                norm_adj_mat_t,
                                                                all_token_embeds, self.node_embeds,
                                                                token_embeds_edge)
            loss_re=self.gama * torch.mean(torch.pow((sim_matrix -self.adj_mat.to_dense()[:,self.itemid_set]), 2))

        if systemid is not None:
            system_embeds = all_user_embeds[systemid]
            batch_size = system_embeds.shape[0]

        prompt_len1 = 1

        prompt_embeds1 = system_embeds.unsqueeze(1)
        prompt_embeds1 = self.prompt_proj1(prompt_embeds1) + prompt_embeds1
        prompt_embeds1 = self.prompt_proj2(prompt_embeds1)
        prompt_embeds1 = prompt_embeds1.reshape(
            batch_size, prompt_len1, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batgch_size, n_head, prompt_len, head_dim)

        return prompt_embeds1, all_entity_embeds, loss_re

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)