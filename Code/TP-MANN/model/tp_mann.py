from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from model.utils import MLP, LayerNorm, OptionalLayer


class Tpmann(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(Tpmann, self).__init__()
        self.input_module = InputModule(config)
        self.update_module = UpdateModule(config=config)
        self.inference_module = InferenceModule(config=config)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        story_embed, query_embed = self.input_module(story, query)
        tpr = self.update_module(story_embed)
        logits = self.inference_module(query_embed, tpr)
        return logits


class InputModule(nn.Module):
    def __init__(self, config):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story, query):
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed[:query_embed.shape[1]])
        return sentence_sum, query_sum


class UpdateModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(UpdateModule, self).__init__()
        self.role_size = config["role_size"]
        self.ent_size = config["entity_size"]
        self.hidden_size = config["hidden_size"]
        self.symbol_size = config["symbol_size"]
        self.layers = config["layers"]
        self.e = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        
        # learn to initialize
        self.register_parameter('fwm', torch.nn.Parameter(torch.Tensor(self.ent_size, self.role_size, self.ent_size).cuda()))
        stdev = 1 / (np.sqrt(self.ent_size + self.ent_size))
        nn.init.uniform_(self.fwm, -stdev, stdev)

        # write_size = self.ent_size + self.role_size + self.ent_size + 1 
        # self.W_write = nn.ModuleList([nn.Linear(self.symbol_size, write_size)]*(self.layers))

    def forward(self, sentence_embed: torch.Tensor) -> torch.Tensor:
        batch_size = sentence_embed.size(0)

        e1, e2 = [module(sentence_embed) for module in self.e]  # (b, s, e) -> (b, s, ent_size)
        r1, r2, r3 = [module(sentence_embed) for module in self.r]  # (b, s, e) -> (b, s, role_size)
        partial_add_w = torch.einsum('bsr,bsf->bsrf', r1, e2)  # (b, s, role_size, ent_size)
        partial_add_b = torch.einsum('bsr,bsf->bsrf', r3, e1)  # (b, s, role_size, ent_size)

        # TPR-RNN steps
        tpr = self.fwm.clone().repeat(batch_size, 1, 1, 1)  # (b, k1_dim, k2_dim, v_dim)

        for layer in range(self.layers):
            w_hat_tmp = torch.einsum('bse,bsr->bser', e1, r1)  # (b, s, ent_size, role_size) # like k1_k2
            w_hat = torch.einsum('bser,berf->bsf', w_hat_tmp, tpr)  # (b, s, ent_size)
            partial_remove_w = torch.einsum('bsr,bsf->bsrf', r1, w_hat)  # (b, s, role_size, ent_size)
            write_op = partial_add_w - partial_remove_w  # (b, s, role_size, ent_size)
            tpr = tpr + torch.einsum('bse,bsrf->berf', e1, write_op)  # (b, ent_size, role_size, ent_size)

            m_hat_tmp = torch.einsum('bse,bsr->bser', e1, r2)  # (b, s, ent_size, role_size)
            m_hat = torch.einsum('bser,berf->bsf', m_hat_tmp, tpr)  # (b, s, ent_size)
            partial_remove_m = torch.einsum('bsr,bsf->bsrf', r2, m_hat)  # (b, s, role_size, ent_size)
            partial_add_m = torch.einsum('bsr,bsf->bsrf', r2, w_hat)  # (b, s, role_size, ent_size)
            move_op = partial_add_m - partial_remove_m  # (b, s, role_size, ent_size)
            tpr = tpr + torch.einsum('bse,bsrf->berf', e1, move_op)  # (b, ent_size, role_size, ent_size)
            
            b_hat_tmp = torch.einsum('bse,bsr->bser', e2, r3)  # (b, s, ent_size, role_size)
            b_hat = torch.einsum('bser,berf->bsf', b_hat_tmp, tpr)  # (b, s, ent_size)
            partial_remove_b = torch.einsum('bsr,bsf->bsrf', r3, b_hat)  # (b, s, role_size, ent_size)
            backlink_op = partial_add_b - partial_remove_b  # (b, s, role_size, ent_size)
            tpr = tpr + torch.einsum('bse,bsrf->berf', e2, backlink_op)  # (b, ent_size, role_size, ent_size)

            f_norm = tpr.view(tpr.shape[0], -1).norm(dim=-1)
            fnorm = torch.relu(f_norm - 1) + 1
            tpr = tpr / fnorm.view(-1, 1, 1, 1)

        return tpr


class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], 9))
        nn.init.xavier_uniform_(self.Z.data)

        self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, query_embed, tpr):
        e1, e2 = [module(query_embed) for module in self.e]
        r1, r2, r3 = [module(query_embed) for module in self.r]

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, tpr))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, tpr))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, tpr))

        step_sum = i1 + i2 + i3
        logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
        return logits
