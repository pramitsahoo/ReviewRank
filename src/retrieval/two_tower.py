import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim = 128, mlp_dims = [256, 128]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        # dynamic MLP construction based on mlp_dims
        layers = []
        in_dim = embedding_dim * 2  # user emb + history pooled emb
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers) # f(*[a, b]) = f(a, b)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # xavier initialization for weights
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, user_id, user_history): # (B,), (B, S)
        user_emb = self.user_embedding(user_id)  # (B, D)

        hist_emb = self.item_embedding(user_history)  # (B, S, D) B users, S items in history, D embedding dim for each item
        mask = (user_history != 0).unsqueeze(-1).float()  # (B, S, 1)
        hist_sum = (hist_emb * mask).sum(dim=1)  # (B, D)
        hist_count = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        hist_pooled = hist_sum / hist_count  # (B, D) mean pooling

        x = torch.cat([user_emb, hist_pooled], dim=-1)  # (B, 2D)
        x = self.mlp(x)
        return F.normalize(x, dim=-1) # l2 normalization for cosine similarity


class ItemTower(nn.Module):
    def __init__(self, num_items, num_categories, embedding_dim = 128, category_dim = 32, mlp_dims = [128]):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, category_dim, padding_idx=0)

        layers = []
        in_dim = embedding_dim + category_dim
        for out_dim in mlp_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, item_id, category_id): # (B,), (B,)
        item_emb = self.item_embedding(item_id)  # (B, D)
        cat_emb = self.category_embedding(category_id)  # (B, C)
        x = torch.cat([item_emb, cat_emb], dim=-1)  # (B, D+C)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embedding_dim=128, category_dim=32, 
                 user_mlp_dims=[256, 128], item_mlp_dims=[128], temperature=0.05):
        super().__init__()
        self.user_tower = UserTower(num_users, num_items, embedding_dim, user_mlp_dims)
        self.item_tower = ItemTower(num_items, num_categories, embedding_dim, category_dim, item_mlp_dims)
        self.temperature = temperature

    def forward(self, user_id, user_history,
                item_id, category_id):
        user_embeds = self.user_tower(user_id, user_history)
        item_embeds = self.item_tower(item_id, category_id)
        return user_embeds, item_embeds

    def compute_loss(self, user_embeds, item_embeds, temperature=None): # InfoNCE loss with in-batch negatives
        t = temperature if temperature is not None else self.temperature
        # in-batch negatives: each row for a user, each column for an item, the correct item for each user is on the diagonal
        sim = user_embeds @ item_embeds.T / t  # (B, B)
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)
