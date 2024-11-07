import torch
from torch import nn
from collections import OrderedDict
import os


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class video_header(nn.Module):
    def __init__(self):
        super().__init__()
        self.vid_header = "Transf"

        if self.vid_header == "Transf":
            embed_dim = 512 # clip_state_dict["text_projection"].shape[1]
            context_length = 77 #clip_state_dict["positional_embedding"].shape[0]
            vocab_size = 49408 # clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = 512 # clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            # transformer_layers = len(
            #     set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()

        x_original = x
        seq_length = t
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original

        return x.mean(dim=1, keepdim=False)


class VideoCLIP(nn.Module):
    def load_text_embedding(self, config, future=False):
        text_emb = torch.load(os.path.join(config.data.data_path, config.data.dataset, 
                            'TextClasses-vitb16', 'text_feats_' + config.model.classifier + ('_future' if future else '') + '.pt'), 
                    weights_only=False)
        text_emb = text_emb.cuda()
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.t()
        return text_emb

    def __init__(self, config) :
        super(VideoCLIP, self).__init__()
        self.fusion_model = video_header()
        self.logit_scale = 100.0
        self.mlp_classifier = None
        if config.model.classifier == 'mlp':
            self.mlp_classifier = nn.Sequential(nn.Linear(512, 21 if config.data.dataset == 'THUMOS14' else 31), nn.Softmax(dim=-1))
        else:
            self.text_embedding = self.load_text_embedding(config, future=False)
            self.future_text_embedding = self.load_text_embedding(config, future=True) if config.data.future_steps > 0 else None
        self.future_relevance = torch.tensor(0.5)
        self.future_extra_layers = nn.Sequential(nn.LayerNorm(512), nn.ReLU())
        # self.future_relevance = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, image):
        fut_logits = None
        image_emb = self.fusion_model(image)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        if self.mlp_classifier is not None:
            logits = self.mlp_classifier(image_emb)
        else:
            logits = self.logit_scale * image_emb @ self.text_embedding
            if self.future_text_embedding is not None:
                fut_emd = self.future_extra_layers(image_emb)
                fut_logits = self.logit_scale * fut_emd @ self.future_text_embedding
                # concat_logits = torch.cat([logits.unsqueeze(-1), fut_logits.unsqueeze(-1)], dim=-1)  # Shape: (B, N_classes, 2)
                # logits = concat_logits.mean(dim=-1) # Shape: (B, N_classes)


        return logits, fut_logits