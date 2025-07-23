from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch

class AllFeedEmbeddingTransformer(nn.Module): 
    def __init__(self, 
                 obs_keys, 
                 context_length=5, 
                 d_model=64, 
                 n_head=8, 
                 dim_feedforward=256, 
                 input_slicing_indices = [0, 3, 67, 131]): 
        super(AllFeedEmbeddingTransformer, self).__init__()
        self.obs_keys = obs_keys
        self.input_slicing_indices = input_slicing_indices

        encoder_dict = nn.ModuleDict()
        for key in self.obs_keys: 
            if key == "agent_pos": 
                encoder_dict[key] = nn.Sequential(
                        nn.Linear(input_slicing_indices[1], d_model), # this only works for 2 cameras
                        nn.GELU(), 
                        nn.Linear(d_model, d_model), 
                        SingleFeedEmbeddingTransformer(
                            d_model=d_model, 
                            n_head=n_head, 
                            dim_feedforward=dim_feedforward, 
                            max_tokens=context_length, 
                            downsample_tokens=input_slicing_indices[1]
                        )
                    )
            else: 
                encoder_dict[key] = nn.Sequential(
                    SingleFeedEmbeddingTransformer(
                        d_model=d_model,
                        n_head=n_head, 
                        dim_feedforward=dim_feedforward,
                        max_tokens=context_length,
                    )
                )
        self.encoder_dict = encoder_dict
    
    def forward(self, obs): 
        # obs is of shape B, n_obs_steps, D_overall_embedding
        temp_dict = {}
        for i, key in enumerate(self.obs_keys): 
            temp_dict[key] = obs[:, :, self.input_slicing_indices[i]:self.input_slicing_indices[i+1]]

        # Pass the sliced inputs through their respective encoders
        encoder_outputs = []
        for key in self.obs_keys: 
            encoder_outputs.append(self.encoder_dict[key](temp_dict[key])[0])

        concatenated = torch.cat(encoder_outputs, dim=-1)
        return concatenated

class SingleFeedEmbeddingTransformer(nn.Module): 
    def __init__(self, 
                 d_model, 
                 n_head, 
                 dim_feedforward, 
                 max_tokens,
                 dropout=0.1, 
                 activation="gelu", 
                 layer_norm_eps=1e-6,
                 norm_first=True, 
                 batch_first=True, 
                 num_layers=8, 
                 downsample_tokens=None
                 ): 
        super(SingleFeedEmbeddingTransformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=batch_first
        )

        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_embed = nn.Parameter(torch.randn(1, 1 + max_tokens, d_model))        

        self.down_sample_tokens = downsample_tokens
        if self.down_sample_tokens: 
            self.downsample = nn.Linear(d_model, self.down_sample_tokens)  

    def forward(self, x): 
        batch_size, seq_len, embed_dim = x.shape 

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, x], dim=1)

        tokens = tokens + self.pos_embed

        out = self.transformer_encoder(tokens)

        if self.down_sample_tokens: 
            out = self.downsample(out)

        cls_out = out[:, 0, :]
        embeddings_out = out[:, 1:, :]

        return cls_out, embeddings_out
