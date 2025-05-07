import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNN, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_prob)]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SimpleNNWithBatchNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNNWithBatchNorm, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_prob)]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TransformerGAPClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers=1, num_heads=4):
        super(TransformerGAPClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_size * 4,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch_size, num_patches, hidden_size]
        x = self.transformer(x)  # Let patches interact
        x = x.mean(dim=1)  # Global Average Pooling over patches
        out = self.fc(x)
        return out



class VisionModel(nn.Module):
    def __init__(self, hidden_size=512, num_heads_img=8, num_layers_img=6):
        super(VisionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads_img = num_heads_img
        self.num_layers_img = num_layers_img

        self.img_trans = MultiLayerCrossModalAttention(num_layers_img, hidden_size, num_heads_img)
        self.crossmodal_attention_pooling = AttentionPooling(hidden_dim=hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

    def forward(self, img_features):
        proj_img, _ = self.img_trans(img_features, img_features, img_features)

        # Attention pooling
        pooled_features = self.crossmodal_attention_pooling(proj_img)

        # Fully connected layer
        output = self.fc(pooled_features)

        return output


# cross-modal attention from the survival paper
class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super(CrossModalMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len

        # Positional Embeddings
        self.positional_embedding = nn.Parameter(torch.randn(seq_len, embed_dim))

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Add positional embeddings to the inputs
        query = query + self.positional_embedding
        key = key + self.positional_embedding
        value = value + self.positional_embedding

        # Standard multi-head attention
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Residual connection and normalization
        output = self.out_proj(attended_values)
        output = self.layer_norm(output + attended_values)

        # Feedforward block with residual connection
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)

        return output, attention_weights


class MultiLayerCrossModalAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super(MultiLayerCrossModalAttention, self).__init__()
        self.layers = nn.ModuleList([CrossModalMultiHeadAttention(embed_dim,
                                                                  num_heads, seq_len=512) for _ in range(num_layers)])

    def forward(self, query, key, value):
        for layer in self.layers:
            query, attn_weights = layer(query, key, value)
        return query, 1


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, transformer_outputs):
        # transformer_outputs shape: (batch_size, seq_length, hidden_dim)
        scores = torch.matmul(transformer_outputs, self.attention_weights)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * transformer_outputs, dim=1)
        return context_vector

class ViTBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module, unfreeze_last_n: int = 0):
        super(ViTBinaryClassifier, self).__init__()

        self.patch_embedding = pretrained_model.patch_embedding
        self.blocks = pretrained_model.blocks  # nn.ModuleList of TransformerBlock
        self.norm = pretrained_model.norm
        self.attention_pooling = pretrained_model.attention_pooling

        hidden_size = pretrained_model.norm.normalized_shape[0]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

        # Freeze everything by default
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks (if N > 0)
        if unfreeze_last_n > 0:
            for block in self.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Always train the classifier
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.patch_embedding(x)
        attn_weights = []
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.attention_pooling(x)
        out = self.classifier(x)
        return out
