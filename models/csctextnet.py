"""
Codes are partially from https://github.com/raoyongming/DenseCLIP
"""
import math
from collections import OrderedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from clip import tokenize

from nets.stdcnet import STDCNet1446, STDCNet813
# from modules.bn import InPlaceABNSync as BatchNorm2d
BatchNorm2d = nn.SyncBatchNorm
# BatchNorm2d = nn.BatchNorm2d
from models.model_stages import ConvBNReLU, AttentionRefinementModule, FeatureFusionModule, BiSeNetOutput, BiSeNet


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPTextContextEncoder(nn.Module):
    arch_settings = {
        'RN50': {
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12,
            'embed_dim': 1024,
        },
        'RN101': {
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12,
            'embed_dim': 512,
        },
    }

    def __init__(self, 
                 context_length=22,
                 encoder_type='RN50',
                 pretrained=None,
                 **kwargs):
        super().__init__()
        assert encoder_type in self.arch_settings
        self.pretrained = pretrained

        self.context_length = context_length
        self.vocab_size = self.arch_settings[encoder_type]['vocab_size']
        transformer_width = self.arch_settings[encoder_type]['transformer_width']
        transformer_heads = self.arch_settings[encoder_type]['transformer_heads']
        transformer_layers = self.arch_settings[encoder_type]['transformer_layers']
        self.embed_dim = self.arch_settings[encoder_type]['embed_dim']

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, self.embed_dim))
        self.init_weights()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            if len(u) == 0 and len(w) == 0:
                print('CLIP checkpoint successfully loaded!')
            else:
                print(u, w, 'are misaligned params in text encoder')
            assert len(u) == 0, f"Missing keys: {u}"

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        """
        Input
            text: (n_class, context_length for class label)
            context: (n_class, context_length for learnable context, transformer_width)
        Ouput
            context-aware text embeddingds: (1, n_class, embed_dim)
        """
        x_text = self.token_embedding(text)  # n_clas, n_text, C
        K1, N1, C1 = x_text.shape
        K2, N2, C2 = context.shape
        assert (K1 == K2) and (C1 == C2)

        eos_indx = text.argmax(dim=-1) + N2

        x = torch.cat([x_text[:, 0:1], context, x_text[:, 1:]], dim=1).reshape(K1, N1+N2, C1)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2).contiguous()  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2).contiguous()  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(1, K1, self.embed_dim)
        return x


class TextContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(TextContextPath, self).__init__()

        self.backbone_name = backbone

        # Text
        text_encoder_config = {
            'backbone': 'RN50',
            'pretrain_model': './checkpoints/CLIP/RN50.pt',
            'dataset_name': 'cityscapes',
            'label_context_length': 5,
            'learn_context_length': 8,
            'context_mode': 'CSC',
        }
        CLASSES_DICT = {
            'cityscapes': ('road', 'sidewalk', 'building', 'wall', 'fence', 
                           'pole', 'traffic light', 'traffic sign', 'vegetation',
                           'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                           'bus', 'train', 'motorcycle', 'bicycle')
        }
        self.CLASSES = CLASSES_DICT[text_encoder_config['dataset_name']]
        self.num_classes = len(self.CLASSES)
        self.label_context_length = text_encoder_config['label_context_length']
        self.learn_context_length = text_encoder_config['learn_context_length']
        self.label_texts = torch.cat([tokenize(c, context_length=self.label_context_length) for c in self.CLASSES]).requires_grad_(False)  # n_class, label_context_length
        self.token_embed_dim = 512
        self.context_mode = text_encoder_config['context_mode']
        if self.context_mode == 'UC':
            self.contexts = nn.Parameter(torch.randn(self.learn_context_length, self.token_embed_dim))
        elif self.context_mode == 'CSC':
            self.contexts = nn.Parameter(torch.randn(self.num_classes, self.learn_context_length, self.token_embed_dim))
        else:
            raise NotImplementedError
        self.text_encoder = CLIPTextContextEncoder(context_length=self.label_context_length + self.learn_context_length, encoder_type=text_encoder_config['backbone'], pretrained=text_encoder_config['pretrain_model'])

        # Context
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes + self.num_classes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes + self.num_classes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes + self.num_classes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes + self.num_classes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        # Text Path
        feat = feat32
        B, C, H, W = feat.shape

        label_texts = self.label_texts.to(feat.device)
        contexts = self.contexts.to(feat.device)
        if contexts.dim() == 2:
            contexts = contexts.unsqueeze(0).expand(self.num_classes, -1, -1)  # n_class, context_context_length, toekn_embed_dim
        text_embeddings = self.text_encoder(label_texts, contexts).expand(B, -1, -1).type(feat.dtype).to(feat.device)  # batch_size, full_context_length, embed_dim

        feat_normalize = F.normalize(feat, dim=1, p=2).view(B, C, H * W)
        text_embeddings_normalize = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.bmm(text_embeddings_normalize, feat_normalize).view(B, text_embeddings_normalize.shape[1], H, W)
        feat32 = torch.cat([feat32, score_map], dim=1)

        # Context Path
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up, score_map  # x8, x16, x32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, fix_params = [], [], []
        for name, module in self.named_modules():
            if 'text_encoder' in name:
                fix_params += list(module.parameters())
            else:
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    wd_params.append(module.weight)
                    if not module.bias is None:
                        nowd_params.append(module.bias)
                elif isinstance(module, BatchNorm2d):
                    nowd_params += list(module.parameters())
        return wd_params, nowd_params, fix_params


class CSCTextNet(BiSeNet):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(CSCTextNet, self).__init__(backbone, n_classes, pretrain_model, use_boundary_2, use_boundary_4, use_boundary_8, use_boundary_16, use_conv_last, heat_map, *args, **kwargs)
        self.cp = TextContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

    def forward(self, x):
        H, W = x.size()[2:]

        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16, score_map = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)

        feat_out_sp4 = self.conv_out_sp4(feat_res4)

        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        feat_outscoremap = F.interpolate(score_map / 0.07, (H, W), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_outscoremap

        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8, feat_outscoremap

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp8, feat_outscoremap

        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16, feat_out32, feat_outscoremap

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params, fix_params = [], [], [], [], []
        for name, child in self.named_children():
            child_params = child.get_params()
            if len(child_params) == 2:
                child_wd_params, child_nowd_params = child_params
            elif len(child_params) == 3:
                child_wd_params, child_nowd_params, child_fix_params = child_params
                fix_params += child_fix_params
            else:
                raise

            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params, fix_params


if __name__ == "__main__":
    # text_encoder = CLIPTextContextEncoder(ontext_length=22, encoder_type='RN50', pretrained='./checkpoints/CLIP/RN50.pt')
    # text_encoder.cuda()
    # text_encoder.eval()
    # text = torch.randint(0, 255, (19, 5)).cuda()
    # context = torch.randn(19, 17, 512).cuda()
    # out = text_encoder(text, context)
    # print(out.shape)

    net = CSCTextNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32, score_map = net(in_ten)
    print(out.shape)
