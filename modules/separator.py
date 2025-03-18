import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchaudio.transforms as T
import torchsort

import laion_clap as clap


def init_parameters(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class ChannelPositionEmbedding(nn.Module):
    def __init__(self, C, H, W):
        super(ChannelPositionEmbedding, self).__init__()
        self.C = C
        self.H = H
        self.W = W

        # Learnable embeddings for height and width
        self.row_embed = nn.Parameter(torch.zeros(C, H), requires_grad=True)  # Shape: (C, H)
        self.col_embed = nn.Parameter(torch.zeros(C, W), requires_grad=True)  # Shape: (C, W)

        init.normal_(self.row_embed, mean=0, std=0.05)
        init.normal_(self.col_embed, mean=0, std=0.05)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.C and H == self.H and W == self.W, \
            f"Input tensor shape must match initialized dimensions (C={self.C}, H={self.H}, W={self.W}), but got shape ({C}, {H}, {W})."

        row_embeddings = self.row_embed.unsqueeze(2).expand(C, H, W)  # Shape: (C, H, W)
        col_embeddings = self.col_embed.unsqueeze(1).expand(C, H, W)  # Shape: (C, H, W)

        pos_embeddings = row_embeddings + col_embeddings  # Shape: (C, H, W)

        pos_embeddings = pos_embeddings.unsqueeze(0).expand(B, C, H, W)  # Shape: (B, C, H, W)
        return x + pos_embeddings


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SEBlock, self).__init__()
        assert channels // ratio * ratio == channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.GELU(),
            nn.Linear(channels // ratio, channels)
        )

    def forward(self, x: torch.Tensor):
        B, C, _, _ = x.shape
        y = F.adaptive_max_pool2d(x, 1).view(B, C)
        y = self.fc(y).view(B, C, 1, 1).expand_as(x)
        return y.sigmoid() * x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.senet = SEBlock(channels)

    def forward(self, x):
        x = x + self.layers(x)
        x = self.senet(x)
        return F.gelu(x)


class UNetBlock(nn.Module):
    def __init__(self, channels, inner_channels, submodule=None, outermost=False, innermost=False):
        super(UNetBlock, self).__init__()
        self.innermost, self.outermost = innermost, outermost
        scale = 1 if innermost else 2
        # down
        self.downconv = nn.Conv2d(channels, inner_channels, 4, stride=2, padding=1)
        self.downnorm = nn.BatchNorm2d(inner_channels)
        self.downresblock = ResBlock(inner_channels)
        self.dropout = nn.Dropout2d() if not innermost else nn.Identity()
        # mid
        self.submodule = submodule if submodule is not None else SEBlock(inner_channels)
        # up
        self.upresblock = ResBlock(inner_channels * scale)
        self.upconv = nn.ConvTranspose2d(inner_channels * scale, channels, 4, stride=2, padding=1)
        self.upnorm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_ = self.downresblock(F.gelu(self.downnorm(self.downconv(x))))
        x_ = self.dropout(x_)
        x_ = self.submodule(x_)
        x_ = self.upnorm(self.upconv(self.upresblock(x_)))
        x_ = F.gelu(x_)
        if x_.shape[2:] != x.shape[2:]:
            x_ = F.interpolate(x_, size=x.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x, x_], dim=1)


class CustomUNet(nn.Module):
    def __init__(self, M=6, H=-1, W=-1, NC=64):
        super(CustomUNet, self).__init__()
        unet = UNetBlock(8 * NC, 8 * NC, innermost=True)
        unet = UNetBlock(4 * NC, 8 * NC, submodule=unet)
        unet = UNetBlock(2 * NC, 4 * NC, submodule=unet)
        unet = UNetBlock(1 * NC, 2 * NC, submodule=unet)
        unet = UNetBlock(1 * NC, 1 * NC, submodule=unet, outermost=True)
        self.unet = unet
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, M, 3, padding=1),
            ChannelPositionEmbedding(M, H, W),
            nn.BatchNorm2d(M),
            nn.GELU(),
            nn.Conv2d(M, NC, 3, padding=1),
            nn.BatchNorm2d(NC),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * NC, NC, 3, padding=1),
            nn.BatchNorm2d(NC),
            nn.GELU(),
            nn.Conv2d(NC, NC, 3, padding=1),
            nn.BatchNorm2d(NC),
            nn.GELU(),
            nn.Conv2d(NC, M, 1)
        )

        self.apply(init_parameters)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.unet(x)
        x = self.out_conv(x)
        return x


class MixITBaseModel(nn.Module):
    def __init__(self, M=6, L=8, sr=16000, clap=None):
        super(MixITBaseModel, self).__init__()
        self.M = M
        self.L = L
        self.sr = sr

        self.stft = T.Spectrogram(power=None)
        self.istft = T.InverseSpectrogram()

        H, W = self.calc_H_W()
        self.unet = CustomUNet(M=M, H=H, W=W)

        self.resample = T.Resample(16000, 48000)
        self.clap = clap
        self.tau1 = 100.
        self.tau2 = nn.Parameter(torch.log(torch.tensor(1 / .07)))

    def calc_H_W(self):
        input_length = self.L * self.sr
        pad = self.stft.pad
        win_length = self.stft.win_length
        hop_length = self.stft.hop_length
        n_fft = self.stft.n_fft

        if self.stft.center:
            num_frames = (input_length + 2 * pad - win_length + n_fft) // hop_length + 1
        else:
            num_frames = (input_length + 2 * pad - win_length) // hop_length + 1
        num_freqs = n_fft // 2 + 1
        return num_freqs, num_frames

    def align_loss(self, audio_embedding, text_embedding):
        B, M, D = audio_embedding.shape

        text_embedding = F.normalize(text_embedding, dim=-1)
        audio_embedding = F.normalize(audio_embedding, dim=-1)

        logits = (torch.matmul(audio_embedding, text_embedding.transpose(1, 2)) * self.tau1).softmax(dim=-1)
        text_embedding = torch.matmul(logits, text_embedding)

        audio_embedding = audio_embedding.reshape(B, M, D)
        text_embedding = F.normalize(text_embedding.reshape(B, M, D), dim=-1)
        sim1 = torch.matmul(audio_embedding, text_embedding.transpose(1, 2)) * self.tau2.exp()
        sim2 = sim1.transpose(1, 2)
        gt = torch.arange(self.M).unsqueeze(0).expand(B, M).cuda()
        align = F.cross_entropy(sim1, gt) + F.cross_entropy(sim2, gt)
        align = align / 2.
        return align

    def rank_loss(self, reference: torch.Tensor):
        def spearmanr(pred, target):
            pred = torchsort.soft_rank(pred)
            pred = pred - pred.mean()
            pred = pred / pred.norm()
            target = target - target.mean()
            target = target / target.norm()
            return (pred * target).sum()

        B, M = reference.shape

        target = self.M - torch.arange(self.M, device=reference.device).unsqueeze(0).expand(B, -1).float()
        spearman = spearmanr(reference, target)
        return 1. - spearman

    def forward(self, audio: torch.Tensor, text_embedding: torch.Tensor = None):
        L = audio.shape[-1]

        stft = self.stft(audio)  # [B, 1, F, T]
        spectrogram = stft.abs().pow(2)  # [B, 1, F, T]
        masks = F.sigmoid(self.unet(spectrogram))  # [B, M, F, T]
        masked_stft = masks * stft  # [B, M, F, T]
        separated_audio = self.istft(masked_stft, length=L)  # [B, M, L]

        # Consistency Projection
        audio = audio.expand_as(separated_audio)  # [B, M, L]
        s = torch.sum(separated_audio, dim=1, keepdim=True).expand_as(audio)  # [B, M, L]
        separated_audio = separated_audio + (audio - s) / self.M

        align = 0.
        rank = 0.
        B = audio.shape[0]
        separated_audio_ = self.resample(separated_audio.reshape(-1, L))  # [B * M, T]
        audio_embedding = self.clap.get_audio_embedding_from_data(separated_audio_, use_tensor=True).reshape(B, self.M,-1)

        if text_embedding is not None:
            align = self.align_loss(audio_embedding, text_embedding)
            audio = self.resample(audio.reshape(-1, L))
            orig_audio_embedding = self.clap.get_audio_embedding_from_data(audio, use_tensor=True)  # [B * M, 512]
            orig_audio_embedding = (orig_audio_embedding).reshape(B, self.M, -1)  # [B, M, 768]
            similarity = F.cosine_similarity(audio_embedding, orig_audio_embedding, dim=-1)  # [B, M]
            rank = self.rank_loss(similarity)

        ret = {}
        ret['align'] = align
        ret['rank'] = rank
        ret['separated_audio'] = separated_audio
        ret['hidden_coeff'] = audio_embedding  # [B, M, N, F]
        return ret


class Separator(nn.Module):
    def __init__(self, M=6, L=8, sr=16000, device='cuda', use_unperm=True):
        super(Separator, self).__init__()
        clap_model = clap.CLAP_Module(enable_fusion=False).cuda().eval()
        clap_model.load_ckpt()

        self.clap = clap_model
        self.M = M
        self.sr, self.L = sr, L
        self.model = MixITBaseModel(M=M, L=L, sr=sr, clap=self.clap)
        self.use_unperm = use_unperm
        if use_unperm:
            self.unperm = self.make_unperm_matrix().to(device)  # [2^M, 2, M]

        self.show_params()

    def show_params(self):
        params = sum(p.numel() for n, p in self.named_parameters() if 'clap' not in n)
        print(f'Parameters: {params / 1_000_000:.2f} M')

    def load_clap(self):
        clap_model = clap.CLAP_Module(enable_fusion=False, ).cuda().eval()
        clap_model.load_ckpt()
        clap_model.requires_grad_(False)
        self.clap = clap_model
        self.model.clap = clap_model

    def make_perm_matrix(self, bsz):
        shift = torch.randint(1, bsz, (1,))
        perm_matrix = torch.zeros(bsz, bsz)
        perm_matrix[torch.arange(bsz), torch.arange(bsz) - shift] = 1
        return perm_matrix

    def make_unperm_matrix(self):
        binary_combinations = torch.tensor([[int(x) for x in format(i, f'0{self.M}b')] for i in range(1, 2 ** self.M)])
        result = torch.zeros((2 ** self.M - 1, 2, self.M), dtype=torch.float32)
        result[:, 0, :] = binary_combinations
        result[:, 1, :] = 1 - binary_combinations
        return nn.Parameter(result, requires_grad=False)

    def snr_loss(self, gt1, gt2, pred):
        criteria = auraloss.time.SISDRLoss(reduction=None)
        unpermed = torch.einsum('ijk,lkm->lijm', self.unperm, pred)  # [B, 2^M, 2, T]
        unpermed1, unpermed2 = unpermed[:, :, 0, :], unpermed[:, :, 1, :]  # [B, 2^M, T]
        gt1, gt2 = gt1.expand_as(unpermed1), gt2.expand_as(unpermed2)
        loss1, loss2 = criteria(unpermed1.contiguous(), gt1.contiguous()), criteria(unpermed2.contiguous(),
                                                                                    gt2.contiguous())  # [B, 2^M]
        loss, _ = torch.min(loss1 + loss2, dim=1)
        loss = torch.mean(loss)
        return loss

    def encode_text(self, labels):
        return self.clap.get_text_embedding(labels, use_tensor=True)

    def encode_audio(self, audio):
        audio = self.model.resample(audio)
        audio_embedding = self.clap.get_audio_embedding_from_data(audio, use_tensor=True)
        return audio_embedding

    def forward(self, audio: torch.Tensor, labels=None, device='cuda'):
        if not self.training or labels is None:
            return self.model(audio.unsqueeze(1))

        labels = torch.stack([self.clap.get_text_embedding(l, use_tensor=True) for l in labels], dim=1).to(device)
        perm_matrix = self.make_perm_matrix(audio.shape[0]).to(device)
        perm_audio = perm_matrix @ audio
        mixed_audio = audio + perm_audio
        perm_label = (perm_matrix @ labels.reshape(labels.shape[0], -1)).reshape(labels.shape[0], -1, 512)
        mixed_label = torch.cat((labels, perm_label), dim=1)  # [B, M, 512]

        output = self.model(mixed_audio.unsqueeze(1), mixed_label)  # [B, M, T]
        loss = self.snr_loss(audio.unsqueeze(1), perm_audio.unsqueeze(1), output['separated_audio'])

        output['mixit_loss'] = loss
        return output
