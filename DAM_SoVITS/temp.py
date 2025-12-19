import torch
from torch.nn import functional as F

mask_ratio = 0.9
x = torch.tensor([[1,2,3,0,0], [1,2,3,4,0], [1,2,3,4,5]])
y = torch.tensor([[1,-1,3,4,3,0], [5,6,-1,8,0,0], [1,2,3,-1,4,5]])
x_lens = torch.tensor([3, 4, 5])
y1_lens = torch.tensor([1, 2, 3])
y2_lens = torch.tensor([3, 1, 2])
B = x.size(0)
x_len = x.size(1)
y_len = y.size(1)
src_len = x_len+y_len+1
y_lens = y1_lens+y2_lens+1
device = x.device


indices = torch.arange(y_len).unsqueeze(0)
y_mask = (indices >= y1_lens.unsqueeze(1)+1) & (indices < y_lens.unsqueeze(1)-1)
y_mask = torch.bernoulli(torch.full(y_mask.shape, mask_ratio) * y_mask).bool()

print(y_mask)

# False是关注
indices = torch.arange(src_len).unsqueeze(0)
x_attn_mask = (indices >= x_lens.unsqueeze(1))
x_attn_mask = x_attn_mask.unsqueeze(1).expand(-1, x_len, -1)
print("x_attn_mask", x_attn_mask.shape)

emo_attn_mask = torch.ones((B, 1, src_len), dtype=torch.bool)
emo_attn_mask[:, :, x_len] = 0
print("emo_attn_mask", emo_attn_mask.shape)

y_attn_mask = y_mask.clone()
indices = torch.arange(y_len).unsqueeze(0)
y_attn_mask[indices >= y_lens.unsqueeze(1)] = True
y_attn_mask = F.pad(y_attn_mask, (x_len+1, 0), value=False)
y_attn_mask[:, :x_len] = x_attn_mask[:, 0, :x_len]
y_attn_mask = y_attn_mask.unsqueeze(1).expand(-1, y_len, -1).clone()
indices = torch.arange(src_len, device=device).unsqueeze(0).unsqueeze(0)
mask = indices > (x_len + 1 + y1_lens).unsqueeze(1).unsqueeze(2)
mask = mask.expand_as(y_attn_mask).clone()
indices = torch.arange(y_len).unsqueeze(0)
mask[indices > y1_lens.unsqueeze(1)] = False
y_attn_mask[mask] = True
print("y_attn_mask", y_attn_mask.shape)

attn_mask = torch.concat([x_attn_mask, emo_attn_mask, y_attn_mask], dim=1)
print("attn_mask", attn_mask.shape)