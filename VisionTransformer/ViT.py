import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=6
        )
        
        self.fc = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.proj(x)  # 将输入图像切分为多个patch并进行投影 [2, 3, 224, 224] 变为 [2, 256, 14, 14]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 将patch展平并转置维度  [2, 256, 14, 14] -> ([2, 256, 196]) -> ([2, 196, 256]) [B,L,C]  切割后与Bert里面的Transformer输入就类似了 [B,L,d]  # V: (bs, len1, dim1)  # A: (bs, len2, dim2) # T: (bs, len3, dim3)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展cls_token为每个样本的初始位置编码 ([1, 1, 256]) -> ([2, 1, 256])
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls_token和patch连接 ([2, 196, 256]) -> ([2, 197, 256])
        
        x = x + self.pos_embed  # 加上位置编码  self.pos_embed.shape=([1, 197, 256])
        
        x = self.transformer(x)  # 经过Transformer编码器
        cls_token = x[:, 0]  # 提取cls_token特征  ([2, 197, 256])[:,0] -> ([2, 256])
        
        x = self.fc(cls_token)  # 经过全连接层得到分类结果
        
        return x

# 创建ViT模型实例
img_size = 224  # 输入图像大小
patch_size = 16  # patch大小
num_classes = 10  # 类别数
dim = 256  # 特征维度
model = ViT(img_size, patch_size, num_classes, dim)

# 创建输入张量
input_tensor = torch.randn(2, 3, img_size, img_size)  # 假设输入大小为 [2, 3, 224, 224]

# 前向传播
output = model(input_tensor)
print("输出张量大小:", output.size())
