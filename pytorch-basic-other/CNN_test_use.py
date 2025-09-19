import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(128, 10)  # 10 个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc2(x)            # 最后一层输出
        return x

def create_mnist_like_digit(digit, save_path=None):
    # 创建28x28黑色画布
    img = Image.new('L', (28, 28), color=0)  # 0=黑色
    draw = ImageDraw.Draw(img)
    
    # 根据数字绘制不同的形状
    if digit == 0:
        draw.ellipse((4, 4, 24, 24), outline=255, width=2)  # 白色圆圈
    elif digit == 1:
        draw.line((14, 4, 14, 24), fill=255, width=3)
    elif digit == 2:
        # 绘制数字2
        draw.line((6, 10, 14, 4), fill=255, width=2)
        draw.line((14, 4, 22, 4), fill=255, width=2)
        draw.line((22, 4, 22, 14), fill=255, width=2)
        draw.line((22, 14, 6, 24), fill=255, width=2)
        draw.line((6, 24, 22, 24), fill=255, width=2)
    elif digit == 3:
        # 绘制数字3
        draw.line((6, 4, 22, 4), fill=255, width=2)
        draw.line((22, 4, 22, 10), fill=255, width=2)
        draw.line((22, 10, 6, 17), fill=255, width=2)
        draw.line((14, 17, 22, 17), fill=255, width=2)
        draw.line((22, 17, 22, 24), fill=255, width=2)
        draw.line((6, 24, 22, 24), fill=255, width=2)
    elif digit == 4:
        # 绘制数字4 
        draw.line([(6, 4), (6, 12)], fill=255, width=2)  # 使用列表传递坐标
        draw.line([(6, 12), (18, 12)], fill=255, width=2)
        draw.line([(18, 4), (18, 24)], fill=255, width=2)
    elif digit == 5:
        # 绘制数字5
        draw.line([(6, 4), (22, 4)], fill=255, width=2)
        draw.line([(6, 4), (6, 12)], fill=255, width=2)
        draw.line([(6, 12), (22, 12)], fill=255, width=2)
        draw.line([(22, 12), (22, 24)], fill=255, width=2)
        draw.line([(6, 24), (22, 24)], fill=255, width=2)
    elif digit == 6:
        # 绘制数字6
        draw.ellipse((4, 4, 24, 16), outline=255, width=2)  # 上半部分
        draw.line([(14, 16), (14, 24)], fill=255, width=2)
        draw.line([(6, 24), (22, 24)], fill=255, width=2)
        draw.line([(6, 16), (6, 24)], fill=255, width=2)
        draw.line([(22, 16), (22, 24)], fill=255, width=2)
    elif digit == 7:
        # 绘制数字7
        draw.line([(6, 4), (22, 4)], fill=255, width=2)
        draw.line([(22, 4), (6, 24)], fill=255, width=2)
    elif digit == 8:
        # 绘制数字8
        draw.arc((4, 2, 24, 14), 0, 360, fill=255, width=2)
        draw.arc((4, 14, 24, 26), 0, 360, fill=255, width=2)
    elif digit == 9:
        # 绘制数字9
        draw.arc((4, 2, 24, 14), 0, 360, fill=255, width=2)
        # draw.line([(14, 14), (14, 4)], fill=255, width=2)
        draw.line([(14, 14), (22, 14)], fill=255, width=2)
        draw.line([(22, 14), (22, 24)], fill=255, width=2)
        draw.line([(6, 24), (22, 24)], fill=255, width=2)
    else:
        raise ValueError("digit must be between 0 and 9")
    
    if save_path:
        img.save(save_path)
    
    return img

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
module = SimpleCNN()
module.load_state_dict(torch.load("pytorch-basic-other\\CNN_test_model_statedict.pth", map_location=device))
module.to(device)
module.eval()

transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化，根据实际训练情况调整
])

imgs_tensor = []
for i in range(10):
    img = create_mnist_like_digit(i)
    img.save(f"pytorch-basic-other\\test_figure\\{i}.jpg")
    img = transform(img).unsqueeze(0)
    imgs_tensor.append(img)

# 将所有张量移动到相同的设备上
batch_tensor = torch.cat(imgs_tensor, dim=0).to(device)

with torch.no_grad():
    output = module(batch_tensor)
    print(f"output shape: {output.shape}")
    v, i = torch.max(output, dim=1)
    print(f"Predicted digits: {i.tolist()}")