

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.cm
from torchinfo import summary
from torchvision.ops import misc as misc_nn_ops
from torch import nn
from torchvision.models.detection.backbone_utils import mobilenet_backbone
import torchvision.models.mobilenet
import copy
import cv2
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

module_name = []
p_in = []
p_out = []

# 定义hook_fn，顾名思义就是把数值从forward计算过程中“勾出”来
def hook_fn(module, inputs, outputs):
    module_name.append(module.__class__)
    print(module)
    p_in.append(inputs)
    p_out.append(outputs)


#%% 加载模型，注册钩子函数
qq=11
qq=mobilenet_backbone("mobilenet_v3_large",pretrained=None,fpn=True)
print(qq)
#model.rpn.head.cls_logits.register_forward_hook(hook_fn) # faster_rcnn

## Retinanet
# model = models.detection.retinanet_resnet50_fpn(pretrained=True)
# model.head.classification_head.cls_logits.register_forward_hook(hook_fn)

## SSD300 VGG16
# model = models.detection.ssd300_vgg16(pretrained=True)
# model.head.classification_head.register_forward_hook(hook_fn)

# print(summary(model, (1,3,300,300), verbose=1))
#%%
# 导入一张图像
Name="scratches_16"
img_file = "visual/Image/"+Name+'.jpg'
img = Image.open(img_file)
ori_img = img.copy()

# 必要的前处理
transform = transforms.Compose([
                transforms.Resize((448,448)),
				transforms.ToTensor(),
				transforms.Normalize(
					[0.485, 0.456, 0.406],
					[0.229, 0.224, 0.225])])
img = transform(img).unsqueeze(0)  # 增加batch维度

model.to(device)
img = img.to(device) # torch.Size([1, 3, 720, 1280])
# EVAL模式
model.eval()
with torch.no_grad():
    model(img)

#%% 特征可视化
def show_feature_map(img_src, conv_features,k):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = Image.open(img_file).convert('RGB')
    height, width = img.size
    conv_features = conv_features.cpu()
    heat = conv_features.squeeze(0)#降维操作,尺寸变为(C,H,W)
    heatmap = torch.mean(heat,dim=0)#对各卷积层(C)求平均值,尺寸变为(H,W)
    # heatmap = torch.max(heat,dim=1).values.squeeze()

    heatmap = heatmap.numpy()#转换为numpy数组
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)#minmax归一化处理
    heatmap = cv2.resize(heatmap,(img.size[0],img.size[1]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_HSV)#颜色变换
    rgbimg=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    cv2.imwrite('visual/'+Name+'_'+str(k)+'.jpg',rgbimg)
    plt.imshow(heatmap)
    
    plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap*0.4+np.array(img)[:,:,::-1] #图像叠加，注意翻转通道，cv用的是bgr
    cv2.imwrite('Visual/superimg.jpg',superimg)#保存结果
    # 可视化叠加至源图像的结果
    img_ = np.array(Image.open('Visual/superimg.jpg').convert('RGB'))
    plt.imshow(img_)
    plt.show()

model_str = model.__str__()

if model_str.startswith('SSD'):
    for k in range(len(module_name)):
        for j in range(len(p_in[0][0])):
            print(p_in[k][0][j].shape)
            print(p_out[k].shape)
            show_feature_map(img_file, p_in[k][0][j])
            # show_feature_map(img_file, torch.sigmoid(p_out[k]))
            print()

if model_str.startswith('RetinaNet'): # retinanet
    for k in range(len(module_name)):# 不同尺寸的特征图
        print(k)
        print(p_in[k][0].shape)
        print(p_out[k].shape)
        # show_feature_map(img_file, p_in[k][j])
        show_feature_map(img_file, torch.sigmoid(p_out[k]))
        print()

if model_str.startswith('FasterRCNN'): # FasterRCNN
    for k in range(len(module_name)):
        print(k)
        print(p_in[k][0].shape)
        print(p_out[k].shape)
        # show_feature_map(img_file, p_in[k][0])
        show_feature_map(img_file, torch.sigmoid(p_out[k]),k)
        print()

print(summary(model, (1,3,300,300), verbose=1))
