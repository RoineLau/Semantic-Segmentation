import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, RandomCrop, Resize, Compose as AlbCompose,
    RandomBrightnessContrast, HueSaturationValue, CLAHE, ShiftScaleRotate,
    ElasticTransform, GridDistortion
)
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassRecall, MulticlassF1Score
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import cv2
from skimage import measure
from scipy.ndimage import label

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# PSPNet模型
class CustomPSPNet(smp.PSPNet):
    def __init__(self, encoder_name, encoder_weights, classes, activation):
        super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes, activation=activation)
        # 添加 Dropout
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.dropout(x)
        return x

# 定义数据集类
class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, balance_data=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(img_dir)
            for file in files if file.endswith("leftImg8bit.png")
        ])
        self.masks = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(mask_dir)
            for file in files if file.endswith("gtFine_labelIds.png")
        ])
        assert len(self.images) == len(self.masks), "图片和掩码数量不匹配！"

        # 数据平衡
        if balance_data:
            self.images, self.masks = self.balance_classes(self.images, self.masks)
   
    def balance_classes(self, images, masks):
        # 初始化类别计数
        class_counts = np.zeros(34, dtype=np.int64)
    
        # 使用进度条计算每个类别的样本数
        print("Calculating initial class distribution...")
        for mask_path in tqdm(masks, desc="Processing Masks"):
            mask = np.array(Image.open(mask_path), dtype=np.int64)
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u] += c
    
        # 可视化初始类别分布
        plt.figure(figsize=(12, 6))
        plt.bar(range(34), class_counts, color='blue')
        plt.title('Initial Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Pixel Count')
        plt.savefig('initial_class_distribution.png')
        plt.close()
    
        # 寻找稀缺类别
        median_count = np.median(class_counts)
        minority_classes = np.where(class_counts < median_count)[0]
    
        # 上采样少数类样本
        augmented_images, augmented_masks = [], []
        print("Balancing data by augmenting minority class samples...")
        for img, mask in tqdm(zip(images, masks), total=len(images), desc="Augmenting Data"):
            mask_array = np.array(Image.open(mask), dtype=np.int64)
            if any(cls in minority_classes for cls in np.unique(mask_array)):
                # 对包含稀缺类的样本进行复制
                augmented_images.extend([img] * 2)
                augmented_masks.extend([mask] * 2)
            else:
                augmented_images.append(img)
                augmented_masks.append(mask)
    
        # 重新计算类别分布
        balanced_class_counts = np.zeros(34, dtype=np.int64)
        print("Calculating balanced class distribution...")
        for mask_path in tqdm(augmented_masks, desc="Processing Augmented Masks"):
            mask = np.array(Image.open(mask_path), dtype=np.int64)
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                balanced_class_counts[u] += c
    
        # 可视化平衡后的类别分布
        plt.figure(figsize=(12, 6))
        plt.bar(range(34), balanced_class_counts, color='green')
        plt.title('Balanced Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Pixel Count')
        plt.savefig('balanced_class_distribution.png')
        plt.close()
    
        # 返回平衡后的数据
        return augmented_images, augmented_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        try:
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            mask = np.array(Image.open(mask_path), dtype=np.int64)
        except Exception as e:
            print(f"跳过损坏的图片: {img_path}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

            image = image.clone().detach().float() if isinstance(image, torch.Tensor) else torch.from_numpy(image).float()
            mask = mask.clone().detach().long() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).long()
            
        return image, mask

train_img_dir = "cityscapes/leftImg8bit/train"
train_mask_dir = "cityscapes/gtFine/train"
val_img_dir = "cityscapes/leftImg8bit/val"
val_mask_dir = "cityscapes/gtFine/val"

# 数据增强与预处理
train_transform = AlbCompose([
    Resize(256,512),
    RandomCrop(256, 512),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.2),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2),
    HueSaturationValue(p=0.2),
    ToTensorV2(transpose_mask=True),
])

val_transform = AlbCompose([
    Resize(256, 512),
    ToTensorV2(transpose_mask=True),
])

# 数据加载器
train_dataset = CityscapesDataset(train_img_dir, train_mask_dir, transform=train_transform, balance_data=True)
val_dataset = CityscapesDataset(val_img_dir, val_mask_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

# 加载预训练模型
model = CustomPSPNet(
    encoder_name="resnet101", 
    encoder_weights="imagenet", 
    classes=34, 
    activation=None
).to(device)


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        ComboLoss combines CrossEntropyLoss and DiceLoss/FocalLoss
        :param alpha: Weight for CrossEntropyLoss
        :param beta: Weight for DiceLoss
        """
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(mode='multiclass')

    def forward(self, pred, target):
        """
        Forward pass of the ComboLoss
        :param pred: Model predictions
        :param target: Ground truth masks
        :return: Combined loss
        """
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * ce_loss + self.beta * dice_loss

# IoU 计算函数
metric = MulticlassJaccardIndex(num_classes=34, ignore_index=None)
def calculate_iou(pred, mask):
    """
    使用 torchmetrics 计算 IoU
    :param pred: 模型预测结果，形状为 [B, num_classes, H, W]
    :param mask: 真实掩码标签，形状为 [B, H, W]
    :return: 平均 IoU 值
    """
    pred = torch.argmax(pred, dim=1)
    iou = metric(pred.cpu(), mask.cpu())
    return iou.item()

# Pixel Accuracy
def calculate_pixel_accuracy(pred, mask):
    pred = torch.argmax(pred, dim=1)
    return (pred == mask).float().mean().item()

iou_metric = MulticlassJaccardIndex(num_classes=34).to(device)

# 验证函数
def validate(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_pa = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            # 使用 torchmetrics 计算 IoU
            total_iou += calculate_iou(outputs, masks)

            # 计算像素准确率
            pred = torch.argmax(outputs, dim=1).to(device)
            total_pa += (pred == masks).float().mean().item()

    return total_loss / len(loader), total_iou / len(loader), total_pa / len(loader)

# 可视化函数
def visualize_predictions(images, masks, preds, save_path):
    for i in range(len(images)):
        plt.figure(figsize=(12, 4))

        # 确保张量从 GPU 转移到 CPU
        image_cpu = images[i].permute(1, 2, 0).cpu().numpy()
        mask_cpu = masks[i].cpu().numpy()
        pred_cpu = preds[i].cpu()

        plt.subplot(1, 3, 1)
        plt.imshow(image_cpu)
        plt.title("Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_cpu, cmap="gray")
        plt.title("Ground Truth")
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_cpu, cmap="jet")
        plt.title("Prediction")
        
        plt.savefig(f"{save_path}/sample_{i}.png")
        plt.close()

# 早停策略类
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 主函数
if __name__ == "__main__":
    num_epochs = 300
    best_loss = float("inf")
    early_stopping = EarlyStopping(patience=15, delta=0.0001)

    # 优化器、调度器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    loss_fn = ComboLoss(alpha=0.5, beta=0.5).to(device)

    # 用于绘制损失曲线和指标曲线的列表
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_pas = []
    val_pas = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_iou, val_pa = validate(val_loader, model, loss_fn, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}, Val PA={val_pa:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, "best_model_4.pth")
            print("Best model saved.")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("loss_curve4.png")
    plt.show()

    # 可视化预测结果
    save_dir = "predictions_4"
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            visualize_predictions(images.cpu(), masks.cpu(), preds, save_dir)
            break