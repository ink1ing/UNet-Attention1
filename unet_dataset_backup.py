"""
UNet数据集定义文件：用于玉米锈病图像分割和病害等级回归的数据加载
处理多光谱TIF图像和JSON标注，生成分割掩码和病害等级标签
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2
from skimage import transform as sk_transform
import warnings

# 全局抑制rasterio地理参考警告
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class CornRustSegmentationDataset(Dataset):
    """
    玉米锈病分割数据集类
    
    功能：
    1. 加载多光谱TIF图像
    2. 解析JSON标注生成分割掩码
    3. 提取病害等级标签
    4. 数据增强和预处理
    """
    
    def __init__(self, data_root, json_root, img_size=128, transform=None, 
                 selected_channels=[100, 200, 300], augment=True):
        """
        初始化数据集
        
        参数:
            data_root: 图像数据根目录
            json_root: JSON标注根目录
            img_size: 图像尺寸，默认128x128
            transform: 数据变换
            selected_channels: 选择的光谱通道，默认[100, 200, 300]
            augment: 是否进行数据增强
        """
        self.data_root = data_root
        self.json_root = json_root
        self.img_size = img_size
        self.transform = transform
        self.selected_channels = selected_channels
        self.augment = augment
        
        # 位置映射：文件夹名到类别ID
        self.position_map = {
            'l': 0,  # 下部
            'm': 1,  # 中部
            't': 2   # 上部
        }
        
        # 病害等级映射
        self.grade_map = {
            '0': 0.0,   # 无病害
            '3': 3.0,   # 轻度
            '5': 5.0,   # 中度
            '7': 7.0,   # 重度
            '9': 9.0    # 极重度
        }
        
        # 收集所有样本
        self.samples = self._collect_samples()
        print(f"数据集初始化完成，共找到 {len(self.samples)} 个样本")
        
        # 数据增强变换
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            self.augment_transform = None
    
    def _collect_samples(self):
        """收集所有有效的样本路径"""
        samples = []
        
        # 遍历所有子目录
        for subdir in os.listdir(self.data_root):
            if not os.path.isdir(os.path.join(self.data_root, subdir)):
                continue
                
            img_dir = os.path.join(self.data_root, subdir)
            
            # 尝试不同的JSON目录命名方式
            json_dir_candidates = [
                os.path.join(self.json_root, subdir),  # 直接匹配
                os.path.join(self.json_root, f"{subdir}_json"),  # 带_json后缀
            ]
            
            json_dir = None
            for candidate in json_dir_candidates:
                if os.path.exists(candidate):
                    json_dir = candidate
                    break
            
            if json_dir is None:
                print(f"警告：找不到对应的JSON目录，尝试了: {json_dir_candidates}")
                continue
            
            # 遍历图像文件
            for img_file in os.listdir(img_dir):
                if not img_file.endswith('.tif'):
                    continue
                    
                # 构建对应的JSON文件路径
                json_file = img_file.replace('.tif', '.json')
                img_path = os.path.join(img_dir, img_file)
                json_path = os.path.join(json_dir, json_file)
                
                if os.path.exists(json_path):
                    samples.append({
                        'img_path': img_path,
                        'json_path': json_path,
                        'subdir': subdir
                    })
                else:
                    print(f"警告：找不到对应的JSON文件 {json_path}")
        
        return samples
    
    def _load_tif_image(self, img_path):
        """加载TIF图像并选择指定通道"""
        try:
            # 抑制rasterio的地理参考警告
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
                with rasterio.open(img_path) as src:
                    # 读取所有波段
                    image = src.read()  # 形状: (bands, height, width)
                    
                    # 选择指定通道
                    if image.shape[0] >= max(self.selected_channels):
                        selected_image = image[self.selected_channels]  # 选择3个通道
                    else:
                        # 如果通道数不足，使用前3个通道或重复通道
                        if image.shape[0] >= 3:
                            selected_image = image[:3]
                        else:
                            # 重复通道到3个
                            selected_image = np.repeat(image, 3, axis=0)[:3]
                    
                    # 转换为 (height, width, channels) 格式
                    selected_image = np.transpose(selected_image, (1, 2, 0))
                    
                    # 归一化到 [0, 1]
                    if selected_image.max() > 1.0:
                        selected_image = selected_image.astype(np.float32) / 65535.0
                    
                    return selected_image
                
        except Exception as e:
            print(f"加载TIF图像失败 {img_path}: {e}")
            # 返回随机图像作为备用
            return np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
    
    def _parse_json_annotation(self, json_path, img_shape):
        """解析JSON标注文件，生成分割掩码和病害等级"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            height, width = img_shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            grade_values = []
            
            # 解析标注
            if 'shapes' in data:
                for shape in data['shapes']:
                    label = shape.get('label', '')
                    points = shape.get('points', [])
                    
                    if not points:
                        continue
                    
                    # 提取病害等级
                    grade = self._extract_grade_from_label(label)
                    if grade is not None:
                        grade_values.append(grade)
                    
                    # 创建多边形掩码
                    if len(points) >= 3:  # 至少需要3个点形成多边形
                        # 将点坐标转换为整数
                        polygon_points = [(int(p[0]), int(p[1])) for p in points]
                        
                        # 创建PIL图像用于绘制多边形
                        pil_mask = Image.new('L', (width, height), 0)
                        ImageDraw.Draw(pil_mask).polygon(polygon_points, outline=1, fill=1)
                        
                        # 转换为numpy数组并合并到主掩码
                        poly_mask = np.array(pil_mask)
                        mask = np.maximum(mask, poly_mask)
            
            # 计算平均病害等级
            if grade_values:
                avg_grade = np.mean(grade_values)
            else:
                avg_grade = 0.0  # 无标注时默认为0
            
            return mask, avg_grade
            
        except Exception as e:
            print(f"解析JSON标注失败 {json_path}: {e}")
            # 返回空掩码和0等级
            height, width = img_shape[:2]
            return np.zeros((height, width), dtype=np.uint8), 0.0
    
    def _extract_grade_from_label(self, label):
        """从标签中提取病害等级"""
        for grade_str, grade_val in self.grade_map.items():
            if grade_str in label:
                return grade_val
        return None
    
    def _resize_image_and_mask(self, image, mask):
        """调整图像和掩码尺寸"""
        # 使用OpenCV调整图像尺寸
        image_resized = cv2.resize(image, (self.img_size, self.img_size), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # 使用最近邻插值调整掩码尺寸（保持标签值不变）
        mask_resized = cv2.resize(mask, (self.img_size, self.img_size), 
                                interpolation=cv2.INTER_NEAREST)
        
        return image_resized, mask_resized
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        img_path = sample['img_path']
        json_path = sample['json_path']
        
        # 加载图像
        image = self._load_tif_image(img_path)
        
        # 解析标注
        mask, grade = self._parse_json_annotation(json_path, image.shape)
        
        # 调整尺寸
        image, mask = self._resize_image_and_mask(image, mask)
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        grade = torch.tensor(grade, dtype=torch.float32)
        
        # 数据增强
        if self.augment_transform is not None:
            # 对图像和掩码应用相同的变换
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # 图像增强
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            
            # 掩码增强（需要特殊处理）
            torch.manual_seed(seed)
            mask_3d = mask.unsqueeze(0).float()  # 添加通道维度
            mask_3d = self.augment_transform(mask_3d)
            mask = mask_3d.squeeze(0).long()  # 移除通道维度并转回long类型
        
        # 应用额外的变换
        if self.transform:
            image = self.transform(image)
        
        return image, mask, grade


def get_unet_dataloaders(data_root, json_root, batch_size=8, img_size=128, 
                        test_size=0.2, val_size=0.1, num_workers=0, 
                        selected_channels=[100, 200, 300]):
    """
    创建UNet训练、验证和测试数据加载器
    
    参数:
        data_root: 图像数据根目录
        json_root: JSON标注根目录
        batch_size: 批次大小
        img_size: 图像尺寸
        test_size: 测试集比例
        val_size: 验证集比例
        num_workers: 数据加载进程数
        selected_channels: 选择的光谱通道
        
    返回:
        train_loader, val_loader, test_loader: 数据加载器
    """
    
    # 创建完整数据集
    full_dataset = CornRustSegmentationDataset(
        data_root=data_root,
        json_root=json_root,
        img_size=img_size,
        selected_channels=selected_channels,
        augment=False  # 先不增强，后面分别设置
    )
    
    # 分割数据集
    indices = list(range(len(full_dataset)))
    
    # 首先分离测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=True
    )
    
    # 再从训练+验证集中分离验证集
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size/(1-test_size), random_state=42, shuffle=True
    )
    
    print(f"数据集分割: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, 测试集 {len(test_indices)}")
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # 为训练集启用数据增强
    train_dataset.dataset.augment = True
    train_dataset.dataset.augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def create_sample_masks(data_root, json_root, output_dir, num_samples=5):
    """
    创建样本掩码可视化，用于验证数据加载是否正确
    
    参数:
        data_root: 图像数据根目录
        json_root: JSON标注根目录
        output_dir: 输出目录
        num_samples: 样本数量
    """
    import matplotlib.pyplot as plt
    
    dataset = CornRustSegmentationDataset(
        data_root=data_root,
        json_root=json_root,
        augment=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        image, mask, grade = dataset[i]
        
        # 转换为numpy格式用于可视化
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(image_np)
        axes[0].set_title(f'原图 (样本 {i})')
        axes[0].axis('off')
        
        # 掩码
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f'分割掩码')
        axes[1].axis('off')
        
        # 叠加显示
        axes[2].imshow(image_np)
        axes[2].imshow(mask_np, alpha=0.5, cmap='Reds')
        axes[2].set_title(f'叠加显示 (等级: {grade:.1f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"样本可视化已保存到 {output_dir}")


if __name__ == "__main__":
    # 测试数据集
    data_root = "./guanceng-bit"
    json_root = "./biaozhu_json"
    
    if os.path.exists(data_root) and os.path.exists(json_root):
        # 创建数据加载器
        train_loader, val_loader, test_loader = get_unet_dataloaders(
            data_root=data_root,
            json_root=json_root,
            batch_size=4,
            num_workers=0
        )
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        for images, masks, grades in train_loader:
            print(f"图像形状: {images.shape}")
            print(f"掩码形状: {masks.shape}")
            print(f"等级形状: {grades.shape}")
            print(f"等级范围: {grades.min():.2f} - {grades.max():.2f}")
            break
        
        # 创建样本可视化
        create_sample_masks(data_root, json_root, "./sample_masks", num_samples=3)
    else:
        print("数据目录不存在，请检查路径设置") 