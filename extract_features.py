"""
电力盗窃检测 (ETD) 特征提取模块 (Phase 4)。
将预训练的条件扩散模型作为特征提取器 (Feature Extractor)，
对多实例窗口数据进行离线推断，提取并缓存降维后的物理感知潜在表征与重构异常得分。
"""
import os
import torch
from dataclasses import dataclass
from typing import Tuple, List
from tqdm import tqdm

# 确保导入路径与项目结构完全对应
from data_loader import LoaderConfig, get_dataloaders
from model import ConditionalUNet1D, GaussianDiffusion1D

# ==============================================================================
# 1. 离线特征提取配置 (Extraction Configuration)
# ==============================================================================
@dataclass
class ExtractionConfig:
    """特征提取与缓存全局配置"""
    # 权重路径 (需与 Phase 3 训练输出对齐)
    diffusion_ckpt: str = "./checkpoints_phase3/best_diffusion_epoch_029_raw.pth"
    cache_dir: str = "./features_cache_lit"
    
    # 输出文件名映射
    train_cache_file: str = "train_feats_log1p_lit.pt"
    test_cache_file: str = "test_feats_log1p_lit.pt"
    
    # 硬件与模型超参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    base_dim: int = 64
    seq_length: int = 256
    timesteps: int = 1000
    
    # 异常得分超参数
    top_k_ratio: float = 0.17


# ==============================================================================
# 2. 核心特征提取引擎 (Core Extraction Engine)
# ==============================================================================
@torch.no_grad()
def extract_and_cache_features(
    loader: torch.utils.data.DataLoader, 
    diffusion: GaussianDiffusion1D, 
    device: torch.device, 
    cache_path: str,
    cfg: ExtractionConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    执行窗口级别的特征提取，合并 U-Net 瓶颈层表征与扩散逆向重构得分。
    
    返回:
        all_feats: [N, K, feat_dim] 用户多窗口特征包
        all_labels: [N] 二分类标签
        all_masks: [N, K] 窗口有效性掩码
        all_cons: [N] 用户 ID 序列
    """
    diffusion.eval()
    all_feats, all_labels, all_masks, all_cons = [], [], [], []
    
    # 动态推导特征维度: Latent Features (base_dim * 8) + Anomaly Score (1)
    feat_dim = diffusion.model.base_dim * 8 + 1 
    
    pbar = tqdm(loader, desc=f"Extracting -> {os.path.basename(cache_path)}")
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)                 
        labels = batch["label"].to(device, non_blocking=True)        
        keep_mask = batch["keep_window_mask"].to(device, non_blocking=True) 
        cons_ids = batch["cons_id"]
        
        # 张量维度重塑: [B, K, C, W] -> [B * K, C, W]
        B, K, C, W = x.shape
        x_flat = x.view(B * K, C, W)
        
        # 展平有效值掩码并初始化全零特征张量
        valid_flat_mask = keep_mask.view(-1) > 0
        feats_flat = torch.zeros((B * K, feat_dim), device=device)
        
        if valid_flat_mask.any():
            # 仅对存在物理意义的有效窗口进行前向传播以节省显存
            x_valid = x_flat[valid_flat_mask]
            x_log = x_valid[:, 0:1, :]
            x_mm = x_valid[:, 1:2, :]
            msk = x_valid[:, 2:3, :]
            
            # 1. 提取 U-Net 潜在物理表征 (Latent Features)
            latent = diffusion.extract_latent_features(x_log, x_mm, msk) 
            # 2. 计算流形逆向重构偏离度 (Anomaly Score)
            score = diffusion.compute_anomaly_score(x_log, x_mm, msk, k_min=0.15, k_max=0.40)
            
            # 级联并回填至有效索引位
            feats_flat[valid_flat_mask] = torch.cat([latent, score.unsqueeze(-1)], dim=-1)
            
        # 恢复多实例包维度: [B * K, feat_dim] -> [B, K, feat_dim]
        feats = feats_flat.view(B, K, -1).cpu()
        
        all_feats.append(feats)
        all_labels.append(labels.cpu())
        all_masks.append(keep_mask.cpu())
        all_cons.extend(cons_ids)
        
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 持久化特征资产
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        "features": all_feats, 
        "labels": all_labels, 
        "masks": all_masks, 
        "cons_ids": all_cons
    }, cache_path)
    
    return all_feats, all_labels, all_masks, all_cons

# ==============================================================================
# 3. 提取流水线调度 (Execution Pipeline)
# ==============================================================================
def main():
    """初始化预训练模型权重并依次提取各数据集划分的张量特征。"""
    cfg = ExtractionConfig()
    device = torch.device(cfg.device)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    
    print("Initializing Conditional Diffusion Engine for Feature Extraction...")
    unet = ConditionalUNet1D(
        in_channels=3, 
        out_channels=1, 
        base_dim=cfg.base_dim
    ).to(device)
    
    diffusion = GaussianDiffusion1D(
        model=unet, 
        seq_length=cfg.seq_length, 
        timesteps=cfg.timesteps
    ).to(device)
    
    # 权重加载与降级兼容性校验
    if not os.path.exists(cfg.diffusion_ckpt):
        raise FileNotFoundError(f"Missing Diffusion Checkpoint: {cfg.diffusion_ckpt}")
        
    print(f"Loading checkpoint weights from: {cfg.diffusion_ckpt}")
    ckpt = torch.load(cfg.diffusion_ckpt, map_location=device, weights_only=False)
    # 优先加载 EMA 权重，若无则回退至 Standard 权重
    state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    diffusion.load_state_dict(state_dict)
    
    loaders = get_dataloaders(LoaderConfig())
    
    # 训练集特征计算与持久化
    train_cache_path = os.path.join(cfg.cache_dir, cfg.train_cache_file)
    print("\n[Step 1/2] Extracting Classifier Training Set...")
    extract_and_cache_features(loaders["clf_train"], diffusion, device, train_cache_path, cfg)
    
    # 测试集特征计算与持久化
    test_cache_path = os.path.join(cfg.cache_dir, cfg.test_cache_file)
    print("\n[Step 2/2] Extracting Testing Set...")
    extract_and_cache_features(loaders["test"], diffusion, device, test_cache_path, cfg)
    
    print("\nFeature extraction and disk caching completed successfully.")

if __name__ == "__main__":
    main()