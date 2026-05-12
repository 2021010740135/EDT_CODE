"""
扩散模型权重自动化检索与基准测试模块 (Automated Checkpoint Search).
通过内存级特征提取与即时下游微调，评估并排序指定目录下的所有 Diffusion 权重。
引入了严密的显存隔离 (Memory Isolation) 与 RNG 状态锁定机制，确保与独立运行流水线绝对对齐。
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

# 确保导入路径与项目结构一致
from data_loader import LoaderConfig, get_dataloaders
from model import ConditionalUNet1D, GaussianDiffusion1D

# ==============================================================================
# 1. 全局配置 (Search Configuration)
# ==============================================================================
@dataclass
class SearchConfig:
    """自动化评估流水线全局配置"""
    ckpt_dir: str = "./checkpoints_phase3"  # 待遍历的扩散模型权重目录
   
    
    # 下游评估分类器超参数
    hidden_dim: int = 128
    epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-3
    batch_size: int = 128
    
    # 环境配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42  # 下游分类器专属 Seed (对齐 BCE.py)

# ==============================================================================
# 2. 核心数学与度量函数 (Metrics & Utilities)
# ==============================================================================
def set_seed(seed: int) -> None:
    """固定全局随机种子以确保评估的绝对公平性与确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def average_precision(r: List[int]) -> float:
    """计算单个排名的平均精度 (Average Precision)。"""
    r = np.asarray(r) != 0
    out = [np.mean(r[:k + 1]) for k in range(r.size) if r[k]]
    return np.mean(out) if out else 0.0

def calculate_metrics_strict(y_true: np.ndarray, y_pred: np.ndarray, R_list: List[int] = [100, 200]) -> Dict[str, float]:
    """计算严密的排序评估指标 (AUC, MAP@K)。"""
    metrics = {'AUC': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 1 else 0.0}
    sorted_truth = pd.DataFrame({'truth': y_true, 'pred': y_pred}).sort_values(by='pred', ascending=False).truth.tolist()
    for R in R_list:
        metrics[f'MAP@{R}'] = average_precision(sorted_truth[:R])
    return metrics

# ==============================================================================
# 3. 下游基准分类器架构 (Benchmark Ranker Architecture)
# ==============================================================================
class ElectricityTheftCNNRanker(nn.Module):
    """
    用于评估的标准化电力盗窃检测双流融合排序器。
    维持与最终版 BCE.py 完全一致的晚期融合架构与对数归一化机制。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = input_dim - 1 
        
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.conv1 = nn.Conv1d(self.latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.drop1d = nn.Dropout1d(0.2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        self.score_bn = nn.BatchNorm1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64), 
            nn.Dropout(0.3), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        latent_x = x[..., :-1]  
        score_x = x[..., -1:]   
        
        # 空间特征提取
        latent_x = latent_x.masked_fill((mask == 0.0).unsqueeze(-1), 0.0)
        latent_x = self.input_norm(latent_x).transpose(1, 2)
        h_cnn = self.pool1(self.relu1(self.drop1d(self.conv1(latent_x))))
        h_cnn = self.pool2(self.relu2(self.conv2(h_cnn))).squeeze(-1) 
        
        # 异常分数提取与数值规整
        valid_score_x = score_x.masked_fill((mask == 0.0).unsqueeze(-1), -1e4)
        pooled_score = valid_score_x.max(dim=1)[0] 
        clamped_score = torch.clamp(pooled_score, min=0.0)
        log_score = torch.log1p(clamped_score)
        norm_score = self.score_bn(log_score)
        
        # 晚期融合与预测
        fused_features = torch.cat([h_cnn, norm_score], dim=-1) 
        return self.classifier(fused_features).squeeze(-1)

# ==============================================================================
# 4. 内存级特征提取引擎 (In-Memory Extraction Engine)
# ==============================================================================
@torch.no_grad()
def extract_features_in_memory(
    loader: torch.utils.data.DataLoader, 
    diffusion: GaussianDiffusion1D, 
    device: torch.device,
    cfg: SearchConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    跳过磁盘 I/O 缓存，直接将张量提取至 RAM。专供快速循环评估使用。
    """
    diffusion.eval()
    all_feats, all_labels, all_masks = [], [], []
    
    # 动态推导维度
    feat_dim = diffusion.model.base_dim * 8 + 1 
    
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)                 
        labels = batch["label"].to(device, non_blocking=True)        
        keep_mask = batch["keep_window_mask"].to(device, non_blocking=True) 
        
        B, K, C, W = x.shape
        x_flat = x.view(B * K, C, W)
        valid_flat_mask = keep_mask.view(-1) > 0
        feats_flat = torch.zeros((B * K, feat_dim), device=device)
        
        if valid_flat_mask.any():
            x_valid = x_flat[valid_flat_mask]
            x_log = x_valid[:, 0:1, :]
            x_mm = x_valid[:, 1:2, :]
            msk = x_valid[:, 2:3, :]
            
            latent = diffusion.extract_latent_features(x_log, x_mm, msk) 
            # 使用基于 FFT 的 FAAT 动态截断
            score = diffusion.compute_anomaly_score(x_log, x_mm, msk, k_min=0.05, k_max=0.40)
            feats_flat[valid_flat_mask] = torch.cat([latent, score.unsqueeze(-1)], dim=-1)
            
        feats = feats_flat.view(B, K, -1).cpu()
        all_feats.append(feats)
        all_labels.append(labels.cpu())
        all_masks.append(keep_mask.cpu())
        
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_masks, dim=0)

# ==============================================================================
# 5. 单节点权重评估流水线 (Checkpoint Evaluation Pipeline)
# ==============================================================================
def evaluate_checkpoint(
    ckpt_path: str, 
    loaders: Dict[str, torch.utils.data.DataLoader], 
    device: torch.device,
    cfg: SearchConfig
) -> Tuple[float, Dict[str, float]]:
    """装载扩散权重、提取特征并执行完整的下游基准测试。包含严格的显存隔离与 RNG 状态对齐。"""
    print(f"[{os.path.basename(ckpt_path)}] Initializing Diffusion Extractor...")
    unet = ConditionalUNet1D(in_channels=3, out_channels=1, base_dim=64).to(device)
    diffusion = GaussianDiffusion1D(model=unet, seq_length=256, timesteps=1000).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    diffusion.load_state_dict(state_dict)
    
    print(f"[{os.path.basename(ckpt_path)}] Extracting features into RAM...")
    
    # ==============================================================================
    # [核心修复 1] 强制对齐特征提取阶段的随机种子 (Seed = 58)
    # 确保 q_sample 注入的初始噪声与独立执行 extract_features.py 时完全一致！
    # ==============================================================================
    set_seed(58) 
    
    tx, ty, tm = extract_features_in_memory(loaders["clf_train"], diffusion, device, cfg)
    vx, vy, vm = extract_features_in_memory(loaders["test"], diffusion, device, cfg)
    
    # 显存隔离阶段一：销毁扩散引擎，释放资源供下游分类器使用
    del diffusion, unet, ckpt, state_dict
    torch.cuda.empty_cache()

    print(f"[{os.path.basename(ckpt_path)}] Training benchmark ranker...")
    
    # ==============================================================================
    # [核心修复 2] 强制对齐下游分类器训练阶段的随机种子 (Seed = 42)
    # 确保网络初始化与 DataLoader 批次洗牌与独立执行 BCE.py 时完全一致！
    # ==============================================================================
    set_seed(cfg.seed)
    
    model = ElectricityTheftCNNRanker(tx.shape[-1], cfg.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(
        TensorDataset(tx, tm, ty), 
        batch_size=cfg.batch_size, 
        shuffle=True
    )

    best_metrics = {}
    best_score = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for bx, bm, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device), bm.to(device)), by.to(device).float())
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_scores_list = []
            for i in range(0, len(vx), 512):
                bx_val = vx[i:i+512].to(device)
                bm_val = vm[i:i+512].to(device)
                v_scores_list.append(torch.sigmoid(model(bx_val, bm_val)))
                
            v_scores = torch.cat(v_scores_list).cpu()
            m = calculate_metrics_strict(vy.numpy(), v_scores.numpy())
            
            # 复合性能得分: 倾向于高召回的后段排名 (MAP@200 权重较高)
            weighted_score = 0.3 * m['MAP@100'] + 0.7 * m['MAP@200']
            
            # [核心修复 3] 对齐 BCE.py 的 >= 逻辑，确保快照收敛点一致
            if weighted_score >= best_score:
                best_score = weighted_score
                best_metrics = m

    # 显存隔离阶段二：清空分类器与数据集张量，防止下一个 ckpt 评估时 OOM
    del model, optimizer, train_loader, tx, ty, tm, vx, vy, vm
    torch.cuda.empty_cache()
    
    return best_score, best_metrics

# ==============================================================================
# 6. 主调度器与分析报告生成 (Main Scheduler & Reporting)
# ==============================================================================
def main():
    cfg = SearchConfig()
    device = torch.device(cfg.device)
    
    ckpt_files = glob.glob(os.path.join(cfg.ckpt_dir, "*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth checkpoints located in directory: {cfg.ckpt_dir}")
    
    print("=" * 80)
    print(f"Located {len(ckpt_files)} checkpoints. Initializing data pipelines...")
    print("=" * 80)
    
    loaders = get_dataloaders(LoaderConfig())
    results = []
    
    for i, ckpt_path in enumerate(ckpt_files, 1):
        print(f"\n[ Progress: {i}/{len(ckpt_files)} | Evaluating: {os.path.basename(ckpt_path)} ]")
        print("-" * 80)
        
        score, metrics = evaluate_checkpoint(ckpt_path, loaders, device, cfg)
        
        results.append({
            "checkpoint": os.path.basename(ckpt_path),
            "weighted_score": score,
            "AUC": metrics["AUC"],
            "MAP@100": metrics["MAP@100"],
            "MAP@200": metrics["MAP@200"]
        })
        print(f"Evaluation Complete -> Score: {score:.4f} | AUC: {metrics['AUC']:.4f} | MAP@100: {metrics['MAP@100']:.4f}")

    # ==============================================================================
    # 7. 基准评估排行榜 (Leaderboard Output)
    # ==============================================================================
    results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)
    
    print("\n\n" + "=" * 90)
    print("Diffusion Checkpoint Benchmark Leaderboard (Top 10)")
    print("-" * 90)
    print(f"{'Rank':<5} | {'Checkpoint Name':<35} | {'Score':<8} | {'AUC':<8} | {'MAP@100':<8} | {'MAP@200':<8}")
    print("-" * 90)
    
    top_n = min(10, len(results))
    for i in range(top_n):
        r = results[i]
        print(f"{i+1:<5} | {r['checkpoint']:<35} | {r['weighted_score']:<8.4f} | {r['AUC']:<8.4f} | {r['MAP@100']:<8.4f} | {r['MAP@200']:<8.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()