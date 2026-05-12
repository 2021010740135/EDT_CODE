"""
电力盗窃检测 (ETD) 下游分类器模块 (Phase 4)。
基于晚期融合 (Late Fusion) 架构与原生 BCE 损失 (Vanilla BCE)，
结合 CNN 空间特征提取与扩散逆向异常得分的对数正则化进行排序预测。
"""
import os
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

# ==============================================================================
# 1. 全局配置 (Classifier Configuration)
# ==============================================================================
@dataclass
class ClassifierConfig:
    """下游晚期融合分类器全局配置"""
    output_dir: str = "./results_phase4_late_fusion_bce"
    cache_dir: str = "./features_cache_lit"
    train_feat_file: str = "train_feats_log1p_lit.pt"
    test_feat_file: str = "test_feats_log1p_lit.pt"
    
    # 网络超参数
    hidden_dim: int = 128
    
    # 训练超参数
    epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-3
    batch_size: int = 128
    
    # 环境配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

# ==============================================================================
# 2. 评估指标引擎 (Evaluation Metrics)
# ==============================================================================
def set_seed(seed: int) -> None:
    """固定全局随机种子。"""
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
    """
    计算基于严格阈值的排序评估指标 (AUC, MAP@K)。
    
    参数:
        y_true: 真实标签数组
        y_pred: 模型预测的异常概率数组
        R_list: MAP 截断点列表
    """
    metrics = {'AUC': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) >= 1 else 0.0}
    sorted_truth = pd.DataFrame({'truth': y_true, 'pred': y_pred}).sort_values(by='pred', ascending=False).truth.tolist()
    
    for R in R_list:
        metrics[f'MAP@{R}'] = average_precision(sorted_truth[:R])
    return metrics

# ==============================================================================
# 3. 核心网络架构 (Network Architecture)
# ==============================================================================
class ElectricityTheftCNNRanker(nn.Module):
    """
    电力盗窃检测双流融合排序器。
    执行多实例包 (MIL) 维度的 CNN 空间特征提取，并与对数规范化的扩散重构分数进行晚期融合。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # 扣除最后一维的 Score，剩余为 U-Net 潜在物理表征维度
        self.latent_dim = input_dim - 1 
        
        # 空间域特征提取组件 (Spatial Feature Extractor)
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.conv1 = nn.Conv1d(self.latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.drop1d = nn.Dropout1d(0.2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        # 异常分数独立标准化层
        self.score_bn = nn.BatchNorm1d(1)
        
        # 晚期融合 MLP 判别器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64), 
            nn.Dropout(0.3), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. 物理意义解耦 (Decoupling)
        latent_x = x[..., :-1]  # [B, K, latent_dim]
        score_x = x[..., -1:]   # [B, K, 1]
        
        # 2. CNN 空间域特征提取
        latent_x = latent_x.masked_fill((mask == 0.0).unsqueeze(-1), 0.0)
        latent_x = self.input_norm(latent_x).transpose(1, 2)
        
        h_cnn = self.pool1(self.relu1(self.drop1d(self.conv1(latent_x))))
        h_cnn = self.pool2(self.relu2(self.conv2(h_cnn))).squeeze(-1)  # [B, hidden_dim * 2]
        
        # 3. 异常分数提取与物理池化
        # 无效序列位填充 -1e4 以隔离梯度，提取最大值代表用户最高异常风险
        valid_score_x = score_x.masked_fill((mask == 0.0).unsqueeze(-1), -1e4)
        pooled_score = valid_score_x.max(dim=1)[0]  # [B, 1]
        
        # 4. 异常分数数值规范化 (Numeric Normalization)
        # 夹紧负值防范 log1p NaN，采用对数压缩长尾分布并通过 BN 消除特征量级失衡
        clamped_score = torch.clamp(pooled_score, min=0.0)
        log_score = torch.log1p(clamped_score)
        norm_score = self.score_bn(log_score)
        
        # 5. 晚期融合 (Late Fusion) 与逻辑回归
        fused_features = torch.cat([h_cnn, norm_score], dim=-1)  # [B, hidden_dim * 2 + 1]
        
        return self.classifier(fused_features).squeeze(-1)

# ==============================================================================
# 4. 训练主引擎 (Training Pipeline)
# ==============================================================================
@torch.no_grad()
def get_cached_features(cfg: ClassifierConfig) -> Tuple[torch.Tensor, ...]:
    """加载前置流水线生成的预提取降维特征与目标标签。"""
    tr = torch.load(os.path.join(cfg.cache_dir, cfg.train_feat_file), weights_only=False)
    te = torch.load(os.path.join(cfg.cache_dir, cfg.test_feat_file), weights_only=False)
    return tr["features"], tr["labels"], tr["masks"], te["features"], te["labels"], te["masks"]

def train_pipeline() -> None:
    """初始化双流排序器并执行单阶段 Vanilla BCE 训练。"""
    cfg = ClassifierConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    print("=" * 80)
    print("Initializing Late Fusion Ranker (Native Baseline + Pure Vanilla BCE)")
    print("=" * 80)
    
    tx, ty, tm, vx, vy, vm = get_cached_features(cfg)
    
    model = ElectricityTheftCNNRanker(tx.shape[-1], cfg.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # 纯原生 BCE 损失，不施加类别权重
    criterion = nn.BCEWithLogitsLoss()
    
    loader = DataLoader(
        TensorDataset(tx, tm, ty), 
        batch_size=cfg.batch_size, 
        shuffle=True
    )

    best_score = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        avg_loss = 0.0
        
        for bx, bm, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device), bm.to(device)), by.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
        scheduler.step()

        # 评估阶段
        model.eval()
        with torch.no_grad():
            # 批次推理以防内存溢出
            v_scores_list = []
            for i in range(0, len(vx), 512):
                bx_val = vx[i:i+512].to(device)
                bm_val = vm[i:i+512].to(device)
                v_scores_list.append(torch.sigmoid(model(bx_val, bm_val)))
                
            v_scores = torch.cat(v_scores_list).cpu()
            metrics = calculate_metrics_strict(vy.numpy(), v_scores.numpy())
            
            # SOTA 保留规则: 基于 MAP 指标的加权加和
            weighted_score = 0.3 * metrics['MAP@100'] + 0.7 * metrics['MAP@200']
            
            if weighted_score >= best_score:
                best_score = weighted_score
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_sota_model_late_fusion.pth"))
                mark = "[SOTA Snapshot Saved]"
            else: 
                mark = ""
                
            print(
                f"Epoch {epoch:02d}/{cfg.epochs} | Loss: {avg_loss/len(loader):.4f} | "
                f"AUC: {metrics['AUC']:.4f} | MAP@100: {metrics['MAP@100']:.4f} | "
                f"MAP@200: {metrics['MAP@200']:.4f} {mark}"
            )

    print("=" * 80)
    print(f"Training Complete. Optimal Fusion Model locked in: {cfg.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    cfg_init = ClassifierConfig()
    os.makedirs(cfg_init.output_dir, exist_ok=True)
    train_pipeline()