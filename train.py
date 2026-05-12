"""
电力盗窃检测 (ETD) 模型训练入口。
执行 Phase 3: 无监督条件扩散模型 (Conditional Diffusion Model) 的流形学习预训练。
包含基于 AMP (自动混合精度) 的加速训练与确定性评估流。
"""
import json
import logging
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# 确保导入路径与项目结构完全对应
from data_loader import LoaderConfig, get_dataloaders
from model import ConditionalUNet1D, GaussianDiffusion1D, unpack_diffusion_batch

# ==============================================================================
# 1. 硬件加速与全局配置 (Hardware Acceleration & Configurations)
# ==============================================================================
# 启用 TensorFloat-32 (TF32) 以加速 Ampere 及更新架构 GPU 的矩阵乘法
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

CONFIG: Dict[str, Any] = {
    # 训练超参数
    "epochs": 30,
    "max_lr": 4e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    
    # 路径与设备
    "save_dir": "./checkpoints_phase3",
    "log_dir": "./logs_phase3_lit",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 58,
    
    # 模型架构超参数
    "timesteps": 1000,
    "seq_length": 256,
    "base_dim": 64,
    "dropout": 0.10,
}

# ==============================================================================
# 2. 基础工具组件 (Utility Functions)
# ==============================================================================
def set_seed(seed: int) -> None:
    """固定全局随机种子以确保实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(log_dir: str) -> logging.Logger:
    """初始化双通道（控制台+文件）日志记录器。"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("diffusion_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def save_json(path: str, payload: dict) -> None:
    """将字典数据序列化保存为 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_checkpoint(
    path: str,
    diffusion: GaussianDiffusion1D,
    optimizer: optim.Optimizer,
    epoch: int,
    metric: float,
) -> None:
    """保存包含模型权重、优化器状态与全局配置的标准检查点快照。"""
    ckpt = {
        "epoch": int(epoch),
        "metric": float(metric),
        "model_state_dict": diffusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
    }
    torch.save(ckpt, path)

# ==============================================================================
# 3. 核心训练与评估逻辑 (Training & Evaluation Engines)
# ==============================================================================
@torch.no_grad()
def evaluate_loss(
    diffusion: GaussianDiffusion1D,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    split_name: str,
) -> float:
    """
    确定性验证 (Deterministic Evaluation)。
    强制使用固定的时间步 (t) 生成器，切断随机采样带来的 Loss 波动，
    确保验证集 Loss 曲线具备真实、严格的收敛指示意义。
    """
    if loader is None:
        return float("inf")

    diffusion.eval()
    total_loss = 0.0
    total_batches = 0
    use_amp = device.type == "cuda"

    # 初始化独立的确定性生成器
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(CONFIG["seed"]) 

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [{split_name}]", leave=False, disable=True)
    
    for batch in pbar:
        x_log, x_mm, mask = unpack_diffusion_batch(batch)
        x_log = x_log.to(device, non_blocking=True)
        x_mm = x_mm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        # 使用固定种子生成时间步
        t = torch.randint(
            0, diffusion.timesteps, (x_log.shape[0],), 
            generator=eval_generator, device=device
        ).long()

        with autocast(device_type="cuda", enabled=use_amp):
            out = diffusion(x_log, x_mm, mask, t)
            loss = out["loss"]

        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    logger.info(f"[{split_name}] Epoch {epoch:03d} (Deterministic): loss = {avg_loss:.6f}")
    return avg_loss

def train_one_epoch(
    diffusion: GaussianDiffusion1D,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> float:
    """
    单轮次训练引擎。执行随机加噪前向传播、损失计算、梯度裁剪与 AMP 混合精度更新。
    """
    diffusion.train()
    total_loss = 0.0
    total_batches = 0
    use_amp = device.type == "cuda"
    current_lr = optimizer.param_groups[0]['lr']

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [Train]", leave=False, disable=True)
    
    for batch in pbar:
        x_log, x_mm, mask = unpack_diffusion_batch(batch)
        x_log = x_log.to(device, non_blocking=True)
        x_mm = x_mm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        # 训练阶段使用随机时间步
        t = torch.randint(0, diffusion.timesteps, (x_log.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            out = diffusion(x_log, x_mm, mask, t)
            loss = out["loss"]

        # 混合精度反向传播与梯度裁剪
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    avg_loss = total_loss / max(total_batches, 1)
    logger.info(f"[Train] Epoch {epoch:03d}: loss = {avg_loss:.6f}")
    return avg_loss

# ==============================================================================
# 4. 主执行流水线 (Main Execution Pipeline)
# ==============================================================================
def main():
    """模型初始化与端到端训练调度入口。"""
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    logger = setup_logger(CONFIG["log_dir"])
    device = torch.device(CONFIG["device"])

    logger.info("=" * 80)
    logger.info("🚀 启动 Phase 3: Diffusion Manifold Learning")
    logger.info("目标重构: x_log | 物理条件: x_mm + mask")
    logger.info("策略: 短周期扩散预训练 | 保存全局最优检查点与定期快照")
    logger.info("=" * 80)

    # 加载数据管道
    loader_cfg = LoaderConfig()
    loaders = get_dataloaders(loader_cfg)
    diff_train_dl = loaders["diff_train"]
    diff_val_dl = loaders["diff_val"]

    save_json(os.path.join(CONFIG["log_dir"], "loader_config.json"), asdict(loader_cfg))
    save_json(os.path.join(CONFIG["log_dir"], "train_config.json"), CONFIG)

    # 实例化网络架构
    unet = ConditionalUNet1D(
        in_channels=3,
        out_channels=1,
        base_dim=CONFIG["base_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model=unet,
        seq_length=CONFIG["seq_length"],
        timesteps=CONFIG["timesteps"],
    ).to(device)

    # 优化器与调度器
    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=CONFIG["max_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    
    # 基于 Epoch 的余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-5
    )
    
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    # 状态追踪器
    history = []
    best_val_loss = float("inf")
    start = time.time()

    # 迭代训练流
    for epoch in range(1, CONFIG["epochs"] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_one_epoch(
            diffusion, diff_train_dl, optimizer, scaler, device, epoch, logger
        )
        
        diff_val_loss = evaluate_loss(
            diffusion, diff_val_dl, device, epoch, logger, split_name="Diff-Val(Test-Normal)"
        )
        
        # Epoch 级别更新学习率
        scheduler.step()

        is_best = False
        is_snapshot = False

        # 1. 跟踪并保存全局最优模型权重 (Best Checkpoint)
        if diff_val_loss < best_val_loss:
            best_val_loss = diff_val_loss
            is_best = True
            best_filename = f"best_diffusion_epoch_{epoch:03d}_raw.pth"
            save_checkpoint(
                os.path.join(CONFIG["save_dir"], best_filename),
                diffusion, optimizer, epoch, best_val_loss
            )
            logger.info(f"🌟 发现当前最优扩散模型 (Epoch {epoch}) | 验证集损失: {best_val_loss:.6f} | 已保存为: {best_filename}")

        # 2. 定期保存固定快照 (Snapshot)，专供下游特征提取一致性比对使用
        if epoch in [10, 20, 30]:
            is_snapshot = True
            snap_filename = f"snapshot_epoch_{epoch:03d}_raw.pth"
            save_checkpoint(
                os.path.join(CONFIG["save_dir"], snap_filename),
                diffusion, optimizer, epoch, diff_val_loss
            )
            logger.info(f"📸 保存定期快照 (Epoch {epoch}) | 已保存为: {snap_filename}")

        # 记录收敛轨迹
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "diff_val_loss": float(diff_val_loss),
            "lr": float(current_lr),
            "is_best": is_best,
            "is_snapshot": is_snapshot
        }
        history.append(record)

    # 序列化历史记录
    save_json(os.path.join(CONFIG["log_dir"], "history.json"), history)

    elapsed = time.time() - start
    logger.info("=" * 80)
    logger.info(f"✅ 预训练任务圆满结束 (总耗时: {elapsed / 3600:.2f} 小时)")
    logger.info(f"🏁 记录的最优验证集损失: {best_val_loss:.6f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()