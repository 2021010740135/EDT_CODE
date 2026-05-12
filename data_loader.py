"""
电力盗窃检测 (ETD) 数据加载与管道模块。
负责将一维时序数据切分为滑动窗口，执行 Min-Max 归一化，并构建 PyTorch 数据加载器。
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置
# ==============================================================================
@dataclass
class WindowBuilderConfig:
    """滑动窗口切分与特征构建流水线配置"""
    input_dir: str = "./results_phase1_lit"
    output_dir: str = "./results_phase2_lit"
    split_names: tuple[str, ...] = ("train", "test")
    
    # 窗口超参数
    window_size: int = 256
    stride: int = 28
    add_tail_window: bool = True
    min_valid_ratio: float = 0.0

@dataclass
class LoaderConfig:
    """PyTorch DataLoader 批处理与并发配置"""
    input_dir: str = "./results_phase2_lit"
    train_file: str = "train_windows.npz"
    test_file: str = "test_windows.npz"
    train_normal_file: str = "train_normal_windows.npz"
    test_normal_file: str = "test_normal_windows.npz"
    
    # 批处理超参数
    batch_size_diffusion: int = 256
    batch_size_classifier: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # 启发式过滤阈值 (0.0 表示保留原生数据分布)
    diffusion_min_valid_ratio: float = 0.0
    classifier_min_valid_ratio: float = 0.0
    seed: int = 58

# 多实例学习与生成模型的窗口掩码阈值。
# 当前均设为 0.0，表示不丢弃任何窗口（维持原生序列长度），
# 仅用于为下游模型生成全通透的注意力掩码 (Attention Mask)。
# ==============================================================================
# 2. 窗口特征构建引擎 (Phase 2)
# ==============================================================================
def compute_window_starts(length: int, window_size: int, stride: int, add_tail_window: bool = True) -> np.ndarray:
    """计算一维时间序列的滑动窗口起始索引序列。"""
    if length < window_size:
        raise ValueError(f"序列长度 ({length}) 小于窗口大小 ({window_size})。")
    
    starts = list(range(0, length - window_size + 1, stride))
    tail_start = length - window_size
    
    if add_tail_window and (not starts or starts[-1] != tail_start):
        starts.append(tail_start)
    return np.array(sorted(set(starts)), dtype=np.int32)

def slice_2d_to_windows(x: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """
    利用高级索引将二维特征矩阵转换为三维窗口张量。
    形状变换: [N, T_steps] -> [N, K_windows, W_size]
    """
    idx = starts[:, None] + np.arange(window_size, dtype=np.int32)[None, :]
    return x[:, idx]

def build_split_windows(payload: Dict[str, np.ndarray], cfg: WindowBuilderConfig) -> Dict[str, np.ndarray]:
    """
    执行全局 Min-Max 归一化并构建多通道特征矩阵。
    返回包含特征张量与元数据的字典字典结构。
    """
    x_log = payload["x_log"].astype(np.float32)
    mask = payload["mask"].astype(np.float32)
    labels = payload["labels"].astype(np.int8)
    cons_ids = payload["cons_ids"].astype(str)
    
    _, seq_len = x_log.shape
    starts = compute_window_starts(seq_len, cfg.window_size, cfg.stride, cfg.add_tail_window)
    
    # 全局 Min-Max 归一化 (Global Min-Max Normalization)
    # 目标形状: [N, seq_len]
    g_min = x_log.min(axis=-1, keepdims=True)
    g_max = x_log.max(axis=-1, keepdims=True)
    denom = np.maximum(g_max - g_min, 1e-6)
    x_mm_global = (x_log - g_min) / denom
    
    # 执行张量切片
    x_log_w = slice_2d_to_windows(x_log, starts, cfg.window_size)
    x_mm_w = slice_2d_to_windows(x_mm_global, starts, cfg.window_size)
    mask_w = slice_2d_to_windows(mask, starts, cfg.window_size)
    
    # 计算有效值比例掩码
    valid_ratio = mask_w.mean(axis=-1).astype(np.float32)
    keep_window_mask = (valid_ratio >= cfg.min_valid_ratio).astype(np.int8)
    
    # 通道堆叠 (Channel stacking). 最终特征张量形状: [N, K, 3, W]
    x = np.stack([x_log_w, x_mm_w, mask_w], axis=2).astype(np.float32)
    window_ends = (starts + cfg.window_size - 1).astype(np.int32)
    
    return {
        "x": x,
        "labels": labels,
        "cons_ids": cons_ids,
        "window_starts": starts,
        "window_ends": window_ends,
        "valid_ratio": valid_ratio,
        "keep_window_mask": keep_window_mask,
        "time_cols": payload["time_cols"],
        "channel_names": np.array(["x_log", "x_mm", "mask"], dtype=object)
    }

def subset_normal_users(payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """提取标签为正常的样本集，专供无监督生成模型（如 Diffusion）训练使用。"""
    normal_mask = payload["labels"] == 0
    idx = np.where(normal_mask)[0]
    out = {}
    
    for k, v in payload.items():
        if k in {"window_starts", "window_ends", "time_cols", "channel_names"}:
            out[k] = v
        elif isinstance(v, np.ndarray) and v.shape[0] == len(normal_mask):
            out[k] = v[idx]
        else:
            out[k] = v
    return out

class NumpyEncoder(json.JSONEncoder):
    """用于序列化审计报告中 NumPy 原生数据类型的 JSON 编码器。"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def run_window_builder():
    """读取 Phase 1 数据并生成滑动窗口张量资产的入口函数。"""
    cfg = WindowBuilderConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    report = {"config": asdict(cfg), "splits": {}}
    
    for split_name in cfg.split_names:
        input_path = os.path.join(cfg.input_dir, f"{split_name}.npz")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"缺失必要的数据文件: {input_path}")
            
        data = np.load(input_path, allow_pickle=True)
        raw_payload = {k: data[k] for k in data.files}
        
        # 构建完整切分
        win_payload = build_split_windows(raw_payload, cfg)
        np.savez_compressed(os.path.join(cfg.output_dir, f"{split_name}_windows.npz"), **win_payload)
        
        # 构建正常用户子集
        normal_payload = subset_normal_users(win_payload)
        np.savez_compressed(os.path.join(cfg.output_dir, f"{split_name}_normal_windows.npz"), **normal_payload)
        
        report["splits"][split_name] = {"users": win_payload["x"].shape[0], "windows_per_user": win_payload["x"].shape[1]}
        report["splits"][f"{split_name}_normal"] = {"users": normal_payload["x"].shape[0], "windows_per_user": normal_payload["x"].shape[1]}

    report_path = os.path.join(cfg.output_dir, "window_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)


# ==============================================================================
# 3. 数据集定义 (Dataset Definitions)
# ==============================================================================
class WindowSplit:
    """包装已加载的窗口张量资产，提供严谨的维度约束校验。"""
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺失窗口资产文件: {path}")

        raw = np.load(path, allow_pickle=True)
        payload = {k: raw[k] for k in raw.files}

        self.path = path
        self.x = payload["x"].astype(np.float32)                      # [N, K, C, W]
        self.labels = payload["labels"].astype(np.int64)              # [N]
        self.cons_ids = payload["cons_ids"].astype(str)               # [N]
        self.valid_ratio = payload["valid_ratio"].astype(np.float32)  # [N, K]
        
        channel_names = [str(c) for c in payload["channel_names"].tolist()]
        expected_channels = ["x_log", "x_mm", "mask"]
        if channel_names != expected_channels:
            raise ValueError(f"通道顺序不匹配。期望 {expected_channels}, 实际获取 {channel_names}")

    @property
    def n_users(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.x.shape[1])


class DiffusionWindowDataset(Dataset):
    """
    面向无监督生成模型（如 Diffusion）的窗口级数据集。
    展平用户(N)与窗口(K)维度。最终输出形状: [C, W]
    """
    def __init__(self, split: WindowSplit, min_valid_ratio: float = 0.0):
        self.split = split
        valid = split.valid_ratio >= min_valid_ratio
        
        self.index: List[Tuple[int, int]] = [
            (u, k) for u in range(split.n_users) for k in range(split.n_windows) if valid[u, k]
        ]
        if not self.index:
            raise ValueError(f"经过 valid_ratio >= {min_valid_ratio} 过滤后, 文件 {split.path} 中不存在有效窗口。")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u, k = self.index[idx]
        return {
            "x": torch.from_numpy(self.split.x[u, k]),
            "valid_ratio": torch.tensor(self.split.valid_ratio[u, k], dtype=torch.float32),
        }


class UserBagDataset(Dataset):
    """
    面向分类模型的多实例学习 (MIL) 用户级数据集。
    保留完整的窗口维度作为"包" (Bag)。最终输出形状: [K, C, W]
    """
    def __init__(self, split: WindowSplit, min_valid_ratio: float = 0.0):
        self.split = split
        self.min_valid_ratio = min_valid_ratio

    def __len__(self) -> int:
        return self.split.n_users

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        valid_ratio = torch.from_numpy(self.split.valid_ratio[idx].copy())
        keep_mask = (valid_ratio >= self.min_valid_ratio).float()
        return {
            "x": torch.from_numpy(self.split.x[idx].copy()),
            "label": torch.tensor(self.split.labels[idx], dtype=torch.float32),
            "keep_window_mask": keep_mask,
            "cons_id": str(self.split.cons_ids[idx])
        }

# ==============================================================================
# 4. DataLoader 装配接口
# ==============================================================================
def get_dataloaders(cfg: LoaderConfig = None) -> Dict[str, DataLoader]:
    """实例化所有的训练与测试 DataLoader，封装固定的随机种子。"""
    if cfg is None:
        cfg = LoaderConfig()
        
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_split = WindowSplit(os.path.join(cfg.input_dir, cfg.train_file))
    test_split = WindowSplit(os.path.join(cfg.input_dir, cfg.test_file))
    train_normal_split = WindowSplit(os.path.join(cfg.input_dir, cfg.train_normal_file))
    test_normal_split = WindowSplit(os.path.join(cfg.input_dir, cfg.test_normal_file))

    datasets = {
        "diff_train": DiffusionWindowDataset(train_normal_split, cfg.diffusion_min_valid_ratio),
        "diff_val": DiffusionWindowDataset(test_normal_split, cfg.diffusion_min_valid_ratio),
        "clf_train": UserBagDataset(train_split, cfg.classifier_min_valid_ratio),
        "test": UserBagDataset(test_split, cfg.classifier_min_valid_ratio),
    }

    common_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
    }
    
    return {
        "diff_train": DataLoader(datasets["diff_train"], batch_size=cfg.batch_size_diffusion, shuffle=True, drop_last=False, **common_kwargs),
        "diff_val": DataLoader(datasets["diff_val"], batch_size=cfg.batch_size_diffusion, shuffle=False, drop_last=False, **common_kwargs),
        "clf_train": DataLoader(datasets["clf_train"], batch_size=cfg.batch_size_classifier, shuffle=True, drop_last=False, **common_kwargs),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size_classifier, shuffle=False, drop_last=False, **common_kwargs),
    }

# ==============================================================================
# 5. 标准执行入口
# ==============================================================================
if __name__ == "__main__":
    # 执行流水线构建张量资产，移除所有局部张量维度调试代码
    run_window_builder()