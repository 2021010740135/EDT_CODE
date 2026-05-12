"""
电力盗窃检测 (ETD) 数据预处理模块。
支持在“本文原生分布保留策略”与“对比论文的插值/截断策略”之间进行无缝切换，
以支持严密的消融实验 (Ablation Study)。
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置 (Global Configuration)
# ==============================================================================
@dataclass
class GlobalConfig:
    """
    原始数据处理与训练/测试集划分的全局配置。
    """
    input_file: str = "./data/electricity.csv"
    output_dir: str = "./results_phase1_lit"

    # 训练集/测试集划分配置
    test_size: float = 0.20
    random_state: int = 58

    # ==========================================
    # 对比实验开关 (Ablation Study Parameters)
    # ==========================================
    # 插值策略:
    # "t-7": 本文提出的周期插补 (保留真实用电周期)
    # "linear_zero": 参考文献 Eq(1) 的策略 (孤立缺失均值插值，连续缺失补零)
    impute_strategy: str = "linear_zero" 
    
    # 截断策略 (Outlier Removal):
    # "none": 本文策略，保留真实极值与原生分布
    # "sigma_rule": 参考文献 Eq(2) 的策略 (均值 + N*标准差截断)
    truncate_strategy: str = "sigma_rule" 
    sigma_multiplier: float = 3.0  # 论文文字描述为 3-sigma，但公式 (2) 为 2*STD。可调节。

# ==============================================================================
# 2. 预处理核心引擎
# ==============================================================================
class SGCCMinimalPreprocessor:
    """
    针对国网 (SGCC) 时间序列数据的预处理器。
    有机结合了本文的周期保留策略与参考文献的统计截断策略。
    """
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.ts_cols = []
        self.report: Dict[str, Any] = {}

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print(f"正在加载数据: {self.cfg.input_file}")
        raw = pd.read_csv(self.cfg.input_file)

        if not {"CONS_NO", "FLAG"}.issubset(raw.columns):
            raise ValueError("数据集中必须包含 CONS_NO 和 FLAG 列。")

        raw = raw.dropna(subset=["FLAG"]).copy()
        raw["FLAG"] = raw["FLAG"].astype(int)
        raw["CONS_NO"] = raw["CONS_NO"].astype(str)

        # 提取并排序时间步列
        self.ts_cols = [c for c in raw.columns if c not in ("CONS_NO", "FLAG")]
        try:
            self.ts_cols = sorted(self.ts_cols, key=lambda x: pd.to_datetime(x))
        except Exception:
            pass

        df_ts = raw[self.ts_cols].copy()
        initial_count = len(df_ts)
        
        labels = raw["FLAG"].reset_index(drop=True).to_numpy(dtype=np.int8)
        cons_ids = raw["CONS_NO"].reset_index(drop=True).to_numpy()

        # 1. 提取有效值掩码: 1.0 表示有效，0.0 表示 NaN (供后续 Diffusion 隔离梯度)
        mask_np = (~df_ts.isna()).to_numpy(dtype=np.float32)

        # ==============================================================================
        # 2. 插值策略 (Imputation Strategy)
        # ==============================================================================
        if self.cfg.impute_strategy == "t-7":
            # 本文方法: 严格的周期平移
            df_shifted = df_ts.shift(7, axis=1)
            df_imputed = df_ts.fillna(df_shifted).fillna(0.0)
        
        elif self.cfg.impute_strategy == "linear_zero":
            # 参考文献 Eq (1): 仅当相邻两侧非空时取均值，否则取 0
            # Pandas 的 limit=1 恰好能实现“仅对孤立单个 NaN 进行插值，连续 NaN 不插值”的效果
            df_imputed = df_ts.interpolate(method='linear', axis=1, limit=1).fillna(0.0)
        else:
            raise ValueError(f"Unknown impute strategy: {self.cfg.impute_strategy}")

        raw_np = df_imputed.to_numpy(dtype=np.float32)

        # ==============================================================================
        # 3. 截断策略 (Truncation Strategy / Outlier Removal)
        # ==============================================================================
        if self.cfg.truncate_strategy == "sigma_rule":
            # 参考文献 Eq (2): 用户级 X_UB = X_AVG + N * X_STD
            means = np.mean(raw_np, axis=1, keepdims=True)
            stds = np.std(raw_np, axis=1, keepdims=True)
            
            x_ub = means + self.cfg.sigma_multiplier * stds
            # 超出上限的截断为 x_ub，低于 0 的截断为 0
            raw_np = np.clip(raw_np, a_min=0.0, a_max=x_ub)
            
        elif self.cfg.truncate_strategy == "none":
            # 本文方法: 保留真实突变极值，防止抹杀短期高强度窃电行为
            raw_np = np.clip(raw_np, a_min=0.0, a_max=None)
        else:
            raise ValueError(f"Unknown truncate strategy: {self.cfg.truncate_strategy}")

        # ==============================================================================
        # 4. 幅度平滑变换
        # ==============================================================================
        # 注意: 参考文献的 Eq(3) Min-Max 归一化已由下游 data_loader.py 自动接管生成 x_mm。
        # 此处保留 Log1p 是为了保证传入扩散模型的主特征通道 (x_log) 幅度稳定，维持实验公平性。
        x_log = np.log1p(raw_np)

        self.report = {
            "initial_samples": initial_count,
            "final_samples": len(df_ts),
            "num_time_steps": len(self.ts_cols),
            "num_positive": int(np.sum(labels == 1)),
            "num_negative": int(np.sum(labels == 0)),
            "mean_missing_ratio": float(1.0 - mask_np.mean()),
            "impute_strategy": self.cfg.impute_strategy,
            "truncate_strategy": self.cfg.truncate_strategy,
            "sigma_multiplier": self.cfg.sigma_multiplier
        }

        return x_log, mask_np, labels, cons_ids

# ==============================================================================
# 3. 分层划分与持久化 (略)
# ==============================================================================
def save_stratified_splits(
    x_log: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    cons_ids: np.ndarray,
    ts_cols: list,
    cfg: GlobalConfig,
    report: dict
):
    print(f"执行 {cfg.impute_strategy} + {cfg.truncate_strategy} 划分...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, stratify=labels, random_state=cfg.random_state
    )

    splits = {"train": idx_train, "test": idx_test}
    report["splits"] = {}

    for name, indices in splits.items():
        s_x_log = x_log[indices]
        s_mask = mask[indices]
        s_labels = labels[indices]
        s_cons = cons_ids[indices]

        npz_path = os.path.join(cfg.output_dir, f"{name}.npz")
        np.savez_compressed(
            npz_path,
            x_log=s_x_log,
            mask=s_mask,
            labels=s_labels,
            cons_ids=s_cons,
            time_cols=np.array(ts_cols, dtype=object)
        )

        report["splits"][name] = {
            "total": len(indices),
            "positives": int(np.sum(s_labels == 1)),
            "negatives": int(np.sum(s_labels == 0))
        }

    report["config"] = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    cfg = GlobalConfig()
    cleaner = SGCCMinimalPreprocessor(cfg)
    x_log, mask, labels, cons_ids = cleaner.run()
    save_stratified_splits(x_log, mask, labels, cons_ids, cleaner.ts_cols, cfg, cleaner.report)