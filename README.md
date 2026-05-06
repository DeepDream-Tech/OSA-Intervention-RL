# OSA Personalized Acoustic Intervention System V2 (OSA个性化声学干预系统)

A classification-based system for preventing obstructive sleep apnea (OSA) events through personalized acoustic interventions delivered via earphones. Trained on real UCDDB clinical data with 95.94% accuracy.

## Architecture V2

```
┌──────────────────────────────────────────────────────────────┐
│                    EARPHONE DEVICE                           │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐        │
│  │ RIP Band│  │ Mic     │  │ IMU    │  │ SpO2     │        │
│  │ (chest/ │  │ (snore  │  │(6-axis │  │(pulse ox)│        │
│  │ abdomen)│  │ audio)  │  │accel/  │  │          │        │
│  └────┬────┘  └────┬────┘  └───┬────┘  └────┬─────┘        │
│       │            │           │             │               │
│  ┌────▼────────────▼───────────▼─────────────▼──────────┐   │
│  │    Multimodal Feature Extractor (8-dim UCDDB)        │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  State Classifier (4 states: Awake, Normal,          │   │
│  │  Snoring, Apnea) + Severity Score                    │   │
│  │  Accuracy: 95.94% | Snoring: 100% | Apnea: 100%      │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  Trend Encoder (Bi-LSTM) - 60-90s temporal patterns  │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  Decision Engine (Rule-Based, Explainable)           │   │
│  │  • Awake/Normal → No intervention                    │   │
│  │  • Snoring + Supine → Directional cue (250Hz)        │   │
│  │  • Apnea → Short burst cue (1000Hz)                  │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │  Binaural Audio Synthesizer (ITD/ILD spatial audio)  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Sensor Modalities (传感器模态)

| 模态 | 传感器 | 关键特征 | 功能 |
|------|--------|---------|------|
| **呼吸力学 (RIP)** | 胸腹带 | 呼吸振幅、频率、相位差（矛盾呼吸识别） | 检测气道阻塞进展 |
| **音频 (Audio)** | 耳机麦克风 | 鼾声RMS、基频稳定性、持续/间断模式 | 区分Sustained Snoring vs Snore Bout |
| **体位 (IMU)** | 6轴加速度计/陀螺仪 | 重力向量、仰卧位识别 | 判断高风险仰卧位 |
| **血氧 (SpO₂)** | 脉搏血氧仪 | 实时数值、下降斜率、ODI指数 | 评估缺氧风险 |

## Intervention Protocol (干预协议)

### Decision Engine Strategy (决策引擎策略)

The V2 system uses explainable rule-based logic:

1. **清醒/正常睡眠**: 无干预 (preserve sleep quality)
2. **打鼾 + 仰卧位**: 方向性Cue (ITD/ILD双耳空间音频引导侧卧, 低频250Hz)
3. **打鼾 + 高严重度**: 监测趋势，恶化时预防性干预
4. **呼吸中断**: 短促声音Cue (中频脉冲1000Hz, 0.5s) 激活气道肌肉
5. **冷却期**: 干预后等待，防止习惯化

Every decision includes a human-readable reason for clinical transparency.

### Intervention Parameters (干预参数)

| Parameter | Range | Description |
|-----------|-------|-------------|
| Loudness | [0, 0.7] | 相对响度 (safety-limited) |
| Frequency | [250, 1000] Hz | 载波频率 (250Hz directional, 1000Hz burst) |
| Duration | [0.5, 2.0] s | 刺激持续时间 |
| ITD | [-1.5, 1.5] ms | 双耳时间差（空间定位） |
| ILD | [-20, 20] dB | 双耳强度差（空间定位） |

## Evaluation Results (评估结果)

### Classification Performance on UCDDB Real Data

| State | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| 清醒 (Awake) | 99.2% | 88.1% | 93.3% | 6,712 |
| 正常睡眠 (Normal) | 93.2% | 99.6% | 96.3% | 10,931 |
| 打鼾 (Snoring) | **100.0%** | **100.0%** | **100.0%** | 2,443 |
| 呼吸中断 (Apnea) | **100.0%** | **100.0%** | **100.0%** | 703 |

**Overall Accuracy: 95.94%**

**Cross-Subject Generalization (LOSO):** 95.92% ± 1.60%

### Key Findings

- **Perfect detection** of critical states (Snoring and Apnea): 100% precision and recall
- **High recall** for normal sleep (99.6%): minimal false alarms
- **Strong generalization**: Individual subject accuracy ranges from 91.23% to 98.43%
- Main confusion occurs between Awake and Normal Sleep (clinically acceptable, no unnecessary intervention)

## File Structure

```
osa_system/
├── __init__.py                 # V2 system exports
├── system_v2.py                # V2 core: StateClassifier, TrendEncoder, DecisionEngine, OSASystemV2
├── signal_processing.py        # 8-dim feature extraction (UCDDB-aligned)
├── audio_synthesis.py          # Binaural audio synthesizer with ITD/ILD
├── ucddb_parser.py             # UCDDB data parser (4-state labels)
├── train_classifier.py         # Classifier training with LOSO cross-validation
├── train_real_signals.py       # Training on real signal features
├── evaluate_real_data.py       # Evaluation on real UCDDB annotations
└── main.py                     # V2 integrated system & CLI
```

## Usage

```bash
# Demo mode (simulated sleep session)
python osa_system/main.py --mode demo --episodes 3

# Evaluate on real UCDDB data
python osa_system/main.py --mode evaluate

# Train classifier on UCDDB
python osa_system/main.py --mode train --epochs 50
```

## References

- UCDDB Dataset: University College Dublin Sleep Apnea Database
- Focal Loss: Lin et al., ICCV 2017
- Bi-LSTM for Sleep: Supratak et al., IEEE TBME 2017
- Acoustic Intervention: Portiloop (arxiv:2107.13473)
- 1D-ViT Sleep: arxiv:2502.17486
- DeepArousal-Net: IEEE TBME 2025
