# OSA Personalized Acoustic Intervention System (OSA个性化声学干预系统)

A reinforcement learning-based system for preventing obstructive sleep apnea (OSA) events through personalized acoustic interventions delivered via earphones.

## Architecture

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
│  │        Multimodal Feature Extractor (33-dim)         │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │     OSA Risk Predictor (Bi-LSTM + Attention)         │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │     Hierarchical Intervention Protocol (FSM)         │   │
│  │  MONITOR → DETECT → DIRECTIONAL CUE → EVALUATE      │   │
│  │                              ↓                       │   │
│  │                       SHORT BURST CUE → COOLDOWN     │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │     SAC RL Agent (6D Continuous Action Space)         │   │
│  │  → [Loudness, Frequency, Duration, Timing, ITD, ILD] │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │     Binaural Audio Synthesizer (L/R channels)        │   │
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

### Hierarchical Strategy (分层干预策略)

1. **检测阶段**: 多模态特征 → Bi-LSTM时序风险预测
2. **评估体位**: 是否仰卧 (supine)?
3. **方向性Cue** (If supine): ITD/ILD双耳空间音频引导侧卧, 低频250Hz, 渐进响度
4. **等待响应**: 30-90秒观察窗口
5. **短促声音Cue** (If no improvement): 中频脉冲1000Hz, 0.5s短时长

### RL Action Space (6D连续动作空间)

| Parameter | Range | Description |
|-----------|-------|-------------|
| Loudness | [0, 1] | 相对响度 |
| Frequency | [20, 4000] Hz | 载波频率 |
| Duration | [0.1, 10] s | 刺激持续时间 |
| Timing | [0, 1] | Epoch内时机 |
| ITD | [-1.5, 1.5] ms | 双耳时间差（空间定位） |
| ILD | [-20, 20] dB | 双耳强度差（空间定位） |

## Evaluation Results (评估结果)

| Agent | Mean Reward | SpO₂ Min | OSA Events | Arousals |
|-------|-----------|----------|-----------|---------|
| No Intervention | 46.6 | 72.0% | 2.0 | 36.1 |
| Rule-Based | 51.8 | 73.5% | 1.9 | 45.1 |
| Random | 96.0 | 81.6% | 1.9 | 84.4 |
| **SAC (Trained)** | **116.0** | **87.2%** | 2.1 | 68.9 |

**SAC agent achieves +15.2% improvement in minimum SpO₂ vs no intervention.**

## File Structure

```
osa_system/
├── __init__.py                 # System initialization
├── signal_processing.py        # 4-modality feature extraction (33-dim)
├── risk_predictor.py           # OSA risk predictor (Bi-LSTM + Attention)
├── environment.py              # Gymnasium simulation environment
├── intervention_protocol.py    # Hierarchical intervention FSM
├── rl_agent.py                 # SAC/PPO training & baselines
├── audio_synthesis.py          # Binaural audio synthesizer
└── main.py                     # Integrated system & CLI
```

## References

- SAC: Haarnoja et al., arxiv:1812.05905
- Acoustic Control MDP: arxiv:2312.05674
- Portiloop: arxiv:2107.13473
- HealthGym: arxiv:2203.06369
- 1D-ViT Sleep: arxiv:2502.17486
- Kinesis: arxiv:2503.14637
