# 预实验模糊声学干预控制器

这个目录是独立程序，不依赖 `osa_system`，也不接入现有强化学习系统。

当前版本是 `fuzzy-v1`：

- 生理输入只使用 `RIP + SpO2`
- 安全输入使用用户清醒按钮
- 控制器只输出 `是否发声 + 响度`
- 播放时机固定为 `0.6s`
- 声音内容不由模型决定，仍由外部播放层自己选

音量如何标定成真实分贝，见 [VOLUME_CALIBRATION.md](/Users/liujia/Desktop/internship/intervention-v0/VOLUME_CALIBRATION.md:1)。

## 1. 模型边界

控制器负责：

- 估计当前呼吸是否进入了需要干预的异常状态
- 计算模糊触发分数 `trigger_score`
- 计算模糊响度分数 `loudness_score`
- 在固定播放节拍下决定当前是否要发声，以及响度该取哪一档

控制器不负责：

- 选择声音文件
- 合成音频
- 播放设备控制
- dB 标定
- 波形设计

所以集成层只需要读取：

- `command.should_play_sound`
- `command.loudness`

## 2. 输入和输出

### 输入

生理输入：

- `RIP`
- `SpO2`

安全输入：

- `button`

其中 `button` 不是模糊控制输入，它只做一件事：

- 用户按下后立即停止干预

### 输出

`PreExperimentController.update()` 返回 `InterventionCommand`，主要字段如下：

| 字段 | 含义 |
|------|------|
| `should_play_sound` | 当前这次 update 是否需要真正触发声音 |
| `loudness` | 当前应该使用的响度 |
| `loudness_level_index` | 当前响度档位索引 |
| `phase` | 控制器状态 |
| `reason` | 本次决策原因 |
| `snapshot` | 本次 update 的中间生理特征和模糊分数 |

## 3. 整体控制流程

当前控制器的处理流程可以概括为：

1. 从 `RIP` 计算当前呼吸幅度
2. 从 `RIP` 估计当前呼吸周期
3. 从 `SpO2` 计算相对个体基线的下降量
4. 根据 `RIP + SpO2` 特征做模糊推理，得到 `trigger_score`
5. 根据 `trigger_score + spo2_delta` 做第二层模糊推理，得到 `loudness_score`
6. 若 `trigger_score` 持续达到开始阈值，则进入干预
7. 干预中仍按固定 `0.6s` 节拍播放，但每次播放前会重新根据 `loudness_score` 选响度
8. 若 `trigger_score` 降到停止阈值以下，或用户按下空格键，则停止

## 4. 特征提取细节

### 4.1 RIP 幅度

当前呼吸幅度定义为：

- 取最近 `rip_amplitude_window_sec = 2.0s` 的 `RIP`
- 用 `95th percentile - 5th percentile` 作为当前幅度

这样做比直接用 `max - min` 更稳一些，能减少尖噪声的影响。

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:419)

### 4.2 RIP 幅度基线

`rip_baseline` 的来源有两种：

1. 手动设置：`controller.set_baseline(rip_amplitude=...)`
2. 自动基线：从最近 `baseline_history_sec = 180s` 的“低风险阶段”收集幅度样本，取 `80th percentile`

高分位数的目的是避免阻塞期的低幅度把基线往下拉。

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:450)

### 4.3 呼吸周期

当前呼吸周期估计方式：

- 取最近 `rip_period_window_sec = 12s` 的 `RIP`
- 分别检测峰值和谷值
- 计算相邻峰或相邻谷之间的时间差
- 使用候选周期的 `50th percentile` 作为当前周期

它不是医学上最复杂的周期估计方法，但对你这版实时原型已经足够。

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:426)

### 4.4 呼吸周期基线

`breath_period_baseline` 的来源也有两种：

1. 手动设置：`controller.set_baseline(breath_period_sec=...)`
2. 自动基线：从低风险阶段收集周期样本，取 `50th percentile`

### 4.5 SpO2 当前值和基线

`SpO2` 使用策略：

- 当前值取最近一次有效 `spo2_pct`
- 若距离当前时间超过 `spo2_stale_after_sec = 10s`，则视为无效
- 自动基线从最近 `spo2_baseline_history_sec = 120s` 的低风险阶段收集，取 `80th percentile`

相对下降量定义为：

```text
spo2_delta = max(0, spo2_baseline - current_spo2)
```

### 4.6 扰动持续时间

模糊控制里没有直接用“已经正式触发了多久”，而是先定义一个更宽松的“可疑扰动”条件：

```text
rip_amplitude_ratio < 0.80
OR breath_period_ratio > 1.10
OR spo2_delta >= 1.0
```

只要满足这个条件，就开始累计 `disturbance duration`。这个持续时间会作为模糊输入 `Persistence`。

这样做的好处是：

- 不会出现“必须先触发，才能开始累计持续时间”的循环依赖
- 可以更早反映事件正在形成

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:485)

## 5. 模糊输入变量

当前 `fuzzy-v1` 使用 4 个模糊输入：

1. `AmplitudeDrop`
   - 来自 `rip_amplitude_ratio`
2. `PeriodProlongation`
   - 来自 `breath_period_ratio`
3. `Persistence`
   - 来自异常扰动持续时间
4. `Desaturation`
   - 来自 `spo2_delta`

### 5.1 `rip_amplitude_ratio`

定义：

```text
rip_amplitude_ratio = current_rip_amplitude / rip_baseline
```

隶属函数：

- `normal`: 梯形，`[0.60, 0.72, 2.0, 2.0]`
- `mild`: 三角形，`[0.40, 0.58, 0.78]`
- `moderate`: 三角形，`[0.18, 0.36, 0.55]`
- `severe`: 梯形，`[0.0, 0.0, 0.18, 0.30]`

解释：

- 值越低，代表相对基线的呼吸幅度越差
- `severe` 更接近明显阻塞或严重低通气

### 5.2 `breath_period_ratio`

定义：

```text
breath_period_ratio = current_breath_period / breath_period_baseline
```

隶属函数：

- `normal`: 梯形，`[0.0, 0.0, 1.08, 1.18]`
- `long`: 三角形，`[1.05, 1.28, 1.50]`
- `very_long`: 梯形，`[1.35, 1.50, 6.0, 6.0]`

解释：

- 值越高，代表呼吸节律越慢、停顿越长

### 5.3 `Persistence`

单位：秒

隶属函数：

- `brief`: 梯形，`[0.0, 0.0, 2.0, 5.0]`
- `sustained`: 三角形，`[3.0, 7.0, 11.0]`
- `prolonged`: 梯形，`[8.0, 10.0, 60.0, 60.0]`

解释：

- 这是对异常持续时间的模糊表达
- 它替代了旧版本里“固定 8.5s 硬门槛”的单一角色

### 5.4 `spo2_delta`

隶属函数：

- `none`: 梯形，`[0.0, 0.0, 0.5, 1.0]`
- `mild`: 三角形，`[0.5, 1.5, 2.5]`
- `moderate`: 三角形，`[1.5, 2.75, 3.75]`
- `high`: 梯形，`[2.5, 3.5, 12.0, 12.0]`

解释：

- 它使用的是“相对个人基线下降了多少”，不是绝对血氧值

## 6. 触发分数 `trigger_score`

### 6.1 规则库

当前实现的规则如下：

- IF `AmplitudeDrop` is `severe` THEN `Risk` is `high`
- IF `AmplitudeDrop` is `severe` AND `Persistence` is `prolonged` THEN `Risk` is `very_high`
- IF `AmplitudeDrop` is `severe` AND `Desaturation` is `high` THEN `Risk` is `very_high`
- IF `AmplitudeDrop` is `moderate` AND `PeriodProlongation` is `very_long` THEN `Risk` is `high`
- IF `AmplitudeDrop` is `moderate` AND `Persistence` is `sustained` THEN `Risk` is `medium`
- IF `AmplitudeDrop` is `moderate` AND `Desaturation` is `high` THEN `Risk` is `high`
- IF `AmplitudeDrop` is `moderate` AND `Desaturation` is `moderate` THEN `Risk` is `medium`
- IF `AmplitudeDrop` is `mild` AND `Desaturation` is `moderate` THEN `Risk` is `medium`
- IF `AmplitudeDrop` is `mild` AND `Desaturation` is `high` THEN `Risk` is `high`
- IF `AmplitudeDrop` is `mild` AND `PeriodProlongation` is `long` THEN `Risk` is `low`
- IF `AmplitudeDrop` is `mild` AND `Persistence` is `sustained` THEN `Risk` is `low`
- IF `AmplitudeDrop` is `normal` AND `Desaturation` is `none` THEN `Risk` is `none`
- IF `AmplitudeDrop` is `severe` AND `Persistence` is `sustained` THEN `Risk` is `medium`

最后一条是一个保护性规则：

- 当 `RIP` 已经很差，但 `SpO2` 还没明显掉下来时，也会随着持续时间慢慢推高风险

### 6.2 解模糊

输出标签与数值映射：

- `none -> 0.00`
- `low -> 0.25`
- `medium -> 0.50`
- `high -> 0.75`
- `very_high -> 1.00`

然后做加权平均：

```text
trigger_score = sum(rule_strength * label_value) / sum(rule_strength)
```

若没有任何规则被激活，则默认 `0.0`。

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:539)

## 7. 响度分数 `loudness_score`

### 7.1 第二层输入

响度不是直接等于 `trigger_score`，而是再做一层模糊推理。

它用到：

- `trigger_score`
- `spo2_delta`

### 7.2 风险隶属函数

`trigger_score` 的风险隶属函数：

- `low`: 梯形，`[0.05, 0.18, 0.32, 0.45]`
- `medium`: 三角形，`[0.30, 0.50, 0.70]`
- `high`: 三角形，`[0.60, 0.76, 0.92]`
- `very_high`: 梯形，`[0.82, 0.92, 1.0, 1.0]`

### 7.3 规则库

- IF `Risk` is `low` THEN `Intensity` is `very_soft`
- IF `Risk` is `medium` THEN `Intensity` is `soft`
- IF `Risk` is `high` THEN `Intensity` is `medium`
- IF `Risk` is `high` AND `Desaturation` is `moderate` THEN `Intensity` is `strong`
- IF `Risk` is `high` AND `Desaturation` is `high` THEN `Intensity` is `strong`
- IF `Risk` is `very_high` THEN `Intensity` is `strong`

### 7.4 解模糊

输出标签与数值映射：

- `very_soft -> 0.18`
- `soft -> 0.35`
- `medium -> 0.60`
- `strong -> 0.90`

如果没有规则激活，则默认退回到截断后的 `trigger_score`。

对应代码：[`controller.py`](/Users/liujia/Desktop/internship/intervention-v0/controller.py:584)

## 8. 从 `loudness_score` 到实际响度

控制器最终不会直接输出 `loudness_score`，而是把它映射到 `config.loudness_levels`。

当前映射方式：

```text
index = int(clamp(loudness_score, 0, 1) * len(loudness_levels))
index 再裁剪到 [0, len(loudness_levels)-1]
```

默认响度表：

```python
(0.20, 0.28, 0.36, 0.44, 0.52, 0.60)
```

所以真正播放出去的是其中某一档，而不是连续值。

## 9. 触发、保持、停止逻辑

### 9.1 开始干预

有两条开始路径：

1. 常规开始

```text
trigger_score >= 0.70
且持续 >= 6.0s
```

2. 快速开始

```text
trigger_score >= 0.85
且持续 >= 3.0s
```

### 9.2 干预中

一旦进入 `intervening`：

- 播放节拍固定 `sound_interval_sec = 0.6`
- 每次播放前重新计算 `loudness_score`
- 响度档位可以随当前风险变化

### 9.3 停止干预

停止条件有两类：

1. 生理停止

```text
trigger_score < 0.40
```

2. 人工停止

```text
button_pressed == True
```

人工停止优先级最高，会立即进入 `user_awake`。

## 10. 状态机

当前状态枚举为：

- `monitoring`
- `event_pending`
- `intervening`
- `recovered`
- `user_awake`

含义如下：

- `monitoring`
  - 风险不足，正常监测
- `event_pending`
  - 风险已经高于启动阈值，但持续时间还没够
- `intervening`
  - 正在按固定节拍发声
- `recovered`
  - 干预中风险已下降到停止阈值以下
- `user_awake`
  - 用户按下按钮，认为已清醒

## 11. 基线更新策略

自动基线不会一直更新，而是只在“比较安全”的时候更新。

当前基线更新条件：

- `button` 没有被按下
- 当前不在 `intervening`
- `trigger_score < trigger_score_stop_threshold`

也就是说：

- 在真正高风险或正在干预时，基线冻结
- 只有低风险阶段，新的 `RIP` 幅度、呼吸周期、`SpO2` 才会被加入基线缓存

这样可以避免事件期数据污染基线。

## 12. 使用

```python
from controller import PreExperimentConfig, PreExperimentController

controller = PreExperimentController(
    PreExperimentConfig(
        rip_fs=25.0,
        loudness_levels=(0.20, 0.30, 0.40, 0.50),
    )
)

# 推荐在实验前的稳定呼吸阶段设置个体基线。
controller.set_baseline(
    rip_amplitude=2.0,
    breath_period_sec=3.0,
    spo2_pct=98.0,
)

command = controller.update(
    rip=[0.01, 0.02, 0.01, -0.01],
    button=[0, 0, 0, 0],
    spo2_pct=97.0,
    timestamp=123.4,
)

if command.should_play_sound:
    play_fixed_cue(loudness=command.loudness)
```

如果集成端只想拿唯一控制量：

```python
loudness = controller.update_loudness(
    rip=[0.01, 0.02, 0.01, -0.01],
    button=[0, 0, 1, 0],
    spo2_pct=97.0,
    timestamp=123.4,
)
```

## 13. 给播放层调用的输出参数

最小调用方式如下：

```python
command = controller.update(
    rip=sample.rip,
    button=sample.awake_button,
    spo2_pct=sample.spo2_pct,
    timestamp=sample.timestamp,
)

if command.should_play_sound:
    make_or_play_sound_cue(
        loudness=command.loudness,
    )
```

如果播放层需要 dB：

```python
def loudness_to_level_db(loudness: float) -> float:
    return -40.0 + float(loudness) * 37.0

if command.should_play_sound:
    make_or_play_sound_cue(
        level_db=loudness_to_level_db(command.loudness),
    )
```

## 14. 从胸带 DataPacket 实时读取

`data_reader.py` 会把 `/home/osa-main` 的 `chestband.data` 事件转换成
`RIP + SpO2 + button` 输入：

```python
from controller import PreExperimentController
from data_reader import ChestbandDataPacketReader

controller = PreExperimentController()
reader = ChestbandDataPacketReader(rip_fs=controller.config.rip_fs)

def on_chestband_data(ev):
    sample = reader.from_event(ev)
    if sample is None:
        return

    command = controller.update(
        rip=sample.rip,
        button=sample.awake_button,
        spo2_pct=sample.spo2_pct,
        timestamp=sample.timestamp,
    )

    if command.should_play_sound:
        play_fixed_cue(loudness=command.loudness)
```

如果想在测试里模拟按钮：

```python
reader.button_source.inject_press()
```

如果程序运行在没有 TTY 的后台环境中，空格键监听会自动退回全 0，不影响 `RIP/SpO2` 读取。

## 15. 记录数据

默认记录目录：

```text
intervention-v0/sessions/<session_id>/
├── meta.json
├── signals.csv
├── signals_0000.npz
├── cue_events.jsonl
└── summary.json
```

`signals.csv` / `signals_####.npz` 会记录：

- `rip`
- `awake_button`
- `spo2_pct`
- `rip_amplitude`
- `rip_baseline`
- `rip_amplitude_ratio`
- `breath_period_sec`
- `breath_period_baseline`
- `breath_period_ratio`
- `spo2_baseline`
- `spo2_delta`
- `trigger_score`
- `loudness_score`
- `cue_triggered`
- `loudness`
- `phase`
- `reason`

最小接入示例：

```python
from controller import PreExperimentConfig, PreExperimentController
from data_reader import ChestbandDataPacketReader
from recorder import PreExperimentRecorder, PreExperimentSessionMeta

config = PreExperimentConfig(rip_fs=25.0)
controller = PreExperimentController(config)
reader = ChestbandDataPacketReader(rip_fs=config.rip_fs)
recorder = PreExperimentRecorder(
    PreExperimentSessionMeta(
        subject_id="pilot_001",
        note="pre-experiment",
        config={
            "rip_fs": config.rip_fs,
            "loudness_levels": list(config.loudness_levels),
        },
    )
)

def on_chestband_data(ev):
    sample = reader.from_event(ev)
    if sample is None:
        return

    command = controller.update(
        rip=sample.rip,
        button=sample.awake_button,
        spo2_pct=sample.spo2_pct,
        timestamp=sample.timestamp,
    )

    cue_params = None
    if command.should_play_sound:
        cue_params = {"loudness": command.loudness}
        make_or_play_sound_cue(**cue_params)

    recorder.record_step(sample, command, cue_params=cue_params)
```

## 16. 交互仿真

目录里带了一个可交互假数据脚本 `demo.py`，它会生成：

- 正常呼吸段
- 一个 OSA 风格事件
- 事件后的恢复段

运行：

```bash
python demo.py
```

终端有焦点时，可以在看到 `phase = intervening` 之后按空格，测试 `user_awake` 分支。

如果只想自动验证按钮中断，可以：

```bash
python demo.py --auto-press-at 25 --time-scale 0
```
