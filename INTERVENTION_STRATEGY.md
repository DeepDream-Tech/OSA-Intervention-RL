# 当前干预策略说明

本文基于当前仓库中的 [controller.py](/Users/liujia/Desktop/internship/fuzzy-v1/controller.py) 实现，总结“目前实际生效”的声音干预策略。

## 1. 一句话总览

当前策略可以概括为三层：

1. 先用 `BreathFinder + AASM 风格规则 + direct low-RIP rule` 判断是否应该开始干预。
2. 开始干预后，不再用“事件窗口一消失就立刻停止”的方式，而是改用 `continue_score` 评估“是否还需要继续干预”。
3. `loudness_score` 现在直接等于 `continue_score`，响度大小由 `continue_score` 决定。

当前真正的核心链路是：

`RIP / SpO2 / BreathFinder breaths -> aasm_risk_score -> continue_score -> loudness_score -> 声音播放/停止/响度`

## 2. 输入、输出与状态

### 2.1 输入

- `RIP` 呼吸波形，支持单点或一个 packet/batch。
- `SpO2` 最新血氧值，可选。
- `button`，用户醒来确认按钮。
- `movement_detected`，当前 packet 是否有明显运动。
- `posture_changed`，体位是否变化。

### 2.2 输出

控制器只输出一件事：当前要不要播放声音，以及播放时的响度。

### 2.3 状态机

当前状态包括：

- `MONITORING`：监测中，尚未干预。
- `EVENT_PENDING`：日志态，表示已经观察到事件条件，但还没正式进入干预。
- `INTERVENING`：正在持续发声干预。
- `RECOVERED`：已恢复，停止干预。
- `USER_AWAKE`：用户按按钮，停止干预。

## 3. 信号预处理与基础特征

### 3.1 RIP 有效性判断

每次 `update()` 先检查当前 RIP batch 是否有效：

- 先平滑 RIP。
- 用平滑后信号的 `P95 - P5` 作为 span。
- 若 `span < 1e-3` 或 `span > 10000`，则认为当前 RIP 无效。

RIP 无效时：

- 当前 `_rip` 缓冲区会被清空。
- 如果正在干预，会直接停掉。
- 如果没在干预，则跳过本次判定。

### 3.2 当前 RIP 振幅

当前振幅来自最近 `2.0s` 的平滑 RIP：

- `rip_amplitude = P95 - P5`

### 3.3 当前呼吸周期

当前呼吸周期来自最近 `12.0s` 的 RIP：

- 对最近窗口内的 RIP 做平滑。
- 分别找 peak 周期和 trough 周期。
- 选择候选数更多的一组。
- 取该组周期的中位数作为 `breath_period_sec`。

### 3.4 当前 SpO2

当前血氧值取最近一个样本，但要求它距离当前时刻不超过 `10s`，否则视为陈旧、按 `None` 处理。

## 4. Baseline 的当前计算方式

### 4.1 RIP baseline

当前用于事件检测的 RIP baseline 不是直接从原始波形做长窗统计，而是来自 `BreathFinder` 提取出的 breath 振幅：

- Breath baseline 窗口：事件时刻前 `120s` 到前 `5s`
- 只使用 `posture_changed` 之后允许纳入 baseline 的 breaths
- 若窗口内 breath 数量至少 `8` 个，取振幅中位数
- 否则 fallback 到最近 `12` 个 breath
- fallback 至少要有 `5` 个 breath

也就是说，当前 `rip_baseline` 是“breath 级振幅 baseline”。

### 4.2 呼吸周期 baseline

呼吸周期 baseline 来自 `_rip_baseline_periods`：

- 只要当前 snapshot 算出了 `breath_period_sec`，就会把它加入历史
- 历史窗口是 `180s`
- baseline 取历史周期的中位数

注意：当前版本没有再用 score 或 safe-window 去严格限制这个 baseline 何时可更新，它基本是持续更新的。

### 4.3 SpO2 baseline

当前用于触发和评分的 SpO2 baseline 为：

- 观察窗口：当前时刻前 `60s` 到前 `3s`
- 若该窗口内样本数至少 `5` 个，取中位数
- 否则 fallback 到“最近 `12` 个、且时间不晚于 `upper=t-3s` 的样本”
- fallback 至少要有 `5` 个样本

然后：

- `spo2_delta = max(0, spo2_baseline - current_spo2)`

## 5. AASM 风格事件检测链路

这一部分是“开始干预”的上游。

### 5.1 BreathFinder 呼吸提取

只要同时满足：

- RIP 信号有效
- `movement_detected=False`
- `BreathFinder` 可用

控制器就会推进 AASM 检测器。

Breath 提取的关键参数：

- 每 `10s` 更新一次 BreathFinder
- 每次给 BreathFinder 最近 `16s` 的 RIP buffer
- 离当前太近的 breath 会丢弃，边界保护是 `3s`

每个 breath 会记录：

- `start`
- `end`
- `amp`
- `baseline`
- `ratio = amp / baseline`
- `confidence`

### 5.2 低幅 breath run 合并

当前会把相邻的低幅 breath 按 gap 合并成 run：

- merge gap：`2s`

后续的 pending event 和 direct trigger 都来自这个 run。

### 5.3 AASM pending event

若某个 breath 的 `ratio <= 0.70`，它会被视为 low幅 breath 候选。

这些 breath 合并成 run 后，若：

- run 持续时间至少 `10s`

则形成一个 pending AASM event。

强弱分级：

- `min_ratio <= 0.70` -> `strong`
- 否则 -> `weak`

另外还会避免重复建事件：

- 如果与已有 pending/confirmed event 重叠
- 或者与已有事件的起止距离在 merge gap `8s` 内

则不会重复创建。

### 5.4 direct low-RIP trigger

除了 AASM pending event 之外，当前还有一条更直接的启动链路：

- 若 breath `ratio <= 0.90`
- 合并成 active run
- 且最近一个 low幅 breath 结束时间离当前不超过 `3.3s`
- 且该 run 至少包含 `3` 个 low幅 breaths
- 且当前 `rip_amplitude_ratio <= 0.85`
- 且这个 run 持续至少 `10s`

则可以形成 direct low-RIP event。

这条链路不等 SpO2 确认，只要 run 满足条件，就会直接准备触发干预。

为了避免同一段事件反复触发，还会记录历史 direct event；若与历史事件在 `8s` merge gap 范围内重叠，就不会重复发起。

### 5.5 SpO2 确认 pending AASM event

pending AASM event 不会立刻触发声音，而是要等待 SpO2 确认。

确认方式：

- lookahead：事件结束后再看 `60s`
- 先计算该事件的 SpO2 baseline
- threshold = `baseline - 3%`
- 在 `event_start ~ event_end+60s` 的样本中寻找低于 threshold 的持续下降区间
- 相邻低于 threshold 的样本如果间隔不超过 `1.5s`，视作同一段
- 选择最长的一段 desaturation
- desaturation 持续时间至少 `10s` 才算确认成功

确认成功后，事件会变成 confirmed AASM event，并在下一次控制器判断中开始干预。

## 6. 当前有哪些“事件条件”

当前代码里有三个相关概念：

### 6.1 `aasm_event_condition_met`

表示已经有 confirmed AASM event。

### 6.2 `direct_trigger_met`

表示当前存在 active direct low-RIP event。

### 6.3 `event_condition_met`

这是更上层的并集条件：

- `event_condition_met = aasm_event_condition_met or direct_trigger_met`

它主要用于给 `continue_score` 提供“当前是否明确处于事件期”的上下文。

## 7. `aasm_risk_score` 的当前计算

`aasm_risk_score` 是当前的原始风险分数，代码中也会被放进：

- `raw_risk_score`
- `SensorSnapshot.raw_trigger_score`（兼容旧字段名）

它不是当前唯一的控制量，但它会给 `continue_score` 提供基础地板。

### 7.1 输入

`aasm_risk_score` 依赖：

- `rip_amplitude_ratio`
- `low_rip_run_duration_sec`
- `spo2_delta`
- `candidate_strength`
- `event_confirmed`

### 7.2 规则

#### 规则 A：看 RIP ratio

- `ratio <= 0.70` -> score 至少 `0.82`
- `0.70 < ratio <= 0.80` -> score 在 `0.58 ~ 0.78` 之间线性变化
- `0.80 < ratio <= 0.90` -> score 在 `0.35 ~ 0.55` 之间线性变化

#### 规则 B：看低幅 run 持续时间

若：

- `duration >= 10s`
- 且 `ratio <= 0.90`

则额外保证：

- score 至少在 `0.68 ~ 0.82` 之间
- 其中 `10s -> 0.68`
- `13s` 左右封顶到 `0.82`

这里用到的上升时间常数是 `fast_trigger_duration_sec = 3.0s`。

#### 规则 C：看 strong 候选

- 若 `candidate_strength == "strong"`，score 至少 `0.85`

#### 规则 D：看血氧下降

若：

- `spo2_delta >= 3.0`

则 score 至少：

- `0.80 + 0.15 * extra`
- 其中 `extra = min((spo2_delta - 3.0) / 2.0, 1.0)`

所以：

- `3%` 下降对应至少 `0.80`
- `5%` 下降对应至少 `0.95`

#### 规则 E：看事件是否已确认

若 `event_confirmed=True`：

- strong event 至少 `0.85`
- weak event 至少 `0.75`

### 7.3 结果范围

最终 `aasm_risk_score` 会被裁剪到 `[0, 1]`。

## 8. `continue_score`：当前真正的持续干预核心

当前开始干预后，是否继续、何时停止，主要看 `continue_score`。

同时：

- `SensorSnapshot.trigger_score` 现在实际返回的是 `continue_score`
- `loudness_score = continue_score`

也就是说，旧的“trigger_score”在当前语义里已经基本变成了“continue_score”。

### 8.1 总体设计思想

`continue_score` 不是只看一拍的事件触发，而是综合看：

- 当前 RIP 振幅是否仍低
- 呼吸周期是否仍拉长
- 低幅呼吸 run 是否还在持续
- SpO2 是否仍在下降
- 最近是否刚发生过确认事件

这样做的目的是：

- 允许先触发 cue
- 然后用一个更慢、更稳的 score 来维持干预
- 直到真正恢复，而不是事件窗口一消失就立刻停

### 8.2 五个组成项

#### 1. `amp_need`

由 `rip_amplitude_ratio` 决定：

- `<= 0.30` -> `1.00`
- `0.30 ~ 0.45` -> `1.00 -> 0.80`
- `0.45 ~ 0.60` -> `0.80 -> 0.50`
- `0.60 ~ 0.75` -> `0.50 -> 0.20`
- `0.75 ~ 0.90` -> `0.20 -> 0.00`
- `> 0.90` -> `0.00`

#### 2. `period_need`

由 `breath_period_ratio` 决定：

- `<= 1.05` -> `0.00`
- `1.05 ~ 1.20` -> `0.00 -> 0.35`
- `1.20 ~ 1.35` -> `0.35 -> 0.70`
- `1.35 ~ 1.55` -> `0.70 -> 1.00`
- `> 1.55` -> `1.00`

#### 3. `run_need`

由 `low_rip_run_duration_sec` 决定：

- `<= 2s` -> `0.00`
- `2 ~ 4s` -> `0.00 -> 0.30`
- `4 ~ 10s` -> `0.30 -> 0.70`
- `10 ~ 16s` -> `0.70 -> 1.00`
- `> 16s` -> `1.00`

#### 4. `spo2_need`

由 `spo2_delta` 决定：

- `<= 1%` -> `0.00`
- `1 ~ 3%` -> `0.00 -> 0.60`
- `3 ~ 5%` -> `0.60 -> 1.00`
- `> 5%` -> `1.00`

#### 5. `memory_need`

由最近一次事件记忆决定：

- 当 `event_condition_met=True` 时，`memory_at = now`
- 之后按指数衰减：

`memory_need = continue_score_memory_peak * exp(-(now - memory_at) / continue_score_memory_decay_sec)`

当前参数下就是：

`memory_need = 1.0 * exp(-dt / 10s)`

### 8.3 基础加权公式

当前 `continue_score` 的 target 为：

```text
target =
  0.40 * amp_need
  + 0.15 * period_need
  + 0.15 * run_need
  + 0.15 * spo2_need
  + 0.15 * memory_need
```

### 8.4 原始风险地板

在上面的基础上，再加一个原始风险地板：

```text
target >= 0.25 * raw_risk_score
```

也就是说，即使五个分项暂时不高，`aasm_risk_score` 仍然会给 `continue_score` 一个最低托底。

### 8.5 事件期地板

如果当前 `event_condition_met=True`：

- strong -> `target >= 0.82`
- weak -> `target >= 0.72`

如果当前虽然还没 confirmed，但低幅 run 已经持续至少 `10s`：

- `target >= 0.55`

这会让干预在事件刚结束后不会瞬间塌到很低。

### 8.6 artifact cap

若当前满足：

- RIP 振幅很低
- 但没有稳定周期信息
- 且 SpO2 下降也不足以补救

则认为更像可疑信号而非真实低通气。

此时会把 `continue_score` target 强行压到停止阈值以下：

```text
target <= continue_score_stop_threshold - 1e-3
```

当前即约等于：

```text
target <= 0.299
```

### 8.7 快速上升、慢速下降

`continue_score` 不是完全瞬时的。

如果新的 `target >= previous_score`：

- 立即上升到 target

如果新的 `target < previous_score`：

- 不会立刻掉下去
- 而是用指数形式慢慢衰减

公式为：

```text
alpha = 1 - exp(-elapsed / continue_score_decay_sec)
score = previous + alpha * (target - previous)
```

当前 `continue_score_decay_sec = 8.0s`。

所以现在的行为是：

- 恶化时，score 升得快
- 恢复时，score 掉得慢

## 9. 启动干预的当前规则

当前只有两条真正会把状态推进到 `INTERVENING` 的路径。

### 9.1 路径一：direct low-RIP trigger

只要存在一个 active low-RIP run：

- `ratio <= 0.90`
- 持续至少 `10s`

则会产生 `_pending_direct_trigger`，并在本次 `update()` 里直接：

- `_start_intervention()`
- 立刻播放第一声 cue

### 9.2 路径二：AASM pending event 被 SpO2 确认

只要 pending AASM event 在 `event_end + 60s` 的观察期内满足：

- `SpO2` 较 baseline 持续下降至少 `3%`
- 且 desaturation 持续至少 `10s`

就会生成 `_pending_confirmed_trigger`，然后在本次 `update()` 里直接：

- `_start_intervention()`
- 立刻播放第一声 cue

### 9.3 当前没有使用“最短干预时间”

当前版本没有强制的 minimum intervention duration。

也就是说：

- 只要启动了，就进入 `continue_score` 驱动的继续/停止逻辑
- 不会被一个“至少先播几秒”的硬规则锁住

## 10. 干预进行中的继续逻辑

一旦进入 `INTERVENING`：

- 每 `0.6s` 最多播放一次声音
- 是否继续不再直接看“AASM 事件窗口还在不在”
- 而是主要看 `continue_score` 和恢复判据

如果还没到 `0.6s` 的固定播放间隔：

- 不播放
- 但状态仍保持 `INTERVENING`

## 11. 停止干预的当前规则

### 11.1 立即停止的情况

以下情况会立刻停：

- RIP 信号无效
- 按钮确认用户已醒
- 当前被判定为 artifact-like low amplitude

### 11.2 恢复停止：`recovery_ready`

当前定义了一个“恢复准备好”的条件 `_recovery_ready(snapshot)`。

必须同时满足：

- `continue_score <= 0.30`
- `rip_amplitude_ratio >= 0.72`
- `low_rip_run_duration_sec <= 2.0s`
- 如果 `breath_period_ratio` 可用，还要求 `breath_period_ratio <= 1.18`

### 11.3 恢复保持时间

不是一满足恢复条件就立刻停，而是要连续保持：

- `continue_score_recovery_hold_sec = 10.0s`

只有连续 `10s` 都满足 `_recovery_ready`，才会判定：

- “呼吸已连续恢复稳定，停止声音干预”

### 11.4 回落停止：没有事件且 score 已很低

还有一条更直接的兜底停止规则：

若：

- `event_condition_met=False`
- 且 `continue_score <= 0.30`

则直接停止，理由是：

- “干预需求已明显回落”

这条规则的意义是：

- 如果事件已经不在了
- 且 `continue_score` 也已经掉到很低
- 就不必再等恢复 hold 满足

## 12. 响度控制逻辑

### 12.1 当前 `loudness_score`

当前：

```text
loudness_score = clamp(continue_score, 0, 1)
```

所以 `loudness_score` 本质就是 `continue_score`。

### 12.2 响度档位

当前共有 6 档：

- index 0 -> `0.20`
- index 1 -> `0.28`
- index 2 -> `0.36`
- index 3 -> `0.44`
- index 4 -> `0.52`
- index 5 -> `0.60`

### 12.3 初始响度 floor

进入干预时，至少从 `loudness_initial_level_index = 1` 开始，也就是至少从 `0.28` 开始。

同时有更高风险 floor：

- 若 `continue_score >= 0.85` 或 `spo2_delta >= 2.5`，至少升到 index `2`，即 `0.36`
- 若 `continue_score >= 0.92` 或 `spo2_delta >= 3.5`，至少升到 index `3`，即 `0.44`

### 12.4 score 到 ceiling 的映射

当前 ceiling index 由下面方式得到：

```text
index = int(clamp(score, 0, 1) * 6)
```

再裁剪到 `[0, 5]`。

因此大致对应：

- `0.00 ~ 0.166` -> 0
- `0.166 ~ 0.333` -> 1
- `0.333 ~ 0.500` -> 2
- `0.500 ~ 0.666` -> 3
- `0.666 ~ 0.833` -> 4
- `0.833 ~ 1.000` -> 5

实际允许的 ceiling 还会再和 floor 取 `max`。

### 12.5 干预过程中的响度更新

每播放 `3` 声，会评估一次是否升档或降档。

#### 降档条件：`recovering`

把当前 snapshot 和上一个评估点 anchor 对比：

- `loudness_score` 下降至少 `0.08`
- `rip_amplitude_ratio` 上升至少 `0.08`
- `spo2_delta` 下降至少 `0.5`

若：

- 至少有 `2` 个恢复信号
- 且没有任何恶化信号

则本个评估窗记为 `recovering`。

若连续 `2` 个评估窗都 `recovering`：

- 响度下降 1 档
- 但不会低于初始档 index `1`

#### 升档条件：`no_recovery`

若：

- 完全没有恢复信号
- 或者存在任意恶化信号

则记为 `no_recovery`。

此时：

- 连续恢复计数清零
- 若当前响度还低于 ceiling，则升 1 档

#### hold 条件

如果既不是明显恢复，也不是明确未恢复，则保持当前档位不变。

### 12.6 恶化阈值

当前恶化判断阈值为：

- `loudness_score` 上升至少 `0.05`
- `rip_amplitude_ratio` 下降至少 `0.05`
- `spo2_delta` 上升至少 `0.5`

## 13. 关键参数与当前默认值

### 13.1 触发与检测相关

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `rip_fs` | `25.0` | RIP 采样率 |
| `rip_amplitude_window_sec` | `2.0` | 当前振幅计算窗口 |
| `rip_period_window_sec` | `12.0` | 当前呼吸周期计算窗口 |
| `spo2_stale_after_sec` | `10.0` | SpO2 超过多久视为陈旧 |
| `aasm_breath_buffer_sec` | `16.0` | BreathFinder 每次处理的 RIP buffer 长度 |
| `aasm_breath_edge_guard_sec` | `3.0` | 太靠近当前时刻的 breath 不采纳 |
| `aasm_breathfinder_update_sec` | `10.0` | BreathFinder 更新间隔 |
| `aasm_rip_baseline_window_sec` | `120.0` | breath 振幅 baseline 的主窗口 |
| `aasm_rip_baseline_guard_sec` | `5.0` | RIP baseline 对当前时刻的保护间隔 |
| `aasm_rip_min_baseline_breaths` | `8` | 主窗口最少 breath 数 |
| `aasm_rip_fallback_breath_count` | `12` | RIP baseline fallback breath 数 |
| `aasm_rip_fallback_min_breaths` | `5` | RIP baseline fallback 至少 breath 数 |
| `aasm_drop_ratio_strong` | `0.70` | strong 低幅阈值 |
| `aasm_drop_ratio_weak` | `0.80` | weak 低幅阈值 |
| `aasm_pending_event_ratio_threshold` | `0.70` | pending event 的 low幅 breath 候选阈值 |
| `aasm_direct_trigger_ratio` | `0.90` | direct low-RIP 低幅阈值 |
| `aasm_min_low_rip_duration_sec` | `10.0` | 低幅 run 至少持续多久才成事件 |
| `aasm_low_rip_merge_gap_sec` | `2.0` | 低幅 breath 合并 gap |
| `aasm_direct_active_grace_sec` | `3.3` | direct trigger 里 active run 的保活窗口 |
| `aasm_direct_min_breath_count` | `3` | direct trigger 至少需要多少个 low幅 breaths |
| `aasm_direct_current_rip_ratio_max` | `0.85` | direct trigger 时当前 RIP ratio 仍需不高于该值 |
| `aasm_event_merge_gap_sec` | `8.0` | 防重复事件的 merge gap |
| `aasm_spo2_baseline_window_sec` | `60.0` | SpO2 baseline 主窗口 |
| `aasm_spo2_baseline_gap_sec` | `3.0` | SpO2 baseline 对当前时刻的保护间隔 |
| `aasm_spo2_min_baseline_samples` | `5` | SpO2 主窗口最少样本数 |
| `aasm_spo2_fallback_sample_count` | `12` | SpO2 baseline fallback 样本数 |
| `aasm_spo2_fallback_min_samples` | `5` | SpO2 baseline fallback 至少样本数 |
| `aasm_spo2_lookahead_sec` | `60.0` | AASM 事件确认的 SpO2 观察窗口 |
| `aasm_spo2_drop_threshold_pct` | `3.0` | SpO2 下降确认阈值 |
| `aasm_spo2_min_desat_duration_sec` | `10.0` | desaturation 至少持续时间 |

### 13.2 continue score 相关

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `continue_score_stop_threshold` | `0.30` | continue score 的停止阈值 |
| `continue_score_decay_sec` | `8.0` | continue score 下降的时间常数 |
| `continue_score_memory_peak` | `1.0` | 事件记忆项峰值 |
| `continue_score_memory_decay_sec` | `10.0` | 事件记忆项衰减时间常数 |
| `continue_score_recovery_hold_sec` | `10.0` | 恢复判据需连续满足多久才停 |
| `continue_score_recovery_rip_ratio_min` | `0.72` | 恢复时要求的最小 RIP ratio |
| `continue_score_recovery_period_ratio_max` | `1.18` | 恢复时允许的最大周期 ratio |
| `continue_score_recovery_low_rip_run_max_sec` | `2.0` | 恢复时允许的最长低幅 run |

### 13.3 响度相关

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `sound_interval_sec` | `0.6` | 两次播放间隔 |
| `loudness_levels` | `(0.20, 0.28, 0.36, 0.44, 0.52, 0.60)` | 6 个响度档位 |
| `loudness_eval_window_sounds` | `3` | 每几声评估一次响度 |
| `loudness_recovery_windows_for_step_down` | `2` | 连续几个恢复窗后降档 |
| `loudness_initial_level_index` | `1` | 初始档位下限 |
| `loudness_high_risk_floor_index` | `2` | 高风险最小档位 |
| `loudness_very_high_floor_index` | `3` | 很高风险最小档位 |
| `high_risk_trigger_score_threshold` | `0.85` | 高风险响度 floor 阈值 |
| `loudness_very_high_trigger_threshold` | `0.92` | 很高风险响度 floor 阈值 |
| `loudness_trigger_recovery_delta` | `0.08` | loudness_score 恢复阈值 |
| `loudness_trigger_worsen_delta` | `0.05` | loudness_score 恶化阈值 |
| `loudness_rip_ratio_recovery_delta` | `0.08` | RIP ratio 恢复阈值 |
| `loudness_rip_ratio_worsen_delta` | `0.05` | RIP ratio 恶化阈值 |
| `loudness_spo2_recovery_delta` | `0.5` | SpO2 delta 恢复阈值 |
| `loudness_spo2_worsen_delta` | `0.5` | SpO2 delta 恶化阈值 |

### 13.4 辅助判据相关

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `airflow_drop_threshold_fraction` | `0.70` | 旧式 apnea-like drop 阈值，等效 ratio 阈值 `0.30` |
| `period_missing_spo2_rescue_threshold` | `2.0` | 无周期信息时，SpO2 至少下降多少才不被当成 artifact |
| `spo2_hypopnea_threshold_pct` | `3.0` | apnea/hypopnea 辅助判断里的 SpO2 阈值 |
| `rip_smoothing_window_sec` | `0.44` | RIP 平滑窗口 |
| `min_valid_rip_span` | `1e-3` | 最小有效 RIP span |
| `max_valid_rip_span` | `10000.0` | 最大有效 RIP span |

## 14. 当前仍保留但不是主决策链路的部分

下面这些内容仍留在代码里，但当前主策略基本不依赖它们：

- 旧的 `_trigger_score(...)` 模糊规则函数仍在文件中，但当前没有被主流程调用。
- `SensorSnapshot.trigger_score` 这个字段名还在，但现在返回的是 `continue_score`。
- `SensorSnapshot.raw_trigger_score` 这个字段名还在，但现在返回的是 `raw_risk_score`。
- `trigger_score_threshold = 0.70` 当前不再作为主触发阈值使用。
- `trigger_score_stop_threshold = 0.40` 当前不再作为主停止阈值使用。
- `baseline_update_trigger_score_max = 0.25` 当前没有参与 baseline 更新 gating。
- `airflow_soft_gate_upper_ratio = 0.45` 对应的 `_soft_gate_cap_for_ratio()` 目前不在主路径里。
- `consensus_*` 参数和 `_windowed_consensus_met()` 当前没有进入主决策链路。
- `trigger_duration_sec`、`apnea_min_duration_sec` 等旧触发参数目前主要属于历史保留。

简化地说：

- 当前真正生效的是 `aasm_risk_score + continue_score + direct/AASM start + recovery stop`
- 旧的 fuzzy trigger 体系现在更多是兼容保留和日志遗留

## 15. 体位变化、运动与按钮的特殊处理

### 15.1 `posture_changed`

体位变化时会：

- 记录 posture change 时间
- 清掉 change 前 `10s` 内的 recent breaths
- 清空 pending AASM events
- 清空 direct trigger / pending trigger
- 让 RIP baseline 至少在 change 后 `10s` 内不使用这些 breaths

### 15.2 `movement_detected`

若当前 packet 标记了 movement：

- AASM 检测推进会跳过这次更新

但注意：

- 当前并不会因为 `movement_detected=True` 就强制停止已开始的干预

### 15.3 `button`

只要当前 button 样本里有任意值 `>= 0.5`：

- 就认为用户已醒
- 立刻停止干预
- 同时清空 continue-score 状态和若干触发状态

## 16. 当前实现上的几个重要注意点

### 16.1 `BreathFinder` 是关键依赖

当前 AASM 风格检测和 direct low-RIP trigger 都依赖 `BreathFinder`。

如果 `_BREATHFINDER is None`：

- `_advance_aasm_detector()` 会直接返回
- breath 提取、pending event、confirmed event、direct trigger 都不会正常推进

这意味着：

- 没有 `BreathFinder` 时，当前主触发链路实际上是不完整的

### 16.2 日志/绘图里仍有部分旧字段名

当前语义已经改成：

- `trigger_score -> continue_score`

但在部分日志、回放、绘图代码里，还保留了旧字段名用于兼容。

因此阅读 CSV 或图时要注意：

- 名字可能还是 `trigger_score`
- 但语义上它已经是在表示 `continue_score`

## 17. 最后的结论

当前版本的策略重点已经从“触发后只要事件窗口结束就停”切换为：

1. 用 direct low-RIP 或 AASM+SpO2 confirmation 开始干预。
2. 用 `continue_score` 把干预维持住。
3. 只有在恢复足够明确，或者 continue score 已明显回落时，才停止。

这正对应你前面提出的思路：

- 触发可以比较果断
- 停止要更谨慎，要看一段恢复过程
- 而不是把 stop 绑定在一个短暂、瞬时的事件标签上
