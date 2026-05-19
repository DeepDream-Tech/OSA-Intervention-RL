# 音量与分贝标定说明

本文档说明如何把控制器输出的相对音量 `loudness`，映射成真实可控的播放音量。

适用范围：

- 当前 `intervention-v0` 控制器
- 控制器只输出 `loudness`
- 声音内容、播放器、设备、系统音量都由外部播放层负责

## 1. 先说结论

控制器输出的 `loudness` 不是“真实分贝”，它只是一个相对控制量。

你不能直接从：

```text
loudness = 0.44
```

推出：

```text
真实声压 = 47 dB
```

因为真实音量取决于整条播放链路：

- 音频文件本身的电平
- 播放软件的增益
- 系统音量
- 声卡
- 耳机/音箱
- 被试耳边的实际声压

所以真正要做的不是“计算分贝”，而是“标定分贝”。

## 2. 这几个量分别是什么

### 2.1 `loudness`

`loudness` 是控制器输出的相对量。

在当前项目里，它来自：

- 模糊控制器的 `loudness_score`
- 再映射到 `config.loudness_levels`

例如：

```python
loudness_levels = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60)
```

控制器最终只会输出这些档位里的某一个。

### 2.2 `dBFS`

`dBFS` 是数字音频内部的电平单位，描述的是“离满刻度还有多远”。

它不等于真实世界里的声音大小。

### 2.3 `dB SPL`

`dB SPL` 是空气中的真实声压级，才是你实验里真正关心的“声音有多大”。

如果你想控制被试听到的实际音量，最终必须落到 `dB SPL` 的标定上。

## 3. 为什么不能直接从 `loudness` 算出真实分贝

因为同一个 `loudness=0.44`，在不同条件下会得到完全不同的真实声压：

- 不同耳机
- 不同电脑
- 不同系统音量
- 不同 cue 文件
- 不同播放器增益

所以：

```text
controller.loudness -> 真实 dB SPL
```

不是理论公式问题，而是设备标定问题。

## 4. 推荐的工程做法

最稳妥的做法是把问题拆成两层：

1. 控制器层：只输出 `loudness`
2. 播放层：负责把 `loudness` 映射成目标 `dB SPL` 或数字增益

也就是说：

```text
controller.loudness
-> 播放层查标定表
-> 目标 dB SPL / 增益 dB
-> 实际播放
```

## 5. 推荐标定流程

### 第一步：固定播放链路

正式标定前，必须先把下面这些固定住：

- 播放设备
- 耳机/耳塞/音箱型号
- 声卡
- 系统音量
- 播放软件
- 采样率
- cue 文件版本

只要这里有一个改了，原来的标定就可能失效。

### 第二步：固定 cue 内容

最好先把要用的 cue 文件做成同一批、同一处理方式。

推荐至少保证：

- 文件格式一致
- 采样率一致
- 时长一致或同级别
- 粗略响度一致

如果不同 cue 本身电平差异很大，就算控制器输出同一个 `loudness`，实际响度也会飘。

最简单的策略是：

- 预实验阶段先只用一个固定 cue

这样标定最稳定。

### 第三步：选择参考测量方式

推荐两种方式：

1. 声级计
2. 校准过的测量麦克风

测量位置要固定：

- 如果是扬声器，固定在被试耳边位置
- 如果是耳机，最好用耳机耦合器或稳定的耳位替代方案

### 第四步：选一个参考播放点

比如先定义：

- cue 文件：`cue_fixed.wav`
- 数字增益：`0 dB`
- 播放链路全部固定

然后实测一次，得到：

```text
reference_spl_db = 42 dB SPL
```

这个值就是后面推导其他增益的参考点。

### 第五步：给每个 `loudness` 档位设目标 SPL

例如可以先做一张目标表：

| loudness | target_dB_SPL |
|----------|---------------|
| 0.20 | 38 |
| 0.28 | 41 |
| 0.36 | 44 |
| 0.44 | 47 |
| 0.52 | 50 |
| 0.60 | 53 |

这张表不是公式推出来的，而是实验设计决定的。

然后你再根据参考测量值，算每一档要加多少数字增益。

### 第六步：把目标 SPL 转成数字增益

如果你已经知道：

- 参考点：`scale = 1.0` 时测得 `42 dB SPL`
- 目标：希望某档达到 `47 dB SPL`

那么需要的增益差就是：

```text
delta_db = target_db_spl - reference_db_spl
         = 47 - 42
         = 5 dB
```

再把它换成线性振幅倍率：

```text
amplitude_scale = 10 ** (delta_db / 20)
```

例如：

```text
10 ** (5 / 20) ≈ 1.778
```

也就是说，这一档需要把播放振幅乘以约 `1.778`。

## 6. 常用公式

### 6.1 振幅倍率转 dB

```text
gain_db = 20 * log10(amplitude_scale)
```

### 6.2 dB 转振幅倍率

```text
amplitude_scale = 10 ** (gain_db / 20)
```

### 6.3 参考 SPL 推目标增益

```text
gain_db = target_spl_db - reference_spl_db
```

注意：这个公式成立的前提是播放链路、cue 文件和测量位置都固定。

## 7. 一个可直接用的标定表模板

建议你最终维护一张这样的表：

| loudness | target_dB_SPL | reference_dB_SPL | gain_dB | amplitude_scale | cue_name | notes |
|----------|---------------|------------------|---------|-----------------|----------|-------|
| 0.20 | 38 | 42 | -4 | 0.631 | cue_fixed.wav | baseline |
| 0.28 | 41 | 42 | -1 | 0.891 | cue_fixed.wav |  |
| 0.36 | 44 | 42 | 2 | 1.259 | cue_fixed.wav |  |
| 0.44 | 47 | 42 | 5 | 1.778 | cue_fixed.wav |  |
| 0.52 | 50 | 42 | 8 | 2.512 | cue_fixed.wav | check clipping |
| 0.60 | 53 | 42 | 11 | 3.548 | cue_fixed.wav | high-risk only |

这张表里的数字只是示例，不代表推荐实验值。

## 8. 推荐在代码里怎么落地

最简单的是在播放层写查表逻辑，不把标定逻辑塞进控制器。

例如：

```python
LOUDNESS_TO_SPL = {
    0.20: 38.0,
    0.28: 41.0,
    0.36: 44.0,
    0.44: 47.0,
    0.52: 50.0,
    0.60: 53.0,
}

LOUDNESS_TO_GAIN_DB = {
    0.20: -4.0,
    0.28: -1.0,
    0.36:  2.0,
    0.44:  5.0,
    0.52:  8.0,
    0.60: 11.0,
}

def gain_db_to_scale(gain_db: float) -> float:
    return 10 ** (gain_db / 20.0)

def loudness_to_playback_scale(loudness: float) -> float:
    gain_db = LOUDNESS_TO_GAIN_DB[float(loudness)]
    return gain_db_to_scale(gain_db)
```

播放时：

```python
if command.should_play_sound:
    scale = loudness_to_playback_scale(command.loudness)
    play_fixed_cue(scale=scale)
```

## 9. 更推荐的记录方式

如果你后面要复盘实验，建议播放层把这些值一起记进 `cue_params`：

- `loudness`
- `target_dB_SPL`
- `gain_dB`
- `amplitude_scale`
- `cue_name`
- `device_name`

例如：

```python
cue_params = {
    "loudness": command.loudness,
    "target_dB_SPL": 47.0,
    "gain_dB": 5.0,
    "amplitude_scale": 1.778,
    "cue_name": "cue_fixed.wav",
    "device_name": "headphone_A",
}
```

这样 `cue_events.jsonl` 里就会同时保留：

- 控制器决定的相对音量
- 播放层实际使用的标定参数

## 10. 最小标定流程清单

如果你想尽快做出第一版，建议按这个最小流程走：

1. 固定一个 cue 文件
2. 固定一个播放设备和系统音量
3. 测出 `scale=1.0` 时的参考 `dB SPL`
4. 给 6 个 `loudness_levels` 各指定一个目标 `dB SPL`
5. 计算每一档的 `gain_dB` 和 `amplitude_scale`
6. 实际播放复测
7. 修正表格，直到每档都接近期望值

## 11. 常见错误

### 错误 1：把 `loudness` 当成真实分贝

这是最常见的误区。

`loudness` 只是控制器输出，不是设备无关的物理量。

### 错误 2：换了耳机还沿用旧标定

只要耳机、音箱、系统音量、播放器有变化，就要重新确认标定。

### 错误 3：不同 cue 文件不做预处理

如果 cue A 本身就比 cue B 响很多，那么同一个 `loudness` 映射过去也不会等响。

### 错误 4：只看软件增益，不测真实声压

软件里的 `+6 dB` 不等于被试耳边一定就是某个固定 `dB SPL`。

## 12. 对当前项目的推荐结论

对 `intervention-v0`，我最推荐的做法是：

1. 控制器继续只输出 `loudness`
2. 播放层维护一张 `loudness -> target_dB_SPL -> gain_dB` 标定表
3. 用固定 cue 先完成第一轮标定
4. 把实际播放参数写进 `cue_params`

这样职责最清晰：

- `controller.py` 负责决策
- 播放层负责真实音量
- `recorder.py` 负责把两边信息都记下来

## 13. 一句话总结

真实分贝不是从控制器里“算出来”的，而是通过固定播放链路、实测参考声压、建立 `loudness -> gain / dB SPL` 标定表来确定的。
