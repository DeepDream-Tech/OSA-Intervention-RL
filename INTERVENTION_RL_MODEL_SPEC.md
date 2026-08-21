# OSA 声学干预强化学习建模规范

> 版本：v1.1  
> 日期：2026-07-20  
> 数学格式：使用 GitHub/KaTeX 兼容的 `$...$` 和 `$$...$$` 定界符  
> 状态：设计基线，所有标记为 `[TBD-CAL]` 的声学或临床阈值必须在设备标定和伦理审批后确定

## 1. 目标与系统边界

本系统不再判断“未来是否会发生 OSA”。该职责完全属于外部 Audio OSA 预测模型。本系统仅在收到外部触发后，利用 Audio、PPG 和 IMU 的实时观测，选择干预声音的响度，并持续判断用户是否恢复。

系统优化目标为：

$$
\text{在阻止或缓解 OSA 风险的同时，最小化响度、累计声音剂量和用户觉醒概率。}
$$

明确的职责边界如下：

1. 外部模型输出 OSA 风险触发；本项目不重复实现 OSA 四分类器。
2. 干预策略唯一可学习的动作是响度；波形、频率、时长和播放节奏均由固定协议配置。
3. 正常停止只有两种业务原因：用户清醒后按下空格键，或生理指标连续恢复正常。
4. 安全超时、传感器故障和声学剂量越界属于保护性中止，不计为成功恢复。
5. 用户是否清醒不由传感器推断，以空格键事件作为人工确认标签。

### 1.1 必要假设

- Audio 采样率为 $f_A=8000\,\text{Hz}$。
- PPG 采样率为 $f_P=25\,\text{Hz}$。
- IMU 采样率为 $f_I=33\,\text{Hz}$。
- 所有数据包包含单调时钟时间戳；不能只依靠样本计数对齐不同设备时钟。
- 外部模型至少输出布尔触发，最好同时输出风险分数和预测时间范围。
- 单路 PPG 波形不能直接产生可信的 SpO2。只有设备提供红光/红外双波长数据或已计算的 SpO2 时，才能将血氧加入本规范。
- 归一化数字幅度不是听力安全单位。部署前必须完成“数字增益到耳道 dB SPL”的设备标定。

### 1.2 核心符号

| 符号 | 含义 |
|---|---|
| $t$ | 连续时间 |
| $k$ | 策略决策步编号 |
| $q_t$ | 控制状态机状态 |
| $p_t$ | 外部模型的最新有效风险分数 |
| $m_t^m$ | 模态 $m$ 的可用性掩码 |
| $H_t$ | 综合生理负担 |
| $a_k$ | 策略请求的归一化响度 |
| $\hat a_k$ | 安全层实际执行的归一化响度 |
| $L_k^{exec}$ | 实际执行声压级，仅在非静音时定义 |
| $D_k$ | episode 内累计声音剂量 |
| $K_t^{space}$ | 空格键人工停止事件 |
| $U_t$ | 声学或干预预算安全约束触发 |
| $F_t$ | 数据、设备或运行时故障 |

## 2. 时间定义

系统使用三个时间尺度：

| 时间尺度 | 符号 | 初始值 | 用途 |
|---|---:|---:|---|
| 原始采样 | $1/f_m$ | 各通道不同 | 采集传感器数据 |
| 特征更新周期 | $\delta$ | 1 s | 更新融合特征和信号质量 |
| 策略决策周期 | $\Delta$ | 5 s | 选择一次响度动作并计算奖励 |

第 $k$ 个策略时刻为：

$$
t_k=t_0+k\Delta,\qquad k=0,1,2,\ldots
$$

每个决策周期内，固定声音模板最多播放一次：

$$
[t_k,t_k+d_{cue}) \text{ 播放},\qquad
[t_k+d_{cue},t_{k+1}) \text{ 无声观察}
$$

初始工程参数建议为：

$$
d_{cue}=0.5\,\text{s},\qquad d_{tail}=0.5\,\text{s},\qquad \Delta=5\,\text{s}
$$

由于播放声音会进入麦克风，以下区间的音频特征禁止用于恢复判断和奖励：

$$
\mathcal M_k=[t_k,t_k+d_{cue}+d_{tail}]
$$

实现时应优先使用播放参考信号进行声学回声消除；即使使用回声消除，仍保留 $\mathcal M_k$ 掩码。

## 3. 外部触发接口

第 $j$ 个外部预测事件定义为：

$$
e_j=(id_j,\tau_j,b_j,p_j,h_j,\theta_{on,j},v_j)
$$

其中：

- $id_j$：事件唯一标识；
- $\tau_j$：事件时间戳；
- $b_j\in\{0,1\}$：是否触发；
- $p_j\in[0,1]$：未来 OSA 风险分数，可选；
- $h_j>0$：预测时间范围，单位为秒，可选；
- $\theta_{on,j}\in[0,1]$：该版本模型使用的触发阈值，可选；
- $v_j$：外部模型版本。

事件在满足以下条件时被接受：

$$
G_j=
b_j
\cdot \mathbf 1[0\le t-\tau_j\le T_{fresh}]
\cdot \mathbf 1[id_j\notin\mathcal I_{seen}]
\cdot \mathbf 1[q_t\in\{\text{MONITORING}\}]
$$

初始设置：

$$
T_{fresh}=10\,\text{s}
$$

同一 $id_j$ 只允许启动一个 episode。处于干预、恢复观察、冷却或人工锁定状态时，不重复创建 episode，但新风险分数仍写入日志。

运行期间的连续风险 $p_t$ 取最新且未过期的风险输出。定义：

$$
m_t^p=\mathbf 1[t-\tau_{p,last}\le T_{risk\_age}],
\qquad T_{risk\_age}=10\,\text{s}
$$

当 $m_t^p=0$ 时，$p_t$ 不进入综合负担或恢复条件。禁止一直保持触发时的旧风险分数，否则旧的高风险可能永久阻止恢复。

预测时间范围在整个 episode 内使用触发事件的值：

$$
m_k^h=\mathbf 1[h_j\text{ 存在且有效}]
$$

当 $m_k^h=0$ 时，$\bar h_k$ 填充为 0，并依靠掩码区分“未知时间范围”和“时间范围为 0”。

外部模型同样必须接收播放掩码或回声消除后的 Audio。若风险分数使用了与 $\mathcal M_k$ 重叠且未去除回声的音频，则强制设置 $m_t^p=0$，避免干预声音抬高或降低自身的风险评分。

如果外部模型只提供布尔值，则令 $p_j$、$h_j$ 和 $\theta_{on,j}$ 的缺失掩码为 0，策略不能把缺失值当作低风险。

若提供连续风险分数，停止阈值必须与触发阈值形成滞回：

$$
0\le\theta_{off}<\theta_{on,j}\le1
$$

本文建议的 $\theta_{off}=0.30$ 只有在外部模型的 $\theta_{on,j}>0.30$ 时才有效，否则必须重新标定。

## 4. 多模态信号与特征

### 4.1 环形缓冲与对齐

对每个模态 $m\in\{A,P,I\}$ 建立独立环形缓冲：

$$
\mathcal B_t^m=\{(x_n^m,\tau_n^m)\mid t-W_m<\tau_n^m\le t\}
$$

初始窗口长度为：

$$
W_A=5\,\text{s},\qquad W_P=15\,\text{s},\qquad W_I=3\,\text{s}
$$

不同通道不在原始波形层面强制重采样。每个特征提取器输出：

$$
z_t^m=(f_t^m,\rho_t^m,d_t^m,m_t^m)
$$

其中 $f_t^m$ 为特征，$\rho_t^m\in[0,1]$ 为信号质量，$d_t^m=t-\tau_{last}^m$ 为数据年龄，$m_t^m\in\{0,1\}$ 为可用性掩码。

可用性定义为：

$$
m_t^m=\mathbf 1[\rho_t^m\ge \rho_{min}^m]\mathbf 1[d_t^m\le d_{max}^m]
$$

初始值建议：

$$
\rho_{min}^A=0.6,\quad \rho_{min}^P=0.6,\quad \rho_{min}^I=0.5,\quad
d_{max}^m=2\delta
$$

### 4.2 Audio 特征

对去直流并带通后的音频 $x_A[n]$，计算短时均方根：

$$
RMS_A=\sqrt{\frac{1}{N}\sum_{n=1}^{N}x_A[n]^2}
$$

定义鼾声频带能量比：

$$
E_{snore}=\frac{\sum_{f=30}^{500}P(f)}{\sum_{f=20}^{4000}P(f)+\epsilon}
$$

建议 Audio 提取器输出以下归一化量：

$$
f_t^A=[s_t,c_t,\eta_t,e_t]
$$

- $s_t\in[0,1]$：鼾声概率或鼾声时间占比；
- $c_t\in[0,1]$：呼吸音中断或异常静音概率；
- $\eta_t\in[0,1]$：呼吸周期规律性，1 表示规律；
- $e_t\in[0,1]$：归一化音频能量异常度。

Audio 异常度定义为：

$$
A_t=0.35s_t+0.35c_t+0.20(1-\eta_t)+0.10e_t
$$

如果音频窗口与任一 $\mathcal M_k$ 重叠且未完成可靠的回声消除，则设置：

$$
m_t^A=0
$$

### 4.3 PPG 特征

设有效脉搏峰时间为 $\pi_1,\ldots,\pi_N$，脉搏间期为：

$$
IBI_i=\pi_i-\pi_{i-1}
$$

心率、脉搏波幅和粗粒度脉率变异性定义为：

$$
HR_t=\frac{60}{\operatorname{median}(IBI_i)}
$$

$$
PPA_t=\operatorname{median}(x_{peak,i}-x_{trough,i})
$$

$$
RMSSD_t=\sqrt{\frac{1}{N-2}\sum_{i=2}^{N-1}(IBI_{i+1}-IBI_i)^2}
$$

25 Hz PPG 的峰值时间分辨率约为 40 ms，因此 $RMSSD_t$ 只作为弱特征，不能替代临床级 ECG HRV。

PPG 提取器输出：

$$
f_t^P=[HR_t,PPA_t,RMSSD_t,\dot{HR}_t,\dot{PPA}_t]
$$

其中趋势使用最近两个有效特征窗口计算：

$$
\dot x_t=\frac{x_t-x_{t-\Delta}}{\Delta}
$$

### 4.4 IMU 特征

对三轴加速度 $\mathbf a[n]=(a_x,a_y,a_z)$，低通估计重力方向 $\mathbf g_t$，并由其计算 roll 和 pitch。运动能量定义为：

$$
M_t=\sqrt{\frac{1}{N}\sum_{n=1}^{N}\|\mathbf a[n]-\mathbf g_t\|_2^2}
$$

IMU 提取器输出：

$$
f_t^I=[u_t,roll_t,pitch_t,M_t,\Delta roll_t]
$$

- $u_t\in[0,1]$：仰卧概率；
- $M_t$：运动能量；
- $|\Delta roll_t|$：翻身幅度。

IMU 用于识别体位改善和排除强运动/觉醒污染。仰卧本身不等于生理异常，因此不能仅因 $u_t$ 较高就判定“未恢复”。

## 5. 个体基线与归一化

在触发前、无声音播放且信号有效的稳定睡眠区间收集个体基线。默认基线长度：

$$
T_{base}=120\,\text{s}
$$

对特征 $x_i$ 使用中位数和 MAD 建立鲁棒基线：

$$
\mu_i=\operatorname{median}(x_i),\qquad
\sigma_i=1.4826\operatorname{median}(|x_i-\mu_i|)+\epsilon
$$

归一化特征为：

$$
z_{i,t}=\operatorname{clip}\left(\frac{x_{i,t}-\mu_i}{\sigma_i},-z_{max},z_{max}\right),
\qquad z_{max}=5
$$

episode 开始后冻结 $\mu_i,\sigma_i$，禁止在干预过程中更新基线，否则异常状态可能逐步被吸收到“正常”基线中。

若有效个体基线不足 $T_{base}$，按以下优先级回退：

1. 同一用户历史夜晚的个体基线；
2. 训练集人群基线；
3. 若两者均不可用，拒绝自动判定恢复，只允许人工停止或安全中止。

定义基线可用标志：

$$
b^{base}=\mathbf 1[\text{存在有效个体基线或人群基线}]
$$

定义分段异常函数：

$$
g(x;l,h)=\operatorname{clip}\left(\frac{x-l}{h-l},0,1\right)
$$

PPG 异常度定义为：

$$
P_t=\max\left(
g(|z_{HR,t}|;1,3),
g(|z_{PPA,t}|;1,3),
0.5g(|z_{RMSSD,t}|;1.5,4),
0.5g(|z_{\dot{HR},t}|;1.5,4),
0.5g(|z_{\dot{PPA},t}|;1.5,4)
\right)
$$

其中 RMSSD 项乘以 0.5，以降低低采样率带来的误差影响。

## 6. 风险/生理负担指标

定义候选分量及对应掩码：

$$
c_t=[p_t,A_t,P_t,u_t],\qquad
m_t=[m_t^p,m_t^A,m_t^P,m_t^I]
$$

初始权重为：

$$
w=[0.35,0.30,0.25,0.10]
$$

风险分数或某个模态缺失时，不能用 0 填充。使用掩码重新归一化：

$$
H_t=\frac{\sum_i w_i m_{i,t}c_{i,t}}{\sum_i w_i m_{i,t}+\epsilon}
$$

$H_t\in[0,1]$ 称为综合生理负担，只用于控制和奖励塑形，不作为医学诊断结果。

在有效、无声的策略窗口上定义改善量：

$$
\Delta H_k=\operatorname{clip}(H_{t_k}-H_{t_{k+1}},-1,1)
$$

若 $H_{t_k}$ 或 $H_{t_{k+1}}$ 无效，则该步的 $\Delta H_k$ 项置为 0，并记录 `reward_mask=0`，不能把缺失数据解释成改善。

## 7. 状态定义

### 7.1 生理状态标签

以下标签用于日志、奖励和评估，不重新承担外部模型的预测职责：

| 标签 | 数学条件 | 含义 |
|---|---|---|
| `UNKNOWN` | 任一关键质量条件不满足 | 无法可靠判断 |
| `AT_RISK` | 已接受外部触发且未恢复 | 已进入干预 episode |
| `RESPONDING` | $\Delta H_k\ge\theta_{resp}$ | 生理负担正在下降 |
| `RECOVERED_WINDOW` | $N_k=1$ | 单个观察窗恢复 |
| `RECOVERED` | 连续 $K$ 个 $N_k=1$ | 满足自动停止条件 |
| `AWAKE_CONFIRMED` | $K_t^{space}=1$ | 用户按空格键确认清醒 |

初始响应阈值：

$$
\theta_{resp}=0.05
$$

### 7.2 控制状态机

控制状态记为：

$$
q_t\in\{MONITORING,ARMED,INTERVENING,RECOVERY\_CHECK,COOLDOWN,
MANUAL\_LOCKOUT,SAFE\_STOP\}
$$

各状态定义如下：

- `MONITORING`：等待新的外部触发，动作强制为 0。
- `ARMED`：触发已接收，检查基线、数据质量和安全条件。
- `INTERVENING`：策略选择响度并播放固定声音模板。
- `RECOVERY_CHECK`：无声观察，计算恢复条件。
- `COOLDOWN`：生理恢复后抑制短期重复干预。
- `MANUAL_LOCKOUT`：用户按空格键后立即静音，必须显式 reset 才能退出。
- `SAFE_STOP`：超时、故障或安全越界导致的静音锁定。

状态转换优先级从高到低为：

$$
\text{人工停止}>\text{安全保护}>\text{生理恢复}>\text{普通状态转换}
$$

`ARMED` 状态的数据就绪条件只检查基线和通道可用性，不使用后文的恢复资格：

$$
C_t^{ready}=\mathbf 1[b^{base}=1]m_t^Am_t^Pm_t^I
$$

活动状态的保护条件为：

$$
C_t^{guard}=U_t\lor F_t\lor
\mathbf 1[t-t_0\ge T_{active,max}]
$$

转换规则为：

| 当前状态 | 条件 | 下一状态 |
|---|---|---|
| 任意非锁定状态 | $K_t^{space}=1$ | `MANUAL_LOCKOUT` |
| `ARMED`/`INTERVENING`/`RECOVERY_CHECK` | $C_t^{guard}=1$ | `SAFE_STOP` |
| `MONITORING` | $G_j=1$ | `ARMED` |
| `ARMED` | $C_t^{ready}=1\land C_t^{guard}=0$ | `INTERVENING` |
| `ARMED` | 校验超过 $T_{arm}$ 仍失败，设置 $F_t=1$ | `SAFE_STOP` |
| `INTERVENING` | 固定 cue 播放结束 | `RECOVERY_CHECK` |
| `RECOVERY_CHECK` | $S_t^{phys}=1$ | `COOLDOWN` |
| `RECOVERY_CHECK` | $S_t^{phys}=0\land C_t^{guard}=0$ 且到达下一决策点 | `INTERVENING` |
| `COOLDOWN` | $t-t_{recover}\ge T_{cool}$ | `MONITORING` |
| `MANUAL_LOCKOUT` | 显式人工 reset | `MONITORING` |
| `SAFE_STOP` | 故障/安全原因解除且显式人工 reset | `MONITORING` |

初始设置：

$$
T_{arm}=10\,\text{s},\qquad T_{cool}=60\,\text{s}
$$

`MANUAL_LOCKOUT` 不因新的预测触发自动退出，防止用户已经清醒时系统再次播放。

## 8. 策略观测空间

真实气道状态不可完全观测，因此本问题按部分可观测约束马尔可夫决策过程（POMDP/CMDP）建模。

角度归一化定义为：

$$
\overline{roll}_k=\operatorname{clip}(roll_k/\pi,-1,1),\qquad
\overline{pitch}_k=\operatorname{clip}(2pitch_k/\pi,-1,1)
$$

单步观测定义为：

$$
o_k=[
p_k,\bar h_k,
s_k,c_k,\eta_k,e_k,A_k,
z_{HR,k},z_{PPA,k},z_{RMSSD,k},z_{\dot{HR},k},z_{\dot{PPA},k},P_k,
u_k,\overline{roll}_k,\overline{pitch}_k,M_k,\Delta roll_k,
\rho_k^A,\rho_k^P,\rho_k^I,
m_k^A,m_k^P,m_k^I,m_k^p,m_k^h,
\hat a_{k-1},\bar T_k,\bar D_k,\bar N_k,
onehot(q_k)
]
$$

其中：

$$
\bar h_k=\operatorname{clip}(h_k/h_{max},0,1),
\qquad h_{max}=60\,\text{s}
$$

$$
\bar T_k=\frac{t_k-t_0}{T_{active,max}},\qquad
\bar D_k=\frac{D_k}{D_{max}},\qquad
\bar N_k=\frac{N_k^{cue}}{N_{max}}
$$

缺失的数值特征填充为训练集标准化后的 0，但必须同时提供相应掩码 $m_k$。

策略状态使用最近 $L=6$ 个决策观测，即 30 秒历史：

$$
s_k=(o_{k-L+1},\ldots,o_k)
$$

实现可使用特征堆叠、GRU 或 LSTM；不能只给单帧观测后假设其满足完全马尔可夫性。

## 9. 动作空间与声学映射

### 9.1 初始离散动作

第一版动作仅为归一化响度：

$$
a_k\in\mathcal A=\{0,0.2,0.4,0.6,0.8,1.0\}
$$

其中 $a_k=0$ 表示本周期静音观察，但不会单独结束 episode。自动结束仍只由连续生理恢复触发。

当 $a_k>0$ 时，请求声压级为：

$$
L_k^{request}=L_{min}+a_k(L_{max}-L_{min})
$$

其中：

- $L_{min}$：经设备标定后可用于干预的最低声压级 `[TBD-CAL]`；
- $L_{max}$：伦理、听力安全和设备限制共同确定的最高声压级 `[TBD-CAL]`。

安全层实际执行的归一化响度记为 $\hat a_k$。设备数字增益必须由标定函数得到：

$$
g_k=C_{device}^{-1}(L_k^{exec})
$$

不能直接把 $a_k$ 当作 PCM 幅度。

### 9.2 固定声音模板

声音模板参数定义为：

$$
\xi=(waveform_0,f_0,d_{cue},fade_{in},fade_{out},interval)
$$

$\xi$ 在一次实验中固定，不进入动作空间。这样策略学习到的差异只能归因于响度，而不是频率、时长或空间参数同时变化。

### 9.3 安全投影

策略动作必须经过独立安全层：

$$
\hat a_k=\Pi_{safe}(a_k;\hat a_{k-1},D_k,t_k)
$$

由声压上限得到当前最大可执行归一化响度：

$$
a_{cap}=\operatorname{clip}\left(
\frac{L_{cap}-L_{min}}{L_{max}-L_{min}},0,1
\right)
$$

然后限制响度级别和单步上升。静音和降低响度必须始终能够立即执行：

$$
\hat a_k=
\begin{cases}
0,&a_k=0\\
\min(a_k,a_{cap},\hat a_{k-1}+\Delta a_{max}),&a_k>0
\end{cases}
$$

其中 $L_{cap}=\min(L_{device},L_{protocol},L_{user})$，$\Delta a_{max}$ 由允许的最大声压变化换算，所有值均为 `[TBD-CAL]`。

当 $\hat a_k>0$ 时：

$$
L_k^{exec}=L_{min}+\hat a_k(L_{max}-L_{min})
$$

当 $\hat a_k=0$ 时，定义数字增益 $g_k=0$，不定义静音对应的 dB SPL 数值。

若执行该动作会违反累计剂量或播放次数约束，则：

$$
\hat a_k=0,\qquad g_k=0,\qquad U_t=1
$$

安全层不能被策略参数更新，也不能被关闭以换取更高奖励。

## 10. 生理恢复和停止条件

### 10.1 单窗口恢复

恢复只能在无声、回声尾音结束且关键信号有效的观察窗中判定。

由于外部模型预测的是“未来事件”，触发瞬间的生理指标可能尚未异常。为防止系统把“触发后一直正常”误记为干预成功，先定义恢复资格门控。

令 $\mathcal J_k$ 为 episode 内截至第 $k$ 步所有综合负担有效的窗口集合。最高有效负担为：

$$
\mathcal J_k=\{j\mid 0\le j\le k,\ H_j\text{ 有效}\},
\qquad
H_k^{peak}=\max_{j\in\mathcal J_k}H_j
$$

若 $\mathcal J_k=\varnothing$，则恢复资格 $E_k=0$。

恢复资格定义为：

$$
E_k=
\mathbf 1[t_k-t_0\ge T_{min\_active}]
\cdot
\mathbf 1\left[
\left(m_k^p=1\land p_k\le\theta_{off}\right)
\lor
\left(H_k^{peak}\ge\theta_{enter}\land H_k\le\theta_{normal}\right)
\right]
$$

初始参数：

$$
T_{min\_active}=10\,\text{s},\qquad
\theta_{enter}=0.45,\qquad
\theta_{normal}=0.25
$$

第一条路径适用于外部模型持续提供风险分数的情况；第二条路径适用于已观察到 Audio/PPG/IMU 异常后又回落的情况。如果只有一次布尔触发，而且生理指标从未异常，则没有足够证据把 episode 标为 `RECOVERED`，最终只能人工停止或保护性超时。这一限制避免把自然未发生的事件错误归因于干预。

有效窗口：

$$
V_k=
\mathbf 1[E_k=1]
\mathbf 1[b^{base}=1]
\mathbf 1[m_k^A=1]
\mathbf 1[m_k^P=1]
\mathbf 1[m_k^I=1]
$$

Audio 正常条件：

$$
N_k^A=\mathbf 1[A_k\le\theta_A],\qquad \theta_A=0.25
$$

PPG 正常条件：

$$
N_k^P=
\mathbf 1[|z_{HR,k}|\le2.0]
\mathbf 1[|z_{PPA,k}|\le2.5]
\mathbf 1[|z_{RMSSD,k}|\le4.0]
\mathbf 1[|z_{\dot{HR},k}|\le2.5]
\mathbf 1[|z_{\dot{PPA},k}|\le2.5]
$$

IMU 无明显觉醒运动条件：

$$
N_k^I=\mathbf 1[M_k\le\theta_M]
$$

$\theta_M$ 使用个体睡眠基线的高分位数确定：

$$
\theta_M=Q_{0.95}(M_{baseline})
$$

若外部模型提供有效风险分数，则风险回落条件为：

$$
N_k^R=\mathbf 1[p_k\le\theta_{off}],\qquad \theta_{off}=0.30
$$

若不提供风险分数，则 $N_k^R=1$，但仍必须满足 Audio、PPG 和 IMU 条件。

单窗口恢复定义为：

$$
N_k=V_kN_k^AN_k^PN_k^IN_k^R
$$

IMU 的体位变化作为恢复证据和奖励，但不是必要条件。即使仍为仰卧位，只要其他生理指标恢复且没有明显运动，也允许判定恢复。

### 10.2 连续恢复

为了抑制单窗口噪声，自动恢复必须连续成立：

$$
S_k^{phys}=\prod_{j=0}^{K-1}N_{k-j}
$$

有可靠个体基线时：

$$
K=3 \quad (15\,\text{s})
$$

仅有人群基线时采用更保守设置：

$$
K=5 \quad (25\,\text{s})
$$

任何无效窗口都会令连续计数归零，缺失数据不能触发“已恢复”。

### 10.3 人工停止

空格键事件定义为：

$$
K_t^{space}=\mathbf 1[key_t=SPACE]
$$

只要 $K_t^{space}=1$，立即执行：

$$
\hat a(t)=0,\qquad g(t)=0,\qquad q_{t^+}=MANUAL\_LOCKOUT
$$

从按键事件到音频输出静音的目标延迟为：

$$
T_{manual\_latency}\le100\,\text{ms}
$$

### 10.4 总停止逻辑

业务停止为：

$$
S_t^{business}=K_t^{space}\lor S_t^{phys}
$$

安全保护性中止为：

$$
S_t^{guard}=U_t\lor F_t\lor
\mathbf 1[t-t_0\ge T_{active,max}]
$$

最终静音命令为：

$$
S_t=S_t^{business}\lor S_t^{guard}
$$

其中 $S_t^{phys}$ 表示当前时刻最近一个完整策略窗口的 $S_k^{phys}$。终止原因按优先级唯一确定：

$$
reason_t=
\begin{cases}
AWAKE,&K_t^{space}=1\\
SAFETY,&K_t^{space}=0\land U_t=1\\
FAULT,&K_t^{space}=0\land U_t=0\land F_t=1\\
TIMEOUT,&K_t^{space}=0\land U_t=0\land F_t=0\land t-t_0\ge T_{active,max}\\
RECOVERED,&S_t^{phys}=1\land S_t^{guard}=0\\
NONE,&\text{其他}
\end{cases}
$$

只有 `RECOVERED` 计为自动干预成功；`AWAKE` 单独统计为人工停止/可能觉醒；其余均为截断或失败。

## 11. 奖励函数

### 11.1 事件变量

定义第 $k$ 个决策步的事件指标：

- $y_k^{rec}=1$：本步首次满足 $S_k^{phys}=1$；
- $y_k^{awake}=1$：本步检测到空格键；
- $y_k^{osa}=1$：本步首次出现真实 OSA 事件；
- $y_k^{turn}=1$：从仰卧转为非仰卧并连续保持两个窗口；
- $y_k^{proj}=1$：策略动作被安全层裁剪，但 episode 仍可继续；
- $y_k^{safety}=1$：本步因声压、剂量或次数约束进入 `SAFE_STOP`；
- $y_k^{timeout}=1$：本步达到最大 active 时间但仍未恢复；
- $y_k^{fault}=1$：本步因外部设备或传感器故障而截断。

其中 $y_k^{osa}$ 必须来自独立的在线事件检测器或离线人工/临床标注，不能直接把同一个预测模型的风险分数当作“真实 OSA 已发生”。如果没有该标签，应禁用此奖励项，并且实验结论只能表述为“降低风险/改善代理指标”，不能表述为“阻止 OSA”。

### 11.2 声音剂量

定义相对于参考声压 $L_{ref}$ 的单步剂量：

$$
\Delta D_k=
\begin{cases}
d_{cue}10^{(L_k^{exec}-L_{ref})/10},&\hat a_k>0\\
0,&\hat a_k=0
\end{cases}
$$

累计剂量为：

$$
D_k=\sum_{j=0}^{k}\Delta D_j
$$

供奖励使用的归一化单步剂量为：

$$
\bar d_k=\frac{\Delta D_k}{D_{max}}
$$

$L_{ref}$ 和 $D_{max}$ 由声学校准及安全协议确定 `[TBD-CAL]`。

空格键的信用分配定义为：若在第 $k_s$ 步按下空格键，则

$$
c_k^{awake}=
\begin{cases}
0.6,&k=k_s\\
0.3,&k=k_s-1\\
0.1,&k=k_s-2\\
0,&\text{其他}
\end{cases}
$$

因此一次人工停止的总惩罚权重仍为 1，但会显式归因到最近三个响度动作。

### 11.3 完整单步奖励

$$
\begin{aligned}
r_k={}&
w_H\Delta H_k
+B_{rec}y_k^{rec}
+B_{turn}y_k^{turn}\\
&-C_{osa}y_k^{osa}
-C_{awake}c_k^{awake}
-C_{safety}y_k^{safety}
-C_{timeout}y_k^{timeout}\\
&-\lambda_L\hat a_k^2
-\lambda_{slew}|\hat a_k-\hat a_{k-1}|
-\lambda_D\bar d_k\\
&-\lambda_t
-\lambda_{proj}y_k^{proj}
\end{aligned}
$$

初始权重建议如下，后续必须通过离线数据和敏感性分析调整：

| 参数 | 初始值 | 作用 |
|---|---:|---|
| $w_H$ | 2.0 | 奖励综合负担下降 |
| $B_{rec}$ | 8.0 | 成功恢复终止奖励 |
| $B_{turn}$ | 0.5 | 奖励稳定体位改善 |
| $C_{osa}$ | 10.0 | 惩罚真实 OSA 发生 |
| $C_{awake}$ | 8.0 | 惩罚人工确认觉醒 |
| $C_{safety}$ | 6.0 | 惩罚耗尽声音剂量或播放次数预算 |
| $C_{timeout}$ | 4.0 | 惩罚在 active 上限内未恢复 |
| $\lambda_L$ | 0.25 | 抑制不必要的高响度 |
| $\lambda_{slew}$ | 0.10 | 抑制响度突变 |
| $\lambda_D$ | 0.50 | 抑制累计声音剂量 |
| $\lambda_t$ | 0.05 | 缩短恢复时间 |
| $\lambda_{proj}$ | 1.0 | 惩罚请求非法动作 |

对未来回报使用：

$$
J(\pi)=\mathbb E_{\pi}\left[\sum_{k=0}^{K_T-1}\gamma^kr_k\right],
\qquad \gamma=0.98
$$

策略目标为：

$$
\pi^*=\arg\max_{\pi}J(\pi)
$$

并满足第 12 节的所有硬约束。

### 11.4 奖励归因规则

1. $\Delta H_k$ 只在无声且数据有效的窗口计算。
2. `AWAKE` 惩罚分配给按键前最近 $K_{credit}=3$ 个动作，避免只惩罚按键所在的静音步。
3. $y_k^{osa}$ 对每次真实事件只惩罚一次，不能按每个采样点重复累计。
4. `RECOVERED` 奖励只在首次满足连续恢复时发放一次。
5. 安全约束不能仅依赖负奖励；即使奖励权重配置错误，安全层也必须阻止危险动作。
6. 无效传感器窗口不提供正向改善奖励，以防策略通过制造信号缺失获得高回报。
7. $y_k^{fault}=1$ 不产生策略惩罚，因为设备故障通常不是动作导致的；该 transition 标记为外生截断并从策略效果统计中单独报告。
8. $y_k^{safety}$ 与 $y_k^{timeout}$ 只在进入截断状态的第一步取 1，避免重复惩罚。
9. $y_k^{proj}$ 与 $y_k^{safety}$ 互斥；如果动作导致 `SAFE_STOP`，只记录 $y_k^{safety}=1$，避免同一安全事件被重复惩罚。

## 12. 安全约束

策略属于约束决策过程，形式化为：

$$
\max_\pi J(\pi)\quad\text{s.t.}\quad C_i(\tau)\le d_i,\ \forall i
$$

必须硬编码以下约束：

### 12.1 声压约束

$$
L_{min}\le L_k^{exec}\le L_{cap}\qquad(a_k>0)
$$

### 12.2 响度上升率

$$
\hat a_k-\hat a_{k-1}\le\Delta a_{max}
$$

该约束在归一化响度域执行，从静音到首个非零动作也有明确定义；降低响度或立即静音不受该上升约束阻止。对应的声压变化必须通过设备标定验证不超过协议允许值。

### 12.3 累计剂量

$$
D_k\le D_{max}
$$

### 12.4 最大 active 时间

初始工程上限：

$$
T_{active,max}=60\,\text{s}
$$

该值属于保护性默认值，需根据实验协议调整。

### 12.5 最大播放次数

$$
N_k^{cue}\le N_{max},\qquad
N_{max}=\left\lfloor\frac{T_{active,max}}{\Delta}\right\rfloor=12
$$

### 12.6 最小无声间隔

$$
T_{quiet,min}=\Delta-d_{cue}=4.5\,\text{s}
$$

### 12.7 信号故障

关键模态连续无效时长为：

$$
T_{invalid}^m(t)=t-\max\{\tau\le t:m_\tau^m=1\}
$$

若某模态从 session 开始后从未有效，则约定 $T_{invalid}^m(t)=+\infty$。

若：

$$
\max(T_{invalid}^A,T_{invalid}^P,T_{invalid}^I)\ge T_{fault},
\qquad T_{fault}=15\,\text{s}
$$

则设置 $F_t=1$，进入 `SAFE_STOP`。不能在关键生理反馈缺失时继续自动增加响度。

### 12.8 数值和运行时故障

其中 $U_t=1$ 表示声学或干预预算约束已经触发，$F_t=1$ 表示数据、设备或运行时故障。以下任一条件令 $F_t=1$：

$$
\operatorname{isnan}(o_k)\lor
\operatorname{isinf}(o_k)\lor
\text{audio\_device\_error}\lor
\text{clock\_rollback}\lor
\text{watchdog\_timeout}
$$

## 13. Episode 定义

episode 在接受新触发时开始：

$$
t_0=\tau_j\quad\text{where}\quad G_j=1
$$

令 $y_k^{guard}=y_k^{safety}\lor y_k^{fault}\lor y_k^{timeout}$。按照人工停止高于保护中止、保护中止高于生理恢复的优先级，正常终止标志为：

$$
terminated_k=y_k^{awake}\lor
(y_k^{rec}\land \neg y_k^{guard})
$$

保护性截断标志：

$$
truncated_k=\neg y_k^{awake}\land y_k^{guard}
$$

真实 OSA 事件 $y_k^{osa}=1$ 默认只产生失败惩罚，不自动停止声音；系统继续闭环直到恢复、人工停止或安全截断。

训练时必须区分 `terminated` 和 `truncated`：

- `terminated=True`：达到业务终点，不继续 bootstrap；
- `truncated=True`：因保护条件截断，是否 bootstrap 由离线算法明确处理，不能混同为成功终点。

## 14. 初始策略与学习路线

### 14.1 数据收集基线策略

在强化学习之前使用确定性、受约束的递增策略：

$$
a_k^{base}=\min(a_0+k\Delta a,a_{base,max})
$$

建议初始归一化参数：

$$
a_0=0.2,\qquad\Delta a=0.2,\qquad a_{base,max}=0.8
$$

所有值仍由安全投影层约束。该策略用于验证状态机、收集响应数据和建立可比较的 baseline，不表示临床推荐响度。

### 14.2 第一版学习算法

由于动作离散、真实探索风险高，建议优先使用离线强化学习：

1. 使用受约束基线策略收集带完整时间戳的数据。
2. 按受试者划分训练、验证和测试集，禁止同一用户跨集合泄漏。
3. 采用 CQL、IQL 或带行为约束的离散 Q-learning。
4. 新策略先运行 `shadow mode`：计算动作但不实际播放，与基线动作比较。
5. 通过离线评估和受控试验后，才允许策略输出经过安全层执行。

如果未来改为连续动作 $a_k\in[0,1]$，可考虑带安全投影的 SAC；状态、奖励和停止条件保持不变。

不建议直接在真实睡眠用户上进行无约束在线探索。

## 15. 日志数据模型

每个决策步至少记录：

```text
session_id, subject_id, episode_id, event_id
timestamp, predictor_model_version, risk_score, prediction_horizon_s
raw_feature_refs, fused_observation, feature_quality, missing_masks
fsm_state_before, policy_action, policy_loudness_spl
executed_loudness_spl, digital_gain, safety_projection_reason
audio_mask_interval, cue_start, cue_end
physiological_burden_before, physiological_burden_after
recovery_window, recovery_streak
space_pressed, realized_osa_event, terminated, truncated, stop_reason
reward_total, reward_components, cumulative_dose
```

必须保存每个奖励分量，不能只保存总奖励，否则无法发现策略是否在利用错误代理目标。

## 16. 评估指标

### 16.1 自动恢复成功率

$$
SR_{recover}=\frac{N_{RECOVERED}}{N_{episodes}}
$$

### 16.2 人工停止/觉醒率

$$
AR_{awake}=\frac{N_{AWAKE}}{N_{episodes}}
$$

### 16.3 保护性中止率

$$
SR_{guard}=\frac{N_{SAFETY}+N_{FAULT}+N_{TIMEOUT}}{N_{episodes}}
$$

### 16.4 恢复时间

$$
TTR_i=t_{recover,i}-t_{trigger,i}
$$

报告中位数、四分位数和 95% 置信区间，不能只报告均值。

### 16.5 平均声音剂量

$$
\overline D=\frac{1}{N}\sum_{i=1}^{N}D_i
$$

### 16.6 条件 OSA 发生率

在存在独立真实事件标签时：

$$
IR_{OSA}=\frac{\sum_i\mathbf 1[episode_i\text{ 内发生 OSA}]}{N_{episodes}}
$$

比较策略必须在相同触发模型、相似风险分层和相同用户划分上进行。观察到干预后风险降低不等于证明干预阻止了 OSA；因果结论需要随机对照、交叉设计或其他有效的反事实设计。

### 16.7 响度效率

$$
E_L=\frac{N_{RECOVERED}}{\sum_i D_i+\epsilon}
$$

该指标只用于同一设备、同一标定和同一实验协议内比较。

## 17. 建议代码结构

```text
osa_system/
├── config.py                 # 采样率、时间窗、阈值和安全配置
├── types.py                  # TriggerEvent、Observation、StepResult 等类型
├── trigger_interface.py      # 外部 OSA 预测模型接口与事件去重
├── signal_processing.py      # Audio/PPG/IMU 缓冲、质量和特征提取
├── feature_fusion.py         # 时间戳对齐、基线归一化和 H_t
├── recovery_monitor.py       # N_k、连续恢复和终止原因
├── intervention_controller.py# 状态机和控制周期
├── loudness_policy.py        # 基线策略及学习策略推理
├── safety.py                 # 独立安全投影与剂量计算
├── audio_synthesis.py        # 固定声音模板和标定增益
├── keyboard_monitor.py       # 非阻塞空格键监听
├── intervention_env.py       # POMDP/CMDP、奖励和 episode 接口
├── event_logger.py           # 逐步日志及奖励分解
├── train_offline.py          # 离线训练
├── evaluate_policy.py        # 按受试者评估
└── main.py                   # 实时运行、回放、shadow 三种模式
```

现有四状态分类器、TrendEncoder、UCDDB 四分类训练和 RIP/SpO2 特征链路不再进入实时运行路径，可迁移到 `legacy/` 以保留历史实验。

## 18. 配置基线

以下配置用于软件开发和仿真，不代表临床最终参数：

```yaml
sampling_rate:
  audio_hz: 8000
  ppg_hz: 25
  imu_hz: 33

trigger:
  freshness_s: 10.0
  risk_max_age_s: 10.0
  horizon_max_s: 60.0
  arm_timeout_s: 10.0

timing:
  feature_tick_s: 1.0
  decision_interval_s: 5.0
  cue_duration_s: 0.5
  echo_tail_mask_s: 0.5
  active_timeout_s: 60.0
  cooldown_s: 60.0
  invalid_signal_timeout_s: 15.0

feature_window:
  audio_s: 5.0
  ppg_s: 15.0
  imu_s: 3.0
  personal_baseline_s: 120.0
  audio_quality_min: 0.60
  ppg_quality_min: 0.60
  imu_quality_min: 0.50
  feature_max_age_s: 2.0

recovery:
  minimum_active_s: 10.0
  burden_enter: 0.45
  burden_normal: 0.25
  audio_abnormality_max: 0.25
  predictor_risk_off: 0.30
  hr_z_abs_max: 2.0
  ppg_amplitude_z_abs_max: 2.5
  rmssd_z_abs_max: 4.0
  hr_slope_z_abs_max: 2.5
  ppg_amplitude_slope_z_abs_max: 2.5
  imu_motion_baseline_quantile: 0.95
  response_delta_burden_min: 0.05
  consecutive_windows_personal_baseline: 3
  consecutive_windows_population_baseline: 5

action:
  levels: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  min_spl: TBD-CAL
  max_spl: TBD-CAL
  max_action_step: TBD-CAL
  dose_reference_spl: TBD-CAL
  dose_budget: TBD-CAL

baseline_policy:
  initial_action: 0.2
  action_increment: 0.2
  max_action: 0.8

reward:
  burden_improvement: 2.0
  recovered: 8.0
  stable_turn: 0.5
  realized_osa: -10.0
  awake_space: -8.0
  safety_stop: -6.0
  active_timeout: -4.0
  loudness_squared: -0.25
  loudness_change: -0.10
  acoustic_dose: -0.50
  active_step: -0.05
  safety_projection: -1.0
  awake_credit: [0.6, 0.3, 0.1]
  gamma: 0.98
```

## 19. 验收条件

### 19.1 状态机测试

1. `MONITORING` 中只接受未处理且未过期的 trigger。
2. 任意活动状态收到空格键后，在 100 ms 目标内静音并进入 `MANUAL_LOCKOUT`。
3. 单个正常窗口不能自动停止；必须连续满足 $K$ 个窗口。
4. 触发时指标已经正常且没有连续风险分数时，不能仅因保持正常而获得 `RECOVERED`。
5. 任一无效关键模态令恢复连续计数清零。
6. `MANUAL_LOCKOUT` 不能被新 trigger 自动解除。
7. 安全条件与恢复条件同一时刻出现时，按优先级记录为 `SAFETY`，人工停止仍具有最高优先级。

### 19.2 奖励测试

1. 相同恢复结果下，更低响度和更低剂量获得更高回报。
2. 相同响度下，更快恢复获得更高回报。
3. 空格键和真实 OSA 事件只在规定信用窗口内产生一次事件惩罚。
4. 传感器缺失不能产生正向 $\Delta H$ 奖励。
5. 安全层裁剪动作时，日志同时保存策略动作和实际动作。

### 19.3 数据接口测试

1. 1 秒 Audio 数据严格对应 8000 个标称样本，PPG 对应 25 个，IMU 对应约 33 个；实际对齐使用时间戳并允许合理时钟抖动。
2. 不同模态丢包时，特征值、质量、年龄和掩码保持一致。
3. 播放及尾音掩码内的 Audio 不参与恢复和正向奖励。
4. 重复 event ID 不创建重复 episode。
5. 所有终止都包含唯一、可审计的 `stop_reason`。

## 20. 实施前必须解决的问题

以下项目未解决前，只能进行软件仿真或离线回放：

1. 确定外部模型是否能提供连续风险分数、预测时间范围和真实事件结果标签。
2. 确认 PPG 是单路波形还是包含红光/红外及 SpO2 输出。
3. 完成目标耳机的数字增益到 dB SPL 标定。
4. 由临床和听力安全负责人确定 $L_{min},L_{max},\Delta a_{max},D_{max}$。
5. 验证播放参考信号、回声消除和 Audio 掩码是否能阻止自激触发。
6. 明确空格键后的恢复方式：建议仅允许人工 reset 或开始新 session。
7. 定义真实 OSA 发生标签来源；否则不得使用 $y_k^{osa}$ 奖励或声称“预防成功率”。
