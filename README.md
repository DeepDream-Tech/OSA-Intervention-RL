# OSA Fuzzy-V2 Intervention

这是 `DeepDream-Tech/OSA-Intervention-RL` 的 `FUZZY-V1` 实验分支对应的本地最新版代码，
用于基于呼吸指标和历史干预状态的模糊控制策略验证。

## 运行

```bash
python -m pip install -r requirements.txt
python controller.py --help
```

`data_reader.py` 负责读取实验数据，`controller.py` 执行策略计算，`plot_session_overview.py`
用于生成会话级可视化。原始 CHE 数据、运行产物和本地虚拟环境不纳入仓库，请通过内部存储交接。

这是研究原型，不是医疗器械，不可用于诊断或治疗。
