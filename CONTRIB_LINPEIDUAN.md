# Linpei Duan (linpeiduan) — 实现总结

## 完成模块

### 1. GTOAgent (`agents/gto_agent.py`)
基于博弈论纳什均衡近似的德州扑克智能体。

**方法：** 对每个合法动作计算期望收益 (EV)，结合 ε-greedy 混合策略选动作。

```
EV(FOLD)  = 0
EV(CALL)  = win_rate × pot − (1−win_rate) × call_amount
EV(RAISE) = win_rate × (pot+raise) − (1−win_rate) × (call+raise)
```

**ε-greedy：** 以 1−ε 概率选 EV 最大动作，ε 概率均匀随机，防止被 exploit。

**结果：** vs RandomAgent 胜率 81.5%，平均收益 +119 chips/局，动作熵 1.69。

---

### 2. OpponentModel (`agents/opponent_model.py`)
对手行为追踪与建模，供 GTOAgent 和 ExpectimaxAgent 集成。

**功能：**
- 按阶段统计对手动作频率（PREFLOP / FLOP / TURN / RIVER）
- 计算风格指标：进攻率 (aggression_rate)、弃牌率 (fold_rate)、主动入池率 (VPIP)
- 基于对手行为修正手牌胜率估计（贝叶斯直觉）
- 推断对手玩家类型标签（TAG / LAG / Nit / Calling Station）
- 小样本时自动降低调整幅度，使用 Laplace 平滑避免过拟合

**选手牌调整逻辑：**
- 对手加注 → 暗示强牌 → 下调我方 win_rate
- 紧对手跟注 → 真实牌力信号 → 小幅下调
- 松对手未弃牌 → 可能 bluff → 小幅上调
- 分阶段推理（如 River 常弃牌的对手在早期街更可能 bluff）

---

### 3. ExpectimaxAgent (`agents/expectimax_agent.py`)
基于搜索树的前向规划智能体。

**搜索树结构：**
- **决策节点 (max)**：我方选动作，遍历所有合法动作
- **机会节点 (chance)**：对手响应，用 GTO EV + softmax 建模对手动作概率
- **机会节点 (chance)**：发公共牌，Monte Carlo 采样剩余牌
- **叶子评估**：`win_rate × pot − cost`，用 MC 胜率 + 底池大小启发式评估

**对手建模：** 每次搜索中采样对手手牌，从对手视角计算每个合法动作的 GTO EV，用 softmax 温度参数转为概率分布。

**深度控制：** `depth_limit` 表示我方剩余决策次数（默认 3）。深度受限于计算资源——每层需 30~100 次对手手牌采样。

**结果：** depth=1, samples=5 时 vs RandomAgent 胜率 95.0%，平均收益 +189 chips/局。即使在浅搜索下也超越 QL 的 88% 和 GTO 的 81%。

---

## 性能一览

| Agent | 胜率 vs Random | 速度 | 方法 |
|-------|:---:|------|------|
| Random | 50% | 极快 | 均匀随机基线 |
| **GTOAgent** | **81.5%** | 快 (~20局/s) | EV 公式 + ε-greedy |
| **ExpectimaxAgent** | **95.0%** | 慢 (~3s/局) | 搜索树 + MC 采样 |
| QLAgent (baseline) | 88.5% | 快 | Q-Learning 训练 |

---

## 文件修改清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `agents/gto_agent.py` | 新增 | GTO 启发式 Agent (104行) |
| `agents/opponent_model.py` | 新增 | 对手行为建模 (240行) |
| `agents/expectimax_agent.py` | 新增 | Expectimax 搜索 Agent (498行) |
| `agents/__init__.py` | 修改 | 导出三个新 Agent |
