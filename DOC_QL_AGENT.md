# Q-Learning Agent 实现详解

> 供报告撰写（Jingyu Liang）和答辩参考。本文档详细说明 RL Agent 的设计动机、状态空间、动作空间、训练流程和核心公式。

---

## 1. 为什么选择 Q-Learning

德州扑克是一个**部分可观测马尔可夫决策过程（POMDP）**：玩家看不到对手手牌，不知道未来的公共牌。Q-Learning 是最经典的 model-free RL 算法，适合本项目的原因：

1. **状态空间可控**：通过桶化（bucketing）将连续的状态离散化为有限的离散状态，使得 Q-table 可以存入内存
2. **无需环境模型**：Q-Learning 是 off-policy 的 model-free 方法，不需要知道转移概率
3. **可解释性强**：Q-table 可直接导出策略，便于分析 Agent 学到了什么
4. **基线价值**：作为 RL 基线，与 Expectimax（搜索式）和 GTO（博弈论式）形成方法论对比

---

## 2. 状态空间设计

### 2.1 原始观察空间

PokerEnv 的 `observe()` 返回 18 个字段的字典（手牌、公共牌、底池、阶段、筹码等），其中包含连续值（底池大小）和高维信息（牌面组合），无法直接作为 Q-table 的 key。

### 2.2 状态编码方案

我们将原始观察编码为 4 维离散元组：

```
state = (hand_strength, board_strength, pot_bucket, stage)
```

| 维度 | 编码方式 | 桶数 | 说明 |
|------|----------|------|------|
| `hand_strength` | Monte Carlo 胜率 → 11 桶 | 11 | `int(win_rate * 11)`，范围 0~10 |
| `board_strength` | treys 牌型等级 → 5 桶 | 5 | 皇家同花顺=4，……，高牌=0 |
| `pot_bucket` | 底池金额 → 5 桶 | 5 | 边界 `[0, 50, 150, 400, 800]` |
| `stage` | 游戏阶段枚举 | 4 | PREFLOP=0, FLOP=1, TURN=2, RIVER=3 |

**总状态数：11 × 5 × 5 × 4 = 1,100**

### 2.3 各维度合理性分析

**hand_strength（手牌强度桶）**

- 方法：从剩余牌中随机采样对手手牌 + 未发出的公共牌，模拟 N 次摊牌，计算我方胜率
- 11 桶的含义：0 = 胜率 [0%, 9.1%)，1 = [9.1%, 18.2%)，……，10 = [90.9%, 100%]
- 这是对 POMDP 隐藏信息（对手手牌）的贝叶斯近似

**board_strength（公共牌强度桶）**

- 方法：用 treys 库计算当前手牌+公共牌的牌型等级（rank_class）
- 牌型映射：皇家同花顺/同花顺/四条 → 4，满堂红/同花/顺子 → 3，三条/两对/一对 → 2，高牌 → 0~1
- 补充手牌胜率无法捕获的信息：同样是 50% 胜率，持顺子和持两对的策略可能不同

**pot_bucket（底池桶）**

- 底池大小直接影响底池赔率（pot odds），是决策的关键因素
- 5 桶边界对应不同的筹码深度：小底池（<50）适合诈唬，大底池（>400）需要更强手牌

**stage（游戏阶段）**

- 不同阶段策略截然不同：Preflop 只看手牌强度，River 需要精确评估

### 2.4 训练 vs 推理的 MC 采样次数

- **训练时：100 次** — 加速训练，约 ~80 it/s
- **推理时：1000 次** — 提高精度，减少策略方差

---

## 3. 动作空间

Agent 可选动作由 `RuleEngine.get_legal_actions()` 动态给出，是以下 4 个动作的子集：

| 动作 | IntEnum | 说明 |
|------|---------|------|
| `FOLD` | 0 | 弃牌，放弃底池 |
| `CALL` | 1 | 跟注，匹配对手下注额 |
| `RAISE_HALF_POT` | 2 | 加注 0.5×底池（Pot-Limit） |
| `RAISE_POT` | 3 | 加注 1×底池（Pot-Limit） |

动作空间大小 = 4（但实际合法动作通常为 2~4 个）。

---

## 4. Q-Learning 算法

### 4.1 Q-Table

```python
Q: Dict[StateTuple, Dict[Action, float]]
```

- Key：`(hand_strength, board_strength, pot_bucket, stage)` — 4 维整数元组
- Value：`{FOLD: q0, CALL: q1, RAISE_HALF: q2, RAISE_POT: q3}` — 各动作的 Q 值
- 实现：`defaultdict(lambda: defaultdict(float))` — 未访问的状态-动作对 Q 值初始化为 0

### 4.2 策略：ε-greedy

```
π(a|s) = {
    均匀随机选择 legal_actions 中的一个，  概率 ε
    argmax_{a ∈ legal} Q(s, a)，           概率 1-ε
}
```

- **训练初期** ε ≈ 1.0，几乎纯随机探索
- **训练后期** ε 衰减至 0.05，95% 利用已学策略
- 衰减公式：`ε ← ε × 0.9995`（每局衰减一次）

ε-greedy 的作用是**探索-利用平衡**：
- 太早收敛 ε → 0 会导致策略陷入局部最优
- 保持 ε = 0.05 的底线探索率防止策略僵化

### 4.3 Q 值更新规则

每次 Agent 执行动作后，观察到转移 `(s, a, r, s', done)`：

```
Q(s, a) ← Q(s, a) + α × [target - Q(s, a)]
```

其中 target 的计算：

```
if done:
    target = r                          # 终局：直接用最终收益
else:
    target = r + γ × max_{a'} Q(s', a') # 非终局：当前收益 + 折扣未来最大Q值
```

**超参数：**

| 参数 | 符号 | 值 | 含义 |
|------|------|----|------|
| 学习率 | α | 0.1 | 每次更新对旧 Q 值的覆盖比例 |
| 折扣因子 | γ | 0.95 | 未来收益的衰减程度 |
| 初始 ε | ε₀ | 1.0 | 初始探索率 |
| 最小 ε | ε_min | 0.05 | 最小探索率 |
| ε 衰减 | - | 0.9995 | 每局乘以的衰减因子 |

**Reward 设计：**

- 非终局步：`r = 0`（延迟奖励，只在游戏结束时给反馈）
- 终局步：`r = payoff / 1000`（归一化到 [-1, 1] 范围，payoff 是筹码净盈亏）

这种**稀疏奖励**设计意味着 Agent 需要通过多局游戏的 Q 值传播（bootstrap），从终局奖励逐步回传到早期决策。例如：Preflop 的好决策（拿好牌加注）要等到 River 摊牌赢钱才能学到，中间经过了 γ 的多次折扣。

---

## 5. 训练流程

### 5.1 混合训练策略

```
每局开始时：
    以 70% 概率选择 self-play（Agent vs 自己）
    以 30% 概率选择 vs RandomAgent
```

**为什么混合？**

- 纯 self-play：Agent 可能陷入自我强化循环，策略崩塌（如双方都只 FOLD）
- 纯 vs Random：学到的策略只针对随机对手，泛化性差
- 混合训练：self-play 提升策略上限，vs Random 保证基本对抗能力

### 5.2 单局训练流程

```
1. env.reset() → 初始观察 obs
2. agent.reset() → 清除内部状态
3. while not done:
       if 当前玩家 == 0 (Agent):
           action = agent.act(obs, legal_actions)  # ε-greedy
           记录 pending = (obs, action)
       else (对手):
           action = opponent.act(obs, legal_actions)
       obs, reward, done, info = env.step(action)
       if 当前玩家 == 0 且有 pending:
           prev_obs, prev_action = pending.pop()
           terminal_reward = info['payoffs'][0]/1000 if done else 0
           agent.observe_transition(prev_obs, prev_action, terminal_reward, obs, done)
4. agent.decay_epsilon()  # ε 衰减
```

### 5.3 训练输出示例

```
Training: 100%|████| 2000/2000 [00:46<00:00]
  eps=0.368  avg_r=-1.7  win=0.26  Q_size=297
Training complete. Q-table saved to q_table.pkl (297 states)
```

- `eps`：当前 ε 值（0.368 表示还在探索-利用过渡期）
- `avg_r`：最近 1000 局平均收益
- `win`：累计胜率
- `Q_size`：Q-table 中已访问的不同状态数（最多 1100）

---

## 6. 实验结果

### 6.1 训练后 vs RandomAgent

| 指标 | 值 |
|------|-----|
| 胜率 | 83-88% |
| 平均收益 | +64 ~ +118 筹码/局 |
| 动作熵 | 1.29 bit（最大 ~2 bit） |
| 动作分布 | FOLD: 3%, CALL: 59%, RAISE_HALF: 30%, RAISE_POT: 8% |

### 6.2 策略分析

- Agent 学会了**手牌强时加注**（RAISE_HALF 占 30%）
- 学会了**手牌弱时弃牌**（FOLD 占 3%，即使随机 Agent 根本不会主动 FOLD 以外弃牌）
- CALL 是最常见动作（59%），因为大部分手牌处于中间强度
- RAISE_POT 较少（8%），说明 Agent 只在最强手牌时全底池加注
- 动作熵 1.29 表明策略有多样性，不是简单的"永远 CALL"

---

## 7. Q-Learning 的局限性

1. **状态离散化丢失信息**：桶化掩盖了同一桶内的差异（如胜率 0.49 和 0.50 被分到不同桶）
2. **稀疏奖励**：只有终局给 reward，中间决策的 Q 值传播慢，需要大量训练局数
3. **无对手建模**：Q-Learning 不区分对手类型，对强对手和弱对手用同一策略
4. **非平稳环境**：self-play 时对手策略也在变，Q-table 可能过时
5. **无法处理连续动作**：加注额被固定为两个档位，无法学习最优加注尺度

这些局限恰好是 Expectimax Agent（对手建模）和 GTO Agent（博弈论均衡）的优势所在，也是本项目的核心对比维度。

---

## 8. 代码结构图

```
main.py train
    │
    ▼
Trainer.__init__()
    ├── PokerEnv()          # 扑克环境
    ├── QLAgent(training=True)  # Q-Learning 智能体
    │       └── StateEncoder(training=True)  # 100次MC采样
    └── RandomAgent()       # 基线对手

Trainer.train()
    │
    ├── 选择对手 (70% self-play, 30% random)
    ├── _play_one_game()
    │       ├── agent.act(obs, legal)          # ε-greedy 选动作
    │       │       └── StateEncoder.encode()  # obs → (hs, bs, pb, stage)
    │       │       └── argmax Q[s][a] over legal actions
    │       ├── env.step(action)              # 执行动作，推进环境
    │       └── agent.observe_transition()     # Q 值更新
    │               └── Q[s][a] += α × (target - Q[s][a])
    └── agent.decay_epsilon()                  # ε ← ε × 0.9995
```

---

## 9. 关键公式汇总（供报告使用）

**Q-Learning 更新：**
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

**ε-greedy 策略：**
$$\pi(a|s) = \begin{cases} \text{random} & \text{with prob } \epsilon \\ \arg\max_a Q(s,a) & \text{with prob } 1-\epsilon \end{cases}$$

**ε 衰减：**
$$\epsilon_{t+1} = \max(\epsilon_{\min},\ \epsilon_t \times \lambda)$$

**状态编码：**
$$s = \Big( \lfloor \text{win\_rate} \times 11 \rfloor,\ \text{board\_rank\_class},\ \text{pot\_bucket},\ \text{stage} \Big)$$

**Monte Carlo 胜率估计：**
$$\text{win\_rate} = \frac{\text{wins} + 0.5 \times \text{ties}}{N}$$
