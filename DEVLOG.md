# Zaichang Yu — 开发日志

## 2026-06-30 完成项

### 1. Poker Engine 关键 Bug 修复

**问题：** `engine/poker_env.py` 的 `step()` 方法在 `RuleEngine.apply_action()` 推进游戏阶段时，不会自动发出公共牌。`deal_community_cards()` 方法存在但从未被调用，导致整局游戏 0 张公共牌，摊牌直接比手牌而非正常 3+1+1 公共牌流程。

**修复：** 在 `step()` 中检测阶段变化，自动调用 `deal_community_cards()`：
- 记录 `old_stage`，执行 `apply_action()` 后比较 `new_stage`
- 若阶段变化且游戏未结束，调用 `self.deal_community_cards()`
- 验证：正常 Preflop→Flop(3张)→Turn(4张)→River(5张)

### 2. 新增文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `agents/random_agent.py` | 随机基线 Agent，从 legal_actions 均匀随机选择 | 14 |
| `agents/ql_agent.py` | Q-learning Agent：Q-table、ε-greedy、observe_transition、save/load | 100 |
| `training/__init__.py` | training 包初始化 | 3 |
| `training/state_encoder.py` | 状态编码器：hand_strength(11)×board_strength(5)×pot_bucket(5)×stage(4)=1100 状态 | 28 |
| `training/trainer.py` | 训练主循环：70% self-play + 30% random 混合训练，ε 衰减，统计输出 | 131 |
| `evaluation/__init__.py` | evaluation 包初始化 | 3 |
| `evaluation/evaluator.py` | 对战评估：胜率、平均收益、动作熵、循环赛 | 93 |
| `main.py` | CLI 入口：train / eval / tournament 三个子命令 | 101 |
| `.gitignore` | 忽略 __pycache__、*.pyc、*.pkl | 3 |

### 3. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `engine/poker_env.py` | `step()` 新增阶段变化检测 + 自动发公共牌（+5 行） |
| `agents/__init__.py` | 导出 RandomAgent、QLAgent |
| `README.md` | 更新项目结构树，标记已实现模块 |

### 4. 接口设计决策

- **状态编码**：`(hand_strength_bucket_11, board_strength_bucket_5, pot_bucket_5, stage_4)` → 总 1,100 状态
- **训练时 MC 采样**：100 次（快速），推理时 1000 次（精确）
- **混合训练**：70% self-play + 30% vs RandomAgent
- **Q-learning 超参**：α=0.1, γ=0.95, ε: 1.0→0.05, decay=0.9995
- **Reward**：终局 payoff/1000 归一化，中间步 reward=0

### 5. 验证结果

- 完整游戏流程：Preflop→Flop(3)→Turn(4)→River(5)→Showdown ✓
- 2000 局训练后 vs Random 胜率：83-88% ✓
- 动作分布合理：FOLD/CALL/RAISE_HALF/RAISE_POT 均有使用 ✓
- 动作熵 1.29（最大~2.0），策略有一定多样性 ✓
- `main.py train/eval/tournament` 三个命令全部端到端可用 ✓

### 6. 待其他成员对接的接口

- `agents/base_agent.py` 的 `act(observation, legal_actions)` — 所有 Agent 统一接口
- `evaluation/evaluator.py` 的 `run_match()` / `run_tournament()` — Mohan Li 可直接使用
- `agents/ql_agent.py` 的 QLAgent 已实现 `observe_transition()` — 可作为训练框架范例
- Linpei Duan 的 ExpectimaxAgent / GTOAgent 只需继承 BaseAgent 并实现 `act()`
