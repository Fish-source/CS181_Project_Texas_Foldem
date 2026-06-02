# Texas Foldem — 简化德州扑克智能体比较研究

基于规则方法、GTO启发策略与强化学习的简化德州扑克智能体比较研究。

## 项目结构

```
Texas_Foldem/
├── config.py                      # 全局常量与配置
├── requirements.txt               # Python依赖
├── engine/                        # 扑克引擎核心
│   ├── __init__.py
│   ├── card.py                    # Card、Deck（52张牌，洗牌，发牌）
│   ├── game_state.py              # Stage/Action枚举、PlayerState/GameState
│   ├── hand_evaluator.py          # treys封装 + Monte Carlo胜率 + 桶化
│   ├── rule_engine.py             # 合法动作、动作执行、阶段推进、摊牌结算
│   └── poker_env.py               # Gym-like环境（reset/step/observe）
├── agents/                        # 智能体
│   ├── __init__.py
│   ├── base_agent.py              # 抽象基类 [已实现]
│   ├── expectimax_agent.py        # Expectimax智能体 [TODO: Linpei Duan]
│   ├── gto_agent.py               # GTO启发式智能体 [TODO: Linpei Duan]
│   └── opponent_model.py          # 对手建模 [TODO: Linpei Duan]
├── training/                      # 训练框架 [TODO]
│   ├── state_encoder.py           # 状态编码器
│   └── trainer.py                 # 训练主循环
├── evaluation/                    # 评估系统 [TODO]
│   ├── evaluator.py               # 胜率/平均收益/动作熵
│   └── visualize.py               # 图表绘制 [TODO: Mohan Li]
└── main.py                        # 入口脚本 [TODO]
```

## 游戏规则设定

| 设定 | 值 |
|------|----|
| 玩家数 | 2人 Heads-up |
| 初始筹码 | 1000 |
| 盲注 | SB=10, BB=20 |
| 游戏阶段 | Preflop → Flop → Turn → River（4阶段） |
| 动作空间 | FOLD, CALL, RAISE_HALF_POT, RAISE_POT |
| 下注结构 | Pot-Limit，每条街最多3次加注 |
| 手牌评估 | treys库 |

## 快速开始

```bash
pip install -r requirements.txt
```

```python
from engine import PokerEnv, HandEvaluator, RuleEngine
from config import Action, Stage

env = PokerEnv()
evaluator = HandEvaluator()
rule_engine = RuleEngine()

# 开始新局
obs = env.reset()

# 查看观察信息
print(obs['hand'])              # [Card, Card]
print(obs['community_cards'])   # [Card, ...]
print(obs['pot'])               # int
print(obs['stage'])             # Stage.PREFLOP/FLOP/TURN/RIVER
print(obs['legal_actions'])     # [Action.FOLD, Action.CALL, ...]

# 计算手牌胜率
win_rate = evaluator.monte_carlo_win_rate(obs['hand'], obs['community_cards'])
bucket = evaluator.hand_strength_bucket(win_rate, n_buckets=3)  # 0=弱 1=中 2=强

# 执行动作
obs, reward, done, info = env.step(Action.CALL)

# 游戏结束时
if done:
    print(info['winner'])       # 获胜玩家索引 (0 or 1)
    print(info['payoffs'])      # [p0_payoff, p1_payoff]
```

---

## 接口文档

### 1. Card & Deck (`engine/card.py`)

```python
from engine.card import Card, Deck, Suit, Rank

# Card 不可变对象
card = Card(Suit.HEARTS, Rank.ACE)   # Ah
card.suit                             # Suit.HEARTS
card.rank                             # Rank.ACE
card.treys_card                       # treys内部整数表示

# Deck: 52张牌洗牌发牌
deck = Deck()
deck.reset()                  # 重置并洗牌
hand = deck.deal(2)           # 发2张牌 -> [Card, Card]
flop = deck.deal(3)           # 发3张公共牌
deck.remaining                # 剩余牌数
```

### 2. GameState (`engine/game_state.py`)

```python
from engine.game_state import PlayerState, GameState

@dataclass PlayerState:
    hand: List[Card]          # 手牌（2张）
    chips: int                # 剩余筹码（初始1000）
    current_bet: int          # 当前街下注额
    total_bet: int            # 本局总下注额
    is_folded: bool           # 是否弃牌
    is_all_in: bool           # 是否全下
    is_active: bool           # property: 未弃牌且未全下

@dataclass GameState:
    players: List[PlayerState]
    community_cards: List[Card]
    pot: int
    stage: Stage              # PREFLOP/FLOP/TURN/RIVER
    current_player: int       # 当前行动玩家 (0 or 1)
    raises_this_street: int   # 当前街加注次数
    dealer_idx: int           # 庄家位置
    last_raiser: int          # 最后加注者
    acted_this_street: List[bool]
    betting_history: List[Tuple[int, Action, int]]  # (player, action, amount)
    is_terminal: bool
    winner: Optional[int]
```

### 3. HandEvaluator (`engine/hand_evaluator.py`)

```python
from engine.hand_evaluator import HandEvaluator

evaluator = HandEvaluator()

# 手牌排名（treys，越小越好）
rank = evaluator.evaluate(hand=[Card, Card], board=[Card, ...])

# Monte Carlo胜率 [0, 1]
win_rate = evaluator.monte_carlo_win_rate(hand, board, num_simulations=1000)

# 桶化
# n_buckets=3: 0=弱, 1=中, 2=强 （用于GTO/Expectimax简单评估）
# n_buckets=11: 0..10 （用于RL状态编码）
bucket = evaluator.hand_strength_bucket(win_rate, n_buckets=3)

# 公共牌强度桶 (0..4)
board_bucket = evaluator.board_strength_bucket(hand, board)
```

### 4. RuleEngine (`engine/rule_engine.py`)

```python
from engine.rule_engine import RuleEngine

engine = RuleEngine()

# 获取合法动作
legal = engine.get_legal_actions(state)   # -> List[Action]

# 执行动作（返回新状态，不修改原状态）
new_state = engine.apply_action(state, Action.RAISE_POT)

# 计算终局收益
payoffs = engine.compute_payoffs(state)   # -> [p0_payoff, p1_payoff]

# 底池桶化
bucket = engine.pot_bucket(pot_size)      # -> 0..4

# 底池赔率
odds = engine.pot_odds(call_amount, pot)  # -> float
```

**Pot桶化边界：** `[0, 50, 150, 400, 800]` → 5档

**加注额计算（Pot-Limit）：**
- `RAISE_HALF_POT`: 额外加注 = 0.5 × (pot + call_amount)
- `RAISE_POT`: 额外加注 = 1.0 × (pot + call_amount)

### 5. PokerEnv (`engine/poker_env.py`)

```python
from engine.poker_env import PokerEnv

env = PokerEnv()

# 重置环境，返回P0的观察
obs = env.reset()

# 执行动作（当前玩家自动从env获取）
obs, reward, done, info = env.step(action)

# 查看指定玩家的观察（POMDP）
obs_p1 = env.observe(player_idx=1)

# 属性
env.current_player   # int
env.is_terminal      # bool
```

**Observation字典结构：**

| Key | 类型 | 说明 |
|-----|------|------|
| `hand` | List[Card] | 自己的手牌 |
| `community_cards` | List[Card] | 当前公共牌 |
| `pot` | int | 底池大小 |
| `stage` | Stage | 当前阶段 |
| `my_bet` | int | 自己当前街下注额 |
| `opp_bet` | int | 对手当前街下注额 |
| `my_chips` | int | 自己剩余筹码 |
| `opp_chips` | int | 对手剩余筹码 |
| `my_total_bet` | int | 自己本局总下注 |
| `opp_total_bet` | int | 对手本局总下注 |
| `is_folded` | bool | 自己是否已弃牌 |
| `opp_folded` | bool | 对手是否已弃牌 |
| `legal_actions` | List[Action] | 当前合法动作 |
| `current_player` | int | 当前行动玩家 |
| `dealer_idx` | int | 庄家位置 |
| `raises_this_street` | int | 当前街加注次数 |
| `betting_history` | List[Tuple] | 完整下注历史 |

**Step返回值：**

| 字段 | 说明 |
|------|------|
| `obs` | 行动后当前玩家的观察 |
| `reward` | 归一化收益（payoff/INITIAL_CHIPS），非终局为0 |
| `done` | 游戏是否结束 |
| `info` | 终局时: `{'payoffs': [int, int], 'winner': int}` |

### 6. BaseAgent (`agents/base_agent.py`)

所有智能体的抽象基类：

```python
from agents.base_agent import BaseAgent
from config import Action

class MyAgent(BaseAgent):
    def act(self, observation: dict, legal_actions: list) -> Action:
        # 必须实现：根据观察和合法动作返回一个动作
        return Action.CALL

    def reset(self):
        # 可选覆写：每局开始时调用
        pass

    def observe_transition(self, state, action, reward, next_state, done):
        # 可选覆写：观察到一次转移（RL智能体用于学习）
        pass
```

---

## 分工实现指引

### Linpei Duan — ExpectimaxAgent

**文件：** `agents/expectimax_agent.py`

```python
from agents.base_agent import BaseAgent
from engine.hand_evaluator import HandEvaluator
from engine.rule_engine import RuleEngine
from config import Action

class ExpectimaxAgent(BaseAgent):
    def __init__(self, hand_evaluator=None, rule_engine=None, depth_limit=3, num_samples=100):
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.rule_engine = rule_engine or RuleEngine()
        self.depth_limit = depth_limit
        self.num_samples = num_samples

    def act(self, observation, legal_actions):
        # 遍历所有合法动作，用Expectimax搜索计算期望收益
        # 返回期望收益最大的动作
        best_action = legal_actions[0]
        best_value = float('-inf')
        for action in legal_actions:
            value = self._expectimax(observation, action, depth=0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _expectimax(self, obs, action, depth):
        # TODO: 实现Expectimax搜索
        # 决策节点：max over actions
        # 机会节点：expectation over opponent hands / community cards
        # 叶子节点：启发式评估函数
        pass

    def _evaluate_leaf(self, observation):
        # 启发式评估：手牌强度 + 底池赔率 + 位置
        win_rate = self.hand_evaluator.monte_carlo_win_rate(
            observation['hand'], observation['community_cards'], self.num_samples
        )
        pot_odds = self.rule_engine.pot_odds(
            observation['opp_bet'] - observation['my_bet'], observation['pot']
        )
        # TODO: 综合评估
        pass
```

**关键点：**
- 搜索树包含决策节点（max）和机会节点（expectation）
- 机会节点需要Monte Carlo采样对手手牌
- 叶子节点用手牌强度+底池赔率启发式评估
- 可从 `observation` 中提取所有需要的信息
- `RuleEngine.apply_action()` 可模拟后续状态（注意：它返回新状态，不修改原状态）
- `RuleEngine.get_legal_actions()` 获取合法动作列表

---

### Linpei Duan — GTOAgent

**文件：** `agents/gto_agent.py`

```python
from agents.base_agent import BaseAgent
from engine.hand_evaluator import HandEvaluator
from engine.rule_engine import RuleEngine
from config import Action
import random

class GTOAgent(BaseAgent):
    def __init__(self, hand_evaluator=None, epsilon=0.1):
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.rule_engine = rule_engine or RuleEngine()
        self.epsilon = epsilon

    def act(self, observation, legal_actions):
        # 1. 计算每个动作的期望收益 (EV)
        evs = {a: self._compute_ev(a, observation) for a in legal_actions}
        # 2. 选择EV最大的动作
        best_action = max(evs, key=evs.get)
        # 3. ε-greedy混合策略
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        return best_action

    def _compute_ev(self, action, observation):
        # TODO: 基于手牌强度和底池赔率计算期望收益
        # EV(call)  = win_rate * pot - (1 - win_rate) * call_amount
        # EV(raise) = win_rate * (pot + raise_amount) - (1 - win_rate) * (call_amount + raise_amount)
        # EV(fold)  = 0
        pass
```

**关键点：**
- EV计算需要 `HandEvaluator.monte_carlo_win_rate()` 计算胜率
- EV(fold) = 0，EV(call) 和 EV(raise) 基于胜率和底池大小
- ε-greedy：1-ε概率选最优动作，ε概率均匀随机
- 这是博弈论纳什均衡的工程化近似

---

### Linpei Duan — 对手建模 (opponent_model)

**文件：** `agents/opponent_model.py`

```python
class OpponentModel:
    def __init__(self):
        self.aggression_count = 0   # 对手加注次数
        self.total_actions = 0      # 对手总动作数

    def update(self, action):
        # 记录对手动作，更新进攻频率
        self.total_actions += 1
        if action in (Action.RAISE_HALF_POT, Action.RAISE_POT):
            self.aggression_count += 1

    @property
    def aggression_rate(self):
        # 对手进攻频率 = 加注次数 / 总次数
        if self.total_actions == 0:
            return 0.0
        return self.aggression_count / self.total_actions

    def adjusted_hand_strength(self, base_win_rate):
        # 根据对手行为调整手牌强度估计
        # 对手越激进 -> 其手牌可能越强 -> 我们的胜率可能被高估
        # TODO: 实现贝叶斯更新或简化版本
        pass
```

---

### Mohan Li — 评估系统

**文件：** `evaluation/evaluator.py`

```python
class Evaluator:
    def run_match(self, agent_a, agent_b, num_games=1000):
        """
        运行agent_a vs agent_b对战
        返回: {
            'win_rate_a': float,       # agent_a胜率
            'avg_reward_a': float,     # agent_a平均收益
            'avg_reward_b': float,     # agent_b平均收益
            'action_entropy_a': float, # agent_a动作熵
            'action_entropy_b': float, # agent_b动作熵
            'actions_a': dict,         # agent_a动作频率分布
            'actions_b': dict,         # agent_b动作频率分布
        }
        """
        pass

    def run_tournament(self, agents, agent_names, num_games=1000):
        """
        多智能体循环对战
        返回: pd.DataFrame (行=agent, 列=统计指标)
        """
        pass
```

**对战流程参考：**

```python
from engine import PokerEnv
from config import Action

env = PokerEnv()
agent_a = ExpectimaxAgent()
agent_b = RandomAgent()

for game in range(num_games):
    obs = env.reset()
    done = False
    while not done:
        current = env.current_player
        agent = agent_a if current == 0 else agent_b
        action = agent.act(obs, obs['legal_actions'])
        obs, reward, done, info = env.step(action)
    # 记录结果: info['winner'], info['payoffs']
```

**文件：** `evaluation/visualize.py`

```python
class Visualizer:
    def plot_win_rates(self, results_dict, save_path=None):
        # 绘制各agent胜率柱状图
        pass

    def plot_training_curve(self, rewards_history, save_path=None):
        # 绘制RL训练曲线
        pass

    def plot_action_distribution(self, action_counts, save_path=None):
        # 绘制动作分布饼图
        pass
```

---

## 状态空间设计

### RL状态编码（11桶模式）

| 特征 | 桶数 | 说明 |
|------|------|------|
| hand_strength | 11 | MC胜率 → 0..10 |
| board_strength | 5 | 公共牌牌力 → 0..4 |
| pot_size | 5 | 底池桶 → 0..4 |
| stage | 4 | PREFLOP/FLOP/TURN/RIVER |

总状态数: 11 × 5 × 5 × 4 = 1,100

### 简化状态编码（3桶模式，用于GTO/Expectimax）

| 特征 | 桶数 | 说明 |
|------|------|------|
| hand_strength | 3 | 弱/中/强 |
| pot_size | 5 | 底池桶 → 0..4 |
| stage | 4 | PREFLOP/FLOP/TURN/RIVER |

---

## 常见问题

**Q: 如何获取对手手牌用于Monte Carlo采样？**
A: 对手手牌是隐藏信息。在MC模拟中，从剩余牌中随机采样对手可能的手牌来估计胜率，而不是知道真实的对手手牌。参见 `HandEvaluator.monte_carlo_win_rate()` 的实现。

**Q: `apply_action()` 是否修改原状态？**
A: 不修改。`RuleEngine.apply_action()` 返回一个深拷贝的新状态，原状态保持不变。这在搜索树遍历中很重要。

**Q: 筹码用完怎么办？**
A: 当玩家筹码不足以支付时，会自动全下（`is_all_in=True`），后续不再需要行动。

**Q: 如何调试单局游戏？**
A: 使用 `env.observe(player_idx)` 查看任意玩家的POMDP观察，结合 `betting_history` 追踪完整对局过程。