"""
OpponentModel — 对手行为追踪与建模

功能：
  1. 追踪对手历史动作，按阶段统计频率
  2. 计算对手风格指标（进攻率、弃牌率、阶段倾向）
  3. 基于对手行为调整手牌胜率估计（贝叶斯直觉）

设计动机：
  德州扑克的 POMDP 特性意味着对手手牌是隐藏信息。但对手的 betting pattern
  泄露了其手牌强度的部分信息。通过追踪对手历史行为，我们可以推断其「玩家类型」
  （激进 / 被动 / 紧 / 松），从而在决策时调整对手手牌强度分布的先验。

与 Agent 集成：
  - GTOAgent: 在 _compute_ev 中调用 adjusted_hand_strength 修正 win_rate
  - ExpectimaxAgent: 在机会节点采样对手手牌时加 bias
"""
from config import Action, Stage


class OpponentModel:
    """对手行为模型。

    追踪对手在不同游戏阶段的动作频率，计算进攻性、弃牌倾向等指标，
    并提供基于对手行为的手牌胜率修正。

    Attributes:
        total_actions: 对手总动作数
        stage_actions: 按阶段分组的动作计数 {Stage: {Action: count}}
        aggression_count: 加注总次数 (RAISE_HALF_POT + RAISE_POT)
        fold_count: 弃牌总次数
    """

    def __init__(self):
        # 按阶段的动作计数
        self.stage_actions = {
            Stage.PREFLOP: {Action.FOLD: 0, Action.CALL: 0,
                           Action.RAISE_HALF_POT: 0, Action.RAISE_POT: 0},
            Stage.FLOP:    {Action.FOLD: 0, Action.CALL: 0,
                           Action.RAISE_HALF_POT: 0, Action.RAISE_POT: 0},
            Stage.TURN:    {Action.FOLD: 0, Action.CALL: 0,
                           Action.RAISE_HALF_POT: 0, Action.RAISE_POT: 0},
            Stage.RIVER:   {Action.FOLD: 0, Action.CALL: 0,
                           Action.RAISE_HALF_POT: 0, Action.RAISE_POT: 0},
        }

        # 汇总计数
        self.total_actions = 0
        self.aggression_count = 0
        self.fold_count = 0

    # ── 更新接口 ──────────────────────────────────────────────

    def update(self, action, stage=None):
        """记录对手的一次动作。

        Args:
            action: Action 枚举值 (FOLD/CALL/RAISE_HALF_POT/RAISE_POT)
            stage: 当前游戏阶段（可选，用于分阶段统计）
        """
        self.total_actions += 1

        if action in (Action.RAISE_HALF_POT, Action.RAISE_POT):
            self.aggression_count += 1
        elif action == Action.FOLD:
            self.fold_count += 1

        if stage is not None and stage in self.stage_actions:
            self.stage_actions[stage][action] += 1

    # ── 全局风格指标 ──────────────────────────────────────────

    @property
    def aggression_rate(self) -> float:
        """对手整体进攻频率。

        进攻率 = 加注次数 / 总动作数。
        高进攻率 (>0.35) → 激进型玩家 (LAG)
        低进攻率 (<0.15) → 被动型玩家 (Nit/Passive)
        """
        if self.total_actions == 0:
            return 0.0
        # 默认先验（没有数据时假设对手中性）
        prior = 0.25
        prior_weight = 5.0  # 相当于 5 次虚拟样本
        observed = self.aggression_count / self.total_actions
        return (prior * prior_weight + observed * self.total_actions) / (prior_weight + self.total_actions)

    @property
    def fold_rate(self) -> float:
        """对手整体弃牌频率。

        高弃牌率 (>0.30) → 紧的玩家，容易通过 bluff 剥削
        低弃牌率 (<0.15) → 松的玩家，bluff 收益低
        """
        if self.total_actions == 0:
            return 0.0
        prior = 0.20
        prior_weight = 5.0
        observed = self.fold_count / self.total_actions
        return (prior * prior_weight + observed * self.total_actions) / (prior_weight + self.total_actions)

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot — 对手主动入池频率。

        VPIP = (跟注次数 + 加注次数) / 总动作数。
        高 VPIP (>0.70) → 松的玩家
        低 VPIP (<0.50) → 紧的玩家
        """
        if self.total_actions == 0:
            return 0.0
        folds = self.fold_count
        vpip_actions = self.total_actions - folds
        return vpip_actions / self.total_actions

    # ── 分阶段指标 ────────────────────────────────────────────

    def stage_aggression(self, stage) -> float:
        """特定阶段的进攻频率。

        Args:
            stage: Stage 枚举值

        Returns:
            该阶段的加注比例 [0, 1]。无数据时返回全局进攻率。
        """
        actions = self.stage_actions.get(stage, {})
        total = sum(actions.values())
        if total == 0:
            return self.aggression_rate  # fallback 到全局
        raises = actions.get(Action.RAISE_HALF_POT, 0) + actions.get(Action.RAISE_POT, 0)
        return raises / total

    def stage_fold_rate(self, stage) -> float:
        """特定阶段的弃牌频率。"""
        actions = self.stage_actions.get(stage, {})
        total = sum(actions.values())
        if total == 0:
            return self.fold_rate
        return actions.get(Action.FOLD, 0) / total

    # ── 手牌胜率调整 ──────────────────────────────────────────

    def adjusted_hand_strength(self, base_win_rate, opponent_last_action, stage=None):
        """基于对手行为修正我方手牌胜率估计。

        核心直觉（贝叶斯）：
          - 对手加注 → 暗示强牌 → 下调我方的 win_rate 估计
          - 对手激进但只跟注 → 实际上可能较弱（未加注 = 示弱信号）
          - 对手很紧但不弃牌 → 有真实牌力
          - 对手很松 → 他们可能在 bluff / 玩弱牌 → 上调用 win_rate

        Args:
            base_win_rate: Monte Carlo 模拟的基础胜率 [0, 1]
            opponent_last_action: 对手最近一次动作
            stage: 当前阶段（可选，用于分阶段行为解读）

        Returns:
            调整后的胜率估计 [0, 1]
        """
        multiplier = 1.0

        agg = self.aggression_rate
        fold = self.fold_rate

        # ── 对手动作信号 ──
        if opponent_last_action in (Action.RAISE_HALF_POT, Action.RAISE_POT):
            # 对手加注：通常意味着强牌
            if agg > 0.40:
                # 非常激进的对手：加注可能是 bluff，信号较弱
                multiplier = 0.92
            elif agg > 0.25:
                multiplier = 0.85
            else:
                # 被动的对手突然加注 → 非常可疑 → 大幅下调
                multiplier = 0.78

        elif opponent_last_action == Action.CALL:
            # 对手跟注：中性信号，但结合对手风格
            if fold > 0.30:
                # 紧的对手跟注 → 有牌
                multiplier = 0.93
            elif agg < 0.15:
                # 被动的对手跟注 → 正常，轻微优势
                multiplier = 1.02

        # ── 分阶段修正 ──
        if stage is not None and self.total_actions >= 10:
            stage_agg = self.stage_aggression(stage)
            stage_fold = self.stage_fold_rate(stage)

            # 对手在 River 经常弃牌 → 他们在早期街可能 bluff → 在早期街可以乐观一点
            if stage == Stage.FLOP or stage == Stage.TURN:
                if self.stage_fold_rate(Stage.RIVER) > 0.40:
                    multiplier = min(multiplier, 1.0) * 1.04

            # 对手在 Preflop 很激进但 Flop 只跟 → 可能是 continuation bet 爱好者
            if stage == Stage.FLOP:
                if self.stage_aggression(Stage.PREFLOP) > 0.50 and stage_agg < 0.20:
                    multiplier = min(multiplier, 1.0) * 1.05

        # ── 样本量不足时降低调整幅度（避免过拟合） ──
        if self.total_actions < 20:
            blend = self.total_actions / 20.0
            multiplier = 1.0 - (1.0 - multiplier) * blend

        adjusted = base_win_rate * multiplier
        return max(0.0, min(1.0, adjusted))

    # ── 对手类型标签 ──────────────────────────────────────────

    @property
    def player_type(self) -> str:
        """推断对手的玩家类型标签。

        Returns:
            玩家类型描述字符串，如 'TAG' / 'LAG' / 'Nit' / 'Calling Station' 等
        """
        if self.total_actions < 10:
            return "Unknown"

        agg = self.aggression_rate
        vpip = self.vpip

        if agg > 0.35 and vpip > 0.65:
            return "LAG (松凶)"
        elif agg > 0.35 and vpip <= 0.65:
            return "TAG (紧凶)"
        elif agg <= 0.20 and vpip > 0.65:
            return "Calling Station (跟注站)"
        elif agg <= 0.20 and vpip <= 0.50:
            return "Nit (极紧)"
        elif agg <= 0.20:
            return "Passive (被动)"
        else:
            return "Balanced (均衡)"

    # ── 工具方法 ──────────────────────────────────────────────

    def reset(self):
        """重置所有统计数据（通常不需要，因为对手建模应该跨局持久）。"""
        self.total_actions = 0
        self.aggression_count = 0
        self.fold_count = 0
        for stage_dict in self.stage_actions.values():
            for action in stage_dict:
                stage_dict[action] = 0

    def summary(self) -> dict:
        """返回对手统计摘要。"""
        return {
            'total_actions': self.total_actions,
            'aggression_rate': round(self.aggression_rate, 3),
            'fold_rate': round(self.fold_rate, 3),
            'vpip': round(self.vpip, 3),
            'player_type': self.player_type,
            'preflop_agg': round(self.stage_aggression(Stage.PREFLOP), 3),
            'flop_agg': round(self.stage_aggression(Stage.FLOP), 3),
            'turn_agg': round(self.stage_aggression(Stage.TURN), 3),
            'river_agg': round(self.stage_aggression(Stage.RIVER), 3),
        }
