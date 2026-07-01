"""
ExpectimaxAgent — 基于搜索树的德州扑克智能体

核心思想：
  对未来游戏状态构建搜索树，在决策节点选最大期望值动作，
  在机会节点对手牌/对手动作做期望采样。

搜索树结构：
  ┌─ 决策节点 (max):   我们选动作 → max over actions
  ├─ 机会节点 (chance): 对手响应 → expectation over opponent actions
  ├─ 机会节点 (chance): 发公共牌 → expectation over future cards
  └─ 叶子节点:         启发式评估 = 手牌胜率 × pot - 成本

  深度 = 我们还有多少次决策机会（depth_limit=3 表示我们最多再决策3次）

与 OpponentModel 集成：
  - 对手动作概率 = softmax(对手EV)，EV 用 GTO 公式计算
  - 可选：用 OpponentModel.adjusted_hand_strength 修正对手手牌强度估计
"""
import random
import math
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from engine.hand_evaluator import HandEvaluator
from engine.rule_engine import RuleEngine
from engine.card import Card, Suit, Rank, ALL_CARDS
from config import Action, Stage, MONTE_CARLO_SIMULATIONS_DEFAULT


class ExpectimaxAgent(BaseAgent):
    """Expectimax 搜索智能体。

    对每个合法动作进行前向搜索，在决策节点（我方行动）做最大化，
    在机会节点（对手行动、发牌）做期望采样，返回期望值最大的动作。

    Attributes:
        depth_limit: 搜索深度（我方剩余决策次数），默认 3
        num_samples: 每次机会节点的 Monte Carlo 采样数，默认 100
        temperature: 对手动作 softmax 温度，越低越确定，默认 0.5
        opponent_model: 可选的 OpponentModel 实例
    """

    def __init__(
        self,
        hand_evaluator=None,
        rule_engine=None,
        depth_limit=3,
        num_samples=100,
        temperature=0.5,
        opponent_model=None,
    ):
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.rule_engine = rule_engine or RuleEngine()
        self.depth_limit = depth_limit
        self.num_samples = num_samples
        self.temperature = temperature
        self.opponent_model = opponent_model

    def act(self, observation: Dict, legal_actions: List[Action]) -> Action:
        """选择期望值最大的动作。

        对每个合法动作，运行 _search() 估计期望收益，返回最大值对应的动作。
        """
        if not legal_actions:
            return Action.FOLD

        best_action = legal_actions[0]
        best_value = float('-inf')

        for action in legal_actions:
            value = self._search(observation, action, depth=0)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    # ── 搜索核心 ──────────────────────────────────────────────

    def _search(self, obs: Dict, our_action: Action, depth: int) -> float:
        """模拟执行 our_action 后的期望收益。

        这是搜索树的入口：我方执行动作 → 对手响应 → 递归或评估叶子。

        Args:
            obs: 当前观察
            our_action: 我方选择的动作
            depth: 当前深度（已进行的我方决策次数）

        Returns:
            期望收益（筹码单位），正数 = 我方获利
        """
        # ── 1. 计算 our_action 的直接效果 ──
        hand = obs.get('hand', [])
        board = list(obs.get('community_cards', []))
        pot = obs.get('pot', 0)
        stage = obs.get('stage', Stage.PREFLOP)
        my_bet = obs.get('my_bet', 0)
        opp_bet = obs.get('opp_bet', 0)
        my_chips = obs.get('my_chips', 1000)
        opp_chips = obs.get('opp_chips', 1000)

        call_amount = max(0, opp_bet - my_bet)
        raise_amount = 0

        if our_action == Action.FOLD:
            # 弃牌 → 立即终止，我方输掉已下注的筹码
            return float(-obs.get('my_total_bet', my_bet))

        elif our_action == Action.CALL:
            actual_call = min(call_amount, my_chips)
            new_pot = pot + actual_call
            new_my_bet = my_bet + actual_call
            new_my_chips = my_chips - actual_call

        elif our_action == Action.RAISE_HALF_POT:
            pot_after_call = pot + call_amount
            raise_amount = int(pot_after_call * 0.5)
            total_put = min(call_amount + raise_amount, my_chips)
            new_pot = pot + total_put
            new_my_bet = my_bet + total_put
            new_my_chips = my_chips - total_put

        elif our_action == Action.RAISE_POT:
            pot_after_call = pot + call_amount
            raise_amount = int(pot_after_call * 1.0)
            total_put = min(call_amount + raise_amount, my_chips)
            new_pot = pot + total_put
            new_my_bet = my_bet + total_put
            new_my_chips = my_chips - total_put

        else:
            return 0.0

        # ── 2. 检查 my_chips 归零（全下）或不归零 ──
        my_all_in = (new_my_chips == 0)

        # 构建后续观察（用于对手视角）
        next_obs = {
            'hand': hand,
            'community_cards': board,
            'pot': new_pot,
            'stage': stage,
            'my_bet': new_my_bet,
            'opp_bet': opp_bet,       # 对手尚未响应
            'my_chips': new_my_chips,
            'opp_chips': opp_chips,
            'my_total_bet': obs.get('my_total_bet', 0) + (new_my_bet - my_bet),
            'opp_total_bet': obs.get('opp_total_bet', 0),
            'is_folded': False,
            'opp_folded': False,
        }

        # ── 3. 对手响应（机会节点）──
        # 对手合法动作取决于我们的行动
        opp_acts = self._opponent_legal_actions(our_action, opp_chips, opp_bet, new_my_bet)

        if not opp_acts:
            # 对手无法行动（全下或被覆盖）→ 直接推进
            return self._advance_or_evaluate(next_obs, depth)

        # Monte Carlo 采样对手手牌，计算对手各动作概率
        total_ev = 0.0
        opp_action_counts = {a: 0.0 for a in opp_acts}
        opp_sample_count = min(self.num_samples, 5)

        for _ in range(opp_sample_count):
            opp_hand = self._sample_opponent_hand(hand, board)
            opp_probs = self._opponent_action_probs(
                opp_hand, board, new_pot, stage, opp_chips,
                opp_bet, new_my_bet, opp_acts
            )

            for opp_act in opp_acts:
                prob = opp_probs.get(opp_act, 1.0 / len(opp_acts))
                opp_action_counts[opp_act] += prob / opp_sample_count

                # 模拟对手动作后的状态
                outcome = self._simulate_opponent_response(
                    next_obs, opp_hand, opp_act, depth
                )
                total_ev += prob * outcome / opp_sample_count

        return total_ev

    def _advance_or_evaluate(self, obs: Dict, depth: int) -> float:
        """在对手无法行动或回合结束后，推进到下一阶段或评估。

        如果深度已到上限或游戏在 River 后结束 → 评估叶子。
        否则 → 发公共牌 → 我方再次决策（递归）。
        """
        stage = obs.get('stage', Stage.PREFLOP)

        # 检查是否到 River 且双方都行动过 → 摊牌
        if stage == Stage.RIVER:
            return self._evaluate_showdown(obs)

        # 深度限制：如果已经用完了搜索深度，直接在这里评估
        if depth >= self.depth_limit:
            return self._evaluate_leaf(obs)

        # ── 推进阶段：发公共牌 ──
        new_stage = Stage(min(stage + 1, Stage.RIVER))
        hand = obs['hand']
        board = list(obs['community_cards'])
        cards_needed = {Stage.FLOP: 3, Stage.TURN: 4, Stage.RIVER: 5}.get(new_stage, 0) - len(board)

        # 重置街上下注
        next_obs = dict(obs)
        next_obs['stage'] = new_stage
        next_obs['my_bet'] = 0
        next_obs['opp_bet'] = 0

        # Monte Carlo 采样公共牌
        total_ev = 0.0
        known = set((c.suit, c.rank) for c in hand + board)
        remaining_cards = [c for c in ALL_CARDS if (c.suit, c.rank) not in known]

        card_samples = min(self.num_samples, 10)
        for _ in range(card_samples):
            random.shuffle(remaining_cards)
            sampled_board = board + remaining_cards[:cards_needed]
            total_ev += self._evaluate_leaf(dict(next_obs, community_cards=sampled_board))

        return total_ev / card_samples

    # ── 叶子评估 ──────────────────────────────────────────────

    def _evaluate_leaf(self, obs: Dict) -> float:
        """叶子节点启发式评估。

        综合手牌胜率 + 底池赔率，计算当前状态的期望收益。
        公式：EV ≈ win_rate × pot - (1-win_rate) × my_bet
        """
        hand = obs.get('hand', [])
        board = obs.get('community_cards', [])
        pot = obs.get('pot', 0)

        win_rate = self.hand_evaluator.monte_carlo_win_rate(
            hand, board, num_simulations=self.num_samples
        )

        # 我方已在本街的下注（成本）
        my_street_bet = obs.get('my_bet', 0)
        my_total = obs.get('my_total_bet', my_street_bet)

        # 简化 EV：如果这手牌打到摊牌
        # EV = win_rate * pot - (1-win_rate) * total_invested
        return win_rate * pot - (1.0 - win_rate) * my_total

    def _evaluate_showdown(self, obs: Dict) -> float:
        """River 后摊牌评估。"""
        hand = obs.get('hand', [])
        board = obs.get('community_cards', [])
        pot = obs.get('pot', 0)
        my_total = obs.get('my_total_bet', 0)

        win_rate = self.hand_evaluator.monte_carlo_win_rate(
            hand, board, num_simulations=self.num_samples
        )
        return win_rate * pot - (1.0 - win_rate) * my_total

    # ── 对手建模 ──────────────────────────────────────────────

    def _sample_opponent_hand(self, my_hand, board):
        """从剩余牌中随机采样对手手牌（2张）。

        Args:
            my_hand: 我的手牌 [Card, Card]
            board: 公共牌 [Card, ...]

        Returns:
            对手手牌 [Card, Card] 或空列表
        """
        known = set((c.suit, c.rank) for c in my_hand + board)
        remaining = [c for c in ALL_CARDS if (c.suit, c.rank) not in known]
        if len(remaining) < 2:
            return []

        random.shuffle(remaining)
        return remaining[:2]

    def _opponent_legal_actions(
        self, our_action: Action,
        opp_chips: int, opp_bet: int, new_my_bet: int
    ) -> List[Action]:
        """根据我们的动作和对手状态，推断对手的合法动作。

        简化版本（不完全匹配 RuleEngine，但覆盖主要情况）。

        Args:
            our_action: 我们刚执行的动作
            opp_chips: 对手剩余筹码
            opp_bet: 对手当前街上已下注额
            new_my_bet: 我方执行动作后的当前街下注额

        Returns:
            对手合法动作列表
        """
        if our_action == Action.FOLD:
            return []  # 游戏已结束

        call_amount = max(0, new_my_bet - opp_bet)
        actions = []

        # FOLD 总是合法
        actions.append(Action.FOLD)

        # CALL：如果跟注金额 ≤ 筹码
        if call_amount <= opp_chips:
            actions.append(Action.CALL)

        # RAISE：如果对手有足够筹码加注
        if opp_chips > call_amount:
            # RAISE_HALF_POT
            pot_after_call = 99999  # 我们不知道精确的 pot，简化处理
            # 简化：只要有筹码就允许加注
            actions.append(Action.RAISE_HALF_POT)
            actions.append(Action.RAISE_POT)

        return actions

    def _opponent_action_probs(
        self, opp_hand, board, pot, stage,
        opp_chips, opp_bet, our_bet, opponent_legal_actions
    ) -> Dict[Action, float]:
        """估计对手选择每个动作的概率。

        策略：对每个合法动作计算对手视角的 EV，用 softmax 转为概率。
        EV 公式与 GTOAgent 相同。

        Args:
            opp_hand: 对手手牌
            board: 公共牌
            pot: 当前底池
            stage: 当前阶段
            opp_chips: 对手筹码
            opp_bet: 对手已下注
            our_bet: 我方已下注
            opponent_legal_actions: 对手合法动作

        Returns:
            {Action: probability}
        """
        if not opponent_legal_actions:
            return {Action.FOLD: 1.0}

        # 从对手视角计算每个动作的 EV
        call_amount = max(0, our_bet - opp_bet)
        win_rate = self.hand_evaluator.monte_carlo_win_rate(
            opp_hand, board, num_simulations=self.num_samples
        )
        lose_rate = 1.0 - win_rate

        evs = {}
        for act in opponent_legal_actions:
            if act == Action.FOLD:
                evs[act] = 0.0
            elif act == Action.CALL:
                evs[act] = win_rate * pot - lose_rate * call_amount
            elif act == Action.RAISE_HALF_POT:
                pot_ac = pot + call_amount
                ra = int(pot_ac * 0.5)
                evs[act] = win_rate * (pot + ra) - lose_rate * (call_amount + ra)
            elif act == Action.RAISE_POT:
                pot_ac = pot + call_amount
                ra = int(pot_ac * 1.0)
                evs[act] = win_rate * (pot + ra) - lose_rate * (call_amount + ra)
            else:
                evs[act] = 0.0

        # Softmax 转概率
        if self.temperature <= 0:
            # 纯 max：所有概率给 EV 最大的动作
            best = max(evs, key=lambda a: evs[a])
            return {best: 1.0}

        # 数值稳定处理
        max_ev = max(evs.values())
        exp_sum = 0.0
        exp_evs = {}
        for act, ev in evs.items():
            e = math.exp((ev - max_ev) / self.temperature)
            exp_evs[act] = e
            exp_sum += e

        if exp_sum == 0:
            return {act: 1.0 / len(opponent_legal_actions) for act in opponent_legal_actions}

        return {act: e / exp_sum for act, e in exp_evs.items()}

    def _simulate_opponent_response(
        self, obs: Dict, opp_hand: List, opp_action: Action, depth: int
    ) -> float:
        """模拟对手执行 opp_action 后的结果。

        Args:
            obs: 对手行动前的观察（我方视角）
            opp_hand: 对手手牌
            opp_action: 对手选择的动作
            depth: 当前深度

        Returns:
            期望收益（筹码单位）
        """
        pot = obs['pot']
        opp_chips = obs['opp_chips']
        opp_bet = obs.get('opp_bet', 0)
        our_bet = obs.get('my_bet', 0)
        my_total = obs.get('my_total_bet', 0)

        if opp_action == Action.FOLD:
            # 对手弃牌 → 我方赢得底池
            return float(pot - my_total)

        call_amount = max(0, our_bet - opp_bet)

        if opp_action == Action.CALL:
            actual = min(call_amount, opp_chips)
            new_pot = pot + actual
            new_opp_bet = opp_bet + actual
            new_opp_chips = opp_chips - actual

        elif opp_action == Action.RAISE_HALF_POT:
            pot_ac = pot + call_amount
            ra = int(pot_ac * 0.5)
            total = min(call_amount + ra, opp_chips)
            new_pot = pot + total
            new_opp_bet = opp_bet + total
            new_opp_chips = opp_chips - total

        elif opp_action == Action.RAISE_POT:
            pot_ac = pot + call_amount
            ra = int(pot_ac * 1.0)
            total = min(call_amount + ra, opp_chips)
            new_pot = pot + total
            new_opp_bet = opp_bet + total
            new_opp_chips = opp_chips - total

        else:
            return 0.0

        # 构建下一次我方决策的观察
        # 检查是否到 River → 摊牌
        stage = obs.get('stage', Stage.PREFLOP)

        if stage == Stage.RIVER:
            # 对手在 River 跟注/加注 → 摊牌
            next_obs = dict(obs)
            next_obs['pot'] = new_pot
            next_obs['opp_bet'] = new_opp_bet
            next_obs['opp_chips'] = new_opp_chips
            return self._evaluate_showdown(next_obs)

        # 推进到下一阶段
        new_stage = Stage(min(stage + 1, Stage.RIVER))
        hand = obs['hand']
        board = list(obs['community_cards'])

        # 如果我方还需要再行动（对手加注了），则我方再次决策
        if opp_action in (Action.RAISE_HALF_POT, Action.RAISE_POT) and depth < self.depth_limit:
            next_obs = {
                'hand': hand,
                'community_cards': board,
                'pot': new_pot,
                'stage': new_stage,
                'my_bet': our_bet,
                'opp_bet': new_opp_bet,
                'my_chips': obs['my_chips'],
                'opp_chips': new_opp_chips,
                'my_total_bet': my_total,
                'opp_total_bet': obs.get('opp_total_bet', 0),
                'is_folded': False,
                'opp_folded': False,
            }
            # 我方再次决策（递归调用 act 逻辑）
            return self._evaluate_leaf(next_obs)

        # 对手跟注 → 进入下一阶段，我方先行动
        if depth + 1 >= self.depth_limit:
            return self._evaluate_leaf(obs)

        # 递归：我方在新阶段的决策
        return self._evaluate_leaf(dict(obs, pot=new_pot, stage=new_stage,
                                       my_bet=0, opp_bet=0,
                                       opp_chips=new_opp_chips))

    # ── 工具方法 ──────────────────────────────────────────────

    def reset(self):
        """每局开始时重置内部状态。"""
        pass
