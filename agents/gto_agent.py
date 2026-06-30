"""
GTOAgent — 基于博弈论纳什均衡近似的德州扑克智能体

核心思想：
  对每个合法动作计算期望收益 (Expected Value, EV)，选择 EV 最大的动作。
  结合 ε-greedy 实现混合策略，避免被对手 exploit。

EV 公式：
  EV(FOLD) = 0
  EV(CALL) = win_rate × pot - (1 - win_rate) × call_amount
  EV(RAISE) = win_rate × (pot + raise_amount) - (1 - win_rate) × (call_amount + raise_amount)

  其中 win_rate 通过 Monte Carlo 模拟估计我方胜率。

ε-greedy 混合策略：
  1-ε 概率选最优动作，ε 概率均匀随机。
  这是博弈论纳什均衡的工程化近似 — 纯策略易被 exploit，混合策略更难对付。
"""
import random

from agents.base_agent import BaseAgent
from engine.hand_evaluator import HandEvaluator
from engine.rule_engine import RuleEngine
from config import Action, MONTE_CARLO_SIMULATIONS_DEFAULT


class GTOAgent(BaseAgent):
    """GTO (Game Theory Optimal) 启发式智能体。

    根据手牌胜率和底池赔率，用 EV 公式评估每个合法动作的期望收益，
    通过 ε-greedy 实现混合策略。

    Attributes:
        epsilon: 随机探索概率（默认 0.1），更大的 epsilon = 更多混合/更难预测
    """

    def __init__(
        self,
        hand_evaluator=None,
        rule_engine=None,
        epsilon=0.1,
        num_simulations=MONTE_CARLO_SIMULATIONS_DEFAULT,
    ):
        self.hand_evaluator = hand_evaluator or HandEvaluator()
        self.rule_engine = rule_engine or RuleEngine()
        self.epsilon = epsilon
        self.num_simulations = num_simulations

    def act(self, observation, legal_actions):
        """选择动作：ε-随机探索 或 EV 最大化。

        Args:
            observation: PokerEnv.observe() 返回的字典，含 hand, pot, my_bet, opp_bet 等
            legal_actions: 当前合法动作列表

        Returns:
            Action 枚举值
        """
        if not legal_actions:
            return Action.FOLD

        # ε-greedy: 以 epsilon 概率随机选动作（混合策略）
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # 否则选 EV 最大的动作
        evs = {action: self._compute_ev(action, observation) for action in legal_actions}
        best_action = max(evs, key=lambda a: evs[a])
        return best_action

    def _compute_ev(self, action, observation):
        """计算给定动作的期望收益 (Expected Value)。

        对 FOLD / CALL / RAISE 分别计算。对 CALL 和 RAISE 需要
        Monte Carlo 估计胜率。

        Args:
            action: 要评估的动作
            observation: 当前观察字典

        Returns:
            float: 该动作的期望收益（筹码单位）
        """
        if action == Action.FOLD:
            return 0.0

        hand = observation.get('hand', [])
        community_cards = observation.get('community_cards', [])
        pot = observation.get('pot', 0)
        my_bet = observation.get('my_bet', 0)
        opp_bet = observation.get('opp_bet', 0)

        # 跟注金额：对手下注 - 我方已下注
        call_amount = max(0, opp_bet - my_bet)

        # Monte Carlo 估计当前胜率
        win_rate = self.hand_evaluator.monte_carlo_win_rate(
            hand, community_cards, self.num_simulations
        )
        lose_rate = 1.0 - win_rate

        if action == Action.CALL:
            # EV(call) = P(win)×pot - P(lose)×call
            return win_rate * pot - lose_rate * call_amount

        elif action == Action.RAISE_HALF_POT:
            # 加注额 = 0.5 × (pot + call_amount)
            pot_after_call = pot + call_amount
            raise_amount = int(pot_after_call * 0.5)
            total_put = call_amount + raise_amount
            return win_rate * (pot + raise_amount) - lose_rate * total_put

        elif action == Action.RAISE_POT:
            # 加注额 = 1.0 × (pot + call_amount)
            pot_after_call = pot + call_amount
            raise_amount = int(pot_after_call * 1.0)
            total_put = call_amount + raise_amount
            return win_rate * (pot + raise_amount) - lose_rate * total_put

        return 0.0

    def reset(self):
        """每局开始时重置内部状态（GTOAgent 无持久状态，无需操作）。"""
        pass
