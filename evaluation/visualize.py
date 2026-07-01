import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

from config import Action, Stage

_ACTION_NAMES = {
    Action.FOLD: 'Fold',
    Action.CALL: 'Call',
    Action.RAISE_HALF_POT: 'Raise Half',
    Action.RAISE_POT: 'Raise Pot',
}
_STAGE_NAMES = {
    Stage.PREFLOP: 'Preflop',
    Stage.FLOP: 'Flop',
    Stage.TURN: 'Turn',
    Stage.RIVER: 'River',
}
_CPalette = {
    'Random': '#e74c3c',
    'GTO': '#3498db',
    'Expectimax': '#2ecc71',
    'QLearning': '#f39c12',
}


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _blend(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


class Visualizer:

    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'figure.facecolor': 'white',
            'axes.facecolor': '#fafafa',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })

    # ━━━ 1. Tournament Heatmap ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_tournament_heatmap(self, results, agent_names, save_path=None):
        n = len(agent_names)
        win_matrix = np.full((n, n), np.nan)
        payoff_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = '%s vs %s' % (agent_names[i], agent_names[j])
                if key in results:
                    win_matrix[i][j] = results[key]['win_rate_a'] * 100
                    payoff_matrix[i][j] = results[key]['avg_reward_a']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        im1 = ax1.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(agent_names, rotation=30, ha='right')
        ax1.set_yticklabels(agent_names)
        ax1.set_title('Win Rate (%)', fontweight='bold', pad=12)
        ax1.set_xlabel('Opponent (Defender)')
        ax1.set_ylabel('Agent (Attacker)')
        for i in range(n):
            for j in range(n):
                if not np.isnan(win_matrix[i][j]):
                    ax1.text(j, i, '%.1f' % win_matrix[i][j],
                             ha='center', va='center', fontsize=10, fontweight='bold',
                             color='white' if win_matrix[i][j] < 20 or win_matrix[i][j] > 80 else 'black')
                else:
                    ax1.text(j, i, '-', ha='center', va='center', fontsize=12, color='gray')
        fig.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(payoff_matrix, cmap='RdYlBu', aspect='auto')
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(agent_names, rotation=30, ha='right')
        ax2.set_yticklabels(agent_names)
        ax2.set_title('Avg Payoff (chips)', fontweight='bold', pad=12)
        ax2.set_xlabel('Opponent (Defender)')
        ax2.set_ylabel('Agent (Attacker)')
        for i in range(n):
            for j in range(n):
                if not np.isnan(payoff_matrix[i][j]):
                    ax2.text(j, i, '%.1f' % payoff_matrix[i][j],
                             ha='center', va='center', fontsize=9, fontweight='bold',
                             color='white' if abs(payoff_matrix[i][j]) > 80 else 'black')
                else:
                    ax2.text(j, i, '-', ha='center', va='center', fontsize=12, color='gray')
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        fig.suptitle('Tournament Cross-Match Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '1_tournament_heatmap.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 2. Radar Chart ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_radar_chart(self, profiles, agent_names, save_path=None):
        categories = ['Win Rate', 'Avg Payoff', 'Action\nDiversity',
                       'Aggression', 'Showdown\nRate', 'Anti-Exploit']
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for name in agent_names:
            p = profiles.get(name, {})
            values = [
                p.get('win_rate', 0) * 5,
                min(max(p.get('avg_reward', 0) / 100, 0), 2),
                p.get('action_entropy', 0) / 0.5,
                p.get('aggression', 0) / 0.3,
                p.get('showdown_rate', 0) / 0.5,
                min(p.get('anti_exploit', 0.5), 1.0),
            ]
            values += values[:1]
            color = _CPalette.get(name, '#95a5a6')
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax.set_yticklabels(['0.5', '1.0', '1.5', '2.0'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Agent Profile Radar', fontweight='bold', pad=20, fontsize=15)

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '2_radar_chart.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 3. Action Distribution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_action_distribution(self, all_actions, agent_names, save_path=None):
        action_labels = ['Fold', 'Call', 'Raise Half', 'Raise Pot']
        action_keys = [Action.FOLD, Action.CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
        n_agents = len(agent_names)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── (a) Grouped bar ──
        ax = axes[0][0]
        x = np.arange(len(action_labels))
        width = 0.8 / n_agents
        for idx, name in enumerate(agent_names):
            counts = all_actions.get(name, {})
            total = sum(counts.values()) or 1
            rates = [counts.get(k, 0) / total for k in action_keys]
            ax.bar(x + idx * width, rates, width, label=name,
                   color=_CPalette.get(name, '#95a5a6'), alpha=0.85, edgecolor='white')
        ax.set_xticks(x + width * (n_agents - 1) / 2)
        ax.set_xticklabels(action_labels)
        ax.set_ylabel('Frequency')
        ax.set_title('(a) Overall Action Distribution', fontweight='bold')
        ax.legend(fontsize=9)

        # ── (b) Pie charts ──
        ax_row = axes[0][1]
        ax_row.axis('off')
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, min(n_agents, 4),
                                                     subplot_spec=ax_row.get_subplotspec().get_gridspec()[1, 1])
        for idx, name in enumerate(agent_names[:4]):
            ax_pie = fig.add_subplot(gs_inner[0, idx])
            counts = all_actions.get(name, {})
            vals = [counts.get(k, 0) for k in action_keys]
            total = sum(vals) or 1
            sizes = [v / total for v in vals]
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            ax_pie.pie(sizes, labels=action_labels, colors=colors,
                       autopct='%1.0f%%', startangle=90, textprops={'fontsize': 7})
            ax_pie.set_title(name, fontsize=9, fontweight='bold')
        ax_row.set_title('(b) Per-Agent Pie Charts', fontweight='bold', loc='left', pad=20)

        # ── (c) Stage-wise stacked bars ──
        ax = axes[1][0]
        stage_labels = ['Preflop', 'Flop', 'Turn', 'River']
        stage_keys = [Stage.PREFLOP, Stage.FLOP, Stage.TURN, Stage.RIVER]
        n_stages = len(stage_labels)
        x = np.arange(n_stages)
        for idx, name in enumerate(agent_names):
            counts = all_actions.get(name, {})
            agg_rates = []
            for sk in stage_keys:
                stage_counts = counts.get('stage_%d' % sk, {})
                total = sum(stage_counts.values()) or 1
                agg = (stage_counts.get(Action.RAISE_HALF_POT, 0) +
                       stage_counts.get(Action.RAISE_POT, 0)) / total
                agg_rates.append(agg)
            offset = (idx - n_agents / 2) * 0.15
            ax.bar(x + offset, agg_rates, 0.13, label=name,
                   color=_CPalette.get(name, '#95a5a6'), alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylabel('Aggression Rate')
        ax.set_title('(c) Aggression Rate by Stage', fontweight='bold')
        ax.legend(fontsize=9)

        # ── (d) Heatmap: agent x stage aggression ──
        ax = axes[1][1]
        data = []
        for name in agent_names:
            counts = all_actions.get(name, {})
            row = []
            for sk in stage_keys:
                stage_counts = counts.get('stage_%d' % sk, {})
                total = sum(stage_counts.values()) or 1
                agg = (stage_counts.get(Action.RAISE_HALF_POT, 0) +
                       stage_counts.get(Action.RAISE_POT, 0)) / total
                row.append(agg)
            data.append(row)
        data = np.array(data)
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
        ax.set_xticks(range(n_stages))
        ax.set_yticks(range(n_agents))
        ax.set_xticklabels(stage_labels)
        ax.set_yticklabels(agent_names)
        for i in range(n_agents):
            for j in range(n_stages):
                ax.text(j, i, '%.2f' % data[i][j], ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if data[i][j] > 0.35 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Aggression Rate')
        ax.set_title('(d) Aggression Heatmap', fontweight='bold')

        fig.suptitle('Action Distribution Analysis', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '3_action_distribution.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 4. Training Curves ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_training_curves(self, rewards_history, win_rates_history=None,
                              epsilon_history=None, qtable_sizes=None,
                              eval_intervals=None, save_path=None):
        n_plots = 2 + (1 if epsilon_history else 0) + (1 if qtable_sizes else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        idx = 0

        # ── Reward curve ──
        ax = axes[idx]; idx += 1
        rewards = np.array(rewards_history)
        if len(rewards) > 50:
            window = min(200, len(rewards) // 5)
            smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(smooth, color='#3498db', linewidth=1.5, label='Smoothed (w=%d)' % window)
        ax.plot(rewards, alpha=0.15, color='#3498db', linewidth=0.5, label='Raw')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (normalized)')
        ax.set_title('Training Reward', fontweight='bold')
        ax.legend(fontsize=9)

        # ── Win rate curve ──
        ax = axes[idx]; idx += 1
        if win_rates_history:
            if eval_intervals:
                x = np.array(eval_intervals)
            else:
                x = np.arange(len(win_rates_history))
            ax.plot(x, np.array(win_rates_history) * 100,
                    'o-', color='#2ecc71', linewidth=2, markersize=4)
            ax.fill_between(x, 0, np.array(win_rates_history) * 100,
                            alpha=0.1, color='#2ecc71')
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate Convergence', fontweight='bold')
        ax.legend(fontsize=9)

        # ── Epsilon decay ──
        if epsilon_history and idx < n_plots:
            ax = axes[idx]; idx += 1
            ax.plot(epsilon_history, color='#e74c3c', linewidth=1.5)
            ax.fill_between(range(len(epsilon_history)), 0, epsilon_history,
                            alpha=0.1, color='#e74c3c')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.set_title('Epsilon Decay', fontweight='bold')

        # ── Q-table size ──
        if qtable_sizes and idx < n_plots:
            ax = axes[idx]; idx += 1
            ax.plot(qtable_sizes, color='#f39c12', linewidth=1.5)
            ax.fill_between(range(len(qtable_sizes)), 0, qtable_sizes,
                            alpha=0.1, color='#f39c12')
            ax.set_xlabel('Episode')
            ax.set_ylabel('States Visited')
            ax.set_title('Q-Table Growth', fontweight='bold')

        fig.suptitle('Q-Learning Training Diagnostics', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '4_training_curves.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 5. Hybrid Opponent Curve ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_hybrid_opponent_curve(self, hybrid_results, agent_names, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        gto_ratios = sorted(hybrid_results.keys())
        for name in agent_names:
            win_rates = [hybrid_results[r].get(name, {}).get('win_rate', 0) * 100
                         for r in gto_ratios]
            payoffs = [hybrid_results[r].get(name, {}).get('avg_reward', 0)
                       for r in gto_ratios]
            color = _CPalette.get(name, '#95a5a6')
            ax1.plot(gto_ratios, win_rates, 'o-', color=color, linewidth=2,
                     markersize=5, label=name)
            ax2.plot(gto_ratios, payoffs, 'o-', color=color, linewidth=2,
                     markersize=5, label=name)

        ax1.set_xlabel('Opponent GTO Ratio')
        ax1.set_ylabel('Our Win Rate (%)')
        ax1.set_title('Win Rate vs Hybrid Opponent', fontweight='bold')
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)
        ax1.set_xticks(gto_ratios)
        ax1.set_xticklabels(['%d%%' % int(r * 100) for r in gto_ratios])

        ax2.set_xlabel('Opponent GTO Ratio')
        ax2.set_ylabel('Our Avg Payoff (chips)')
        ax2.set_title('Payoff vs Hybrid Opponent', fontweight='bold')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)
        ax2.set_xticks(gto_ratios)
        ax2.set_xticklabels(['%d%%' % int(r * 100) for r in gto_ratios])

        fig.suptitle('Performance vs Realistic (Hybrid) Opponents\n(GTO ratio = how rational the opponent is)',
                      fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '5_hybrid_opponent_curve.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 6. Rolling Win Rate ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_rolling_win_rate(self, rolling_data, agent_names, window=100, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 5))

        for name in agent_names:
            raw = rolling_data.get(name, [])
            if not raw:
                continue
            rates = []
            for i in range(len(raw)):
                start = max(0, i - window + 1)
                segment = raw[start:i + 1]
                rates.append(sum(segment) / len(segment) * 100)
            color = _CPalette.get(name, '#95a5a6')
            ax.plot(range(len(rates)), rates, linewidth=1.2, alpha=0.9,
                    color=color, label=name)

        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Rolling Win Rate (%s-game window)' % window)
        ax.set_title('Strategy Stability: Rolling Win Rate', fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '6_rolling_win_rate.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 7. Opponent Modeling Impact ━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_opponent_modeling(self, modeling_results, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # ── (a) Bar chart: adjusted vs unadjusted ──
        ax = axes[0]
        opponents = list(modeling_results.get('adjusted_vs_unadjusted', {}).keys())
        if opponents:
            x = np.arange(len(opponents))
            width = 0.35
            unadj = [modeling_results['adjusted_vs_unadjusted'][o].get('unadjusted', 0) * 100
                     for o in opponents]
            adj = [modeling_results['adjusted_vs_unadjusted'][o].get('adjusted', 0) * 100
                   for o in opponents]
            ax.bar(x - width / 2, unadj, width, label='Without Modeling',
                   color='#bdc3c7', edgecolor='white')
            ax.bar(x + width / 2, adj, width, label='With Modeling',
                   color='#3498db', edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(opponents, fontsize=9)
            ax.set_ylabel('Win Rate (%)')
            ax.set_title('(a) Opponent Modeling Impact', fontweight='bold')
            ax.legend()

        # ── (b) Player type confusion matrix ──
        ax = axes[1]
        type_results = modeling_results.get('player_type_analysis', {})
        if type_results:
            types = list(type_results.keys())
            metrics = ['Win Rate', 'Fold Exploit', 'Bluff Catch']
            data = []
            for t in types:
                d = type_results[t]
                data.append([
                    d.get('win_rate', 0) * 100,
                    d.get('fold_exploit', 0) * 100,
                    d.get('bluff_catch', 0) * 100,
                ])
            data = np.array(data)
            im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax.set_xticks(range(len(types)))
            ax.set_yticks(range(len(metrics)))
            ax.set_xticklabels(types, fontsize=9, rotation=30, ha='right')
            ax.set_yticklabels(metrics)
            for i in range(len(metrics)):
                for j in range(len(types)):
                    ax.text(j, i, '%.1f' % data[j][i], ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if data[j][i] < 20 or data[j][i] > 80 else 'black')
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title('(b) Strategy vs Player Type', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('(b) Strategy vs Player Type', fontweight='bold')

        fig.suptitle('Opponent Modeling & Exploitation Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '7_opponent_modeling.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 8. Game Theory Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_game_theory_analysis(self, tournament_results, agent_names, save_path=None):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        n = len(agent_names)

        # ── (a) Nash Equilibrium Proximity ──
        ax1 = fig.add_subplot(gs[0, 0])
        nash_scores = {}
        for name in agent_names:
            min_win = 100
            for other in agent_names:
                if other == name:
                    continue
                key = '%s vs %s' % (name, other)
                if key in tournament_results:
                    min_win = min(min_win, tournament_results[key]['win_rate_a'] * 100)
            nash_scores[name] = min_win

        sorted_names = sorted(agent_names, key=lambda x: nash_scores.get(x, 0), reverse=True)
        colors = [_CPalette.get(n, '#95a5a6') for n in sorted_names]
        bars = ax1.barh(range(n), [nash_scores.get(n, 0) for n in sorted_names],
                        color=colors, edgecolor='white', height=0.6)
        ax1.set_yticks(range(n))
        ax1.set_yticklabels(sorted_names)
        ax1.set_xlabel('Min Win Rate across Opponents (%)')
        ax1.set_title('(a) Nash Equilibrium Proximity\n(higher = harder to exploit)',
                       fontweight='bold', fontsize=11)
        for bar, val in zip(bars, [nash_scores.get(n, 0) for n in sorted_names]):
            ax1.text(val + 1, bar.get_y() + bar.get_height() / 2,
                     '%.1f%%' % val, va='center', fontsize=10, fontweight='bold')

        # ── (b) Best Response & Regret ──
        ax2 = fig.add_subplot(gs[0, 1])
        regrets = {}
        best_responses = {}
        for name in agent_names:
            max_loss = 0
            best_opp = ''
            for other in agent_names:
                if other == name:
                    continue
                key = '%s vs %s' % (name, other)
                if key in tournament_results:
                    win = tournament_results[key]['win_rate_a'] * 100
                    loss = 50 - win  # regret from 50% baseline
                    if loss > max_loss:
                        max_loss = loss
                        best_opp = other
            regrets[name] = max_loss
            best_responses[name] = best_opp

        colors_regret = [_CPalette.get(n, '#95a5a6') for n in agent_names]
        bars = ax2.bar(range(n), [regrets.get(n, 0) for n in agent_names],
                       color=colors_regret, edgecolor='white', width=0.6)
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(agent_names, fontsize=9)
        ax2.set_ylabel('Max Regret (pp from 50%)')
        ax2.set_title('(b) Exploitability (Max Regret)\n(lower = more robust)',
                       fontweight='bold', fontsize=11)
        for bar, name in zip(bars, agent_names):
            val = regrets.get(name, 0)
            opp = best_responses.get(name, '')
            color = 'white' if val > 15 else 'black'
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() / 2,
                     'vs %s' % opp, ha='center', va='center',
                     fontsize=8, fontweight='bold', color=color, rotation=90)

        # ── (c) Rock-Paper-Scissors Flow ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        advantage_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = '%s vs %s' % (agent_names[i], agent_names[j])
                if key in tournament_results:
                    advantage_matrix[i][j] = tournament_results[key]['win_rate_a'] * 100 - 50

        angles_pos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy = 0.5, 0.5
        radius = 0.3
        positions = {}
        for i, name in enumerate(agent_names):
            px = cx + radius * np.cos(angles_pos[i] - np.pi / 2)
            py = cy + radius * np.sin(angles_pos[i] - np.pi / 2)
            positions[name] = (px, py)
            color = _CPalette.get(name, '#95a5a6')
            circle = plt.Circle((px, py), 0.08, color=color, alpha=0.7, transform=ax3.transAxes)
            ax3.add_patch(circle)
            ax3.text(px, py, name, ha='center', va='center',
                     fontsize=7, fontweight='bold', transform=ax3.transAxes)

        for i in range(n):
            for j in range(i + 1, n):
                adv = advantage_matrix[i][j]
                if abs(adv) < 1:
                    continue
                winner = agent_names[i] if adv > 0 else agent_names[j]
                loser = agent_names[j] if adv > 0 else agent_names[i]
                x1, y1 = positions[winner]
                x2, y2 = positions[loser]
                lw = min(abs(adv) / 5, 3)
                ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                             xycoords='axes fraction', textcoords='axes fraction',
                             arrowprops=dict(arrowstyle='->', color=_CPalette.get(winner, '#333'),
                                            lw=lw, alpha=0.6))
        ax3.set_title('(c) Advantage Flow Graph\n(arrow = beats)',
                       fontweight='bold', fontsize=11)

        # ── (d) Dominance Ladder ──
        ax4 = fig.add_subplot(gs[1, 1])
        avg_wins = {}
        for name in agent_names:
            total_win = 0
            count = 0
            for other in agent_names:
                if other == name:
                    continue
                key = '%s vs %s' % (name, other)
                if key in tournament_results:
                    total_win += tournament_results[key]['win_rate_a'] * 100
                    count += 1
            avg_wins[name] = total_win / count if count > 0 else 0

        ranked = sorted(agent_names, key=lambda x: avg_wins.get(x, 0), reverse=True)
        for rank, name in enumerate(ranked):
            avg = avg_wins.get(name, 0)
            color = _CPalette.get(name, '#95a5a6')
            y_pos = (n - 1 - rank) * 0.15 + 0.1
            rect = FancyBboxPatch((0.05, y_pos), avg / 100 * 0.85, 0.1,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.8,
                                   transform=ax4.transAxes)
            ax4.add_patch(rect)
            ax4.text(0.03, y_pos + 0.05, '#%d' % (rank + 1),
                     ha='right', va='center', fontsize=10, fontweight='bold',
                     transform=ax4.transAxes)
            ax4.text(0.08, y_pos + 0.05, '%s (%.1f%%)' % (name, avg),
                     ha='left', va='center', fontsize=10, fontweight='bold',
                     color='white', transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('(d) Overall Dominance Ranking\n(avg win rate across all opponents)',
                       fontweight='bold', fontsize=11)

        fig.suptitle('Game Theory & Strategic Analysis', fontsize=16, fontweight='bold', y=1.02)
        path = save_path or os.path.join(self.output_dir, '8_game_theory_analysis.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 9. Chip EV Distribution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_chip_ev_distribution(self, chip_ev_data, agent_names, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax_hist = axes[0][0]
        for name in agent_names:
            evs = chip_ev_data.get(name, [])
            if evs:
                ax_hist.hist(evs, bins=60, alpha=0.5, label=name,
                             color=_CPalette.get(name, '#95a5a6'), density=True)
        ax_hist.set_xlabel('Chip EV')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('(a) Payoff Distribution', fontweight='bold')
        ax_hist.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_hist.legend(fontsize=9)

        ax_box = axes[0][1]
        box_data = []
        box_labels = []
        for name in agent_names:
            evs = chip_ev_data.get(name, [])
            if evs:
                box_data.append(evs)
                box_labels.append(name)
        if box_data:
            bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)
            for patch, name in zip(bp['boxes'], box_labels):
                patch.set_facecolor(_CPalette.get(name, '#95a5a6'))
                patch.set_alpha(0.6)
        ax_box.set_ylabel('Chip EV')
        ax_box.set_title('(b) Payoff Box Plot', fontweight='bold')
        ax_box.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax_cum = axes[1][0]
        for name in agent_names:
            evs = chip_ev_data.get(name, [])
            if evs:
                sorted_ev = np.sort(evs)
                cumprob = np.arange(1, len(sorted_ev) + 1) / len(sorted_ev)
                ax_cum.plot(sorted_ev, cumprob, linewidth=2,
                            color=_CPalette.get(name, '#95a5a6'), label=name)
        ax_cum.set_xlabel('Chip EV')
        ax_cum.set_ylabel('Cumulative Probability')
        ax_cum.set_title('(c) Cumulative Distribution (CDF)', fontweight='bold')
        ax_cum.legend(fontsize=9)
        ax_cum.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax_stats = axes[1][1]
        ax_stats.axis('off')
        rows = [['Agent', 'Mean', 'Std', 'Median', 'Win%', 'Best', 'Worst']]
        for name in agent_names:
            evs = chip_ev_data.get(name, [])
            if evs:
                evs = np.array(evs)
                rows.append([
                    name,
                    '%.1f' % np.mean(evs),
                    '%.1f' % np.std(evs),
                    '%.1f' % np.median(evs),
                    '%.1f%%' % (100 * np.mean(evs > 0)),
                    '%.0f' % np.max(evs),
                    '%.0f' % np.min(evs),
                ])
        table = ax_stats.table(cellText=rows[1:], colLabels=rows[0],
                                loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#ecf0f1' if key[0] % 2 == 0 else 'white')
        ax_stats.set_title('(d) Payoff Statistics', fontweight='bold')

        fig.suptitle('Chip EV Distribution & Risk Analysis', fontsize=16,
                      fontweight='bold', y=1.02)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '9_chip_ev_distribution.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path

    # ━━━ 10. Head-to-Head Detail ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def plot_head_to_head(self, h2h_results, save_path=None):
        matchups = list(h2h_results.keys())
        n_matchups = len(matchups)
        if n_matchups == 0:
            return None

        fig, axes = plt.subplots(n_matchups, 3, figsize=(18, 4 * n_matchups))
        if n_matchups == 1:
            axes = axes.reshape(1, -1)

        for row, matchup in enumerate(matchups):
            r = h2h_results[matchup]

            # Win rate bar
            ax = axes[row][0]
            ax.barh([0, 1], [r.get('win_rate_a', 0) * 100, r.get('win_rate_b', 0) * 100],
                    color=['#3498db', '#e74c3c'], edgecolor='white', height=0.5)
            ax.set_yticks([0, 1])
            names = matchup.split(' vs ')
            ax.set_yticklabels(names[:2] if len(names) >= 2 else ['A', 'B'])
            ax.set_xlabel('Win Rate (%)')
            ax.set_xlim(0, 100)
            ax.set_title(matchup, fontweight='bold', fontsize=11)

            # Action distribution
            ax = axes[row][1]
            action_labels = ['Fold', 'Call', 'RHalf', 'RPot']
            action_keys = [Action.FOLD, Action.CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
            for side, prefix, color, offset in [('a', 'A', '#3498db', -0.15),
                                                  ('b', 'B', '#e74c3c', 0.15)]:
                acts = r.get('actions_%s' % side, {})
                total = sum(acts.values()) or 1
                rates = [acts.get(k, 0) / total for k in action_keys]
                ax.bar(np.arange(4) + offset, rates, 0.25,
                       color=color, alpha=0.7, label=prefix, edgecolor='white')
            ax.set_xticks(range(4))
            ax.set_xticklabels(action_labels, fontsize=8)
            ax.set_ylabel('Frequency')
            ax.set_title('Actions', fontsize=10)
            ax.legend(fontsize=8)

            # Key stats
            ax = axes[row][2]
            ax.axis('off')
            stats_text = (
                "Win Rate: %.1f%% vs %.1f%%\n"
                "Avg Payoff: %.1f vs %.1f\n"
                "Entropy: %.3f vs %.3f\n"
                "Avg Pot: %.0f\n"
                "Showdown: %.1f%%\n"
                "Avg Steps: %.1f"
            ) % (
                r.get('win_rate_a', 0) * 100, r.get('win_rate_b', 0) * 100,
                r.get('avg_reward_a', 0), r.get('avg_reward_b', 0),
                r.get('action_entropy_a', 0), r.get('action_entropy_b', 0),
                r.get('avg_pot', 0),
                r.get('showdown_rate', 0) * 100,
                r.get('avg_game_length', 0),
            )
            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
            ax.set_title('Statistics', fontsize=10)

        fig.suptitle('Head-to-Head Detailed Comparison', fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, '10_head_to_head.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print('  [saved] %s' % path)
        return path
