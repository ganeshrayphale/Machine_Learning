"""
Linearity Check: Diabetes Dataset Features vs Target Variable
Checks whether each feature has a linear relationship with the target.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_diabetes
from scipy import stats

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names  # age, sex, bmi, bp, s1..s6

# ─────────────────────────────────────────────────────────
# 1.  SCATTER PLOTS  (Feature vs Target)
# ─────────────────────────────────────────────────────────
n_features = X.shape[1]
n_cols = 5
n_rows = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9))
fig.patch.set_facecolor('#0f0f1a')
fig.suptitle(
    'Scatter Plots: Each Feature vs Target (Disease Progression)',
    color='white', fontsize=16, fontweight='bold', y=1.01
)

axes = axes.flatten()
pearson_results = {}

for i, (feat, ax) in enumerate(zip(feature_names, axes)):
    x_col = X[:, i]

    # Pearson r and p-value
    r, p = stats.pearsonr(x_col, y)
    pearson_results[feat] = (r, p)

    # OLS trendline
    slope, intercept, *_ = stats.linregress(x_col, y)
    x_line = np.linspace(x_col.min(), x_col.max(), 200)
    y_line = slope * x_line + intercept

    # Color dots by distance from trendline (residual magnitude)
    residuals = y - (slope * x_col + intercept)
    norm = plt.Normalize(residuals.min(), residuals.max())
    sc = ax.scatter(x_col, y, c=residuals, cmap='coolwarm',
                    alpha=0.55, s=18, edgecolors='none')
    ax.plot(x_line, y_line, color='#facc15', lw=1.8, label='OLS fit')

    # Significance star
    star = '★' if p < 0.05 else '✗'
    color_r = '#22d3ee' if abs(r) > 0.3 else '#f87171'
    ax.set_title(f'{feat}  r={r:.2f} {star}', color=color_r,
                 fontsize=10, fontweight='bold')
    ax.set_xlabel(feat, color='#94a3b8', fontsize=8)
    ax.set_ylabel('Target', color='#94a3b8', fontsize=8)
    ax.set_facecolor('#1e1e2e')
    ax.tick_params(colors='#64748b', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

plt.tight_layout()
plt.savefig('linearity_scatter.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.show()
print("Saved: linearity_scatter.png")

# ─────────────────────────────────────────────────────────
# 2.  PEARSON CORRELATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print(f"{'Feature':<8}  {'Pearson r':>10}  {'p-value':>12}  {'Linear?':>8}")
print("═" * 55)
for feat, (r, p) in sorted(pearson_results.items(), key=lambda x: -abs(x[1][0])):
    verdict = '✅ YES' if p < 0.05 else '❌ NO '
    print(f"{feat:<8}  {r:>10.4f}  {p:>12.2e}  {verdict}")
print("═" * 55)
print("★ p < 0.05  →  statistically significant linear relationship")

# ─────────────────────────────────────────────────────────
# 3.  RANKED BAR CHART  (|r| values)
# ─────────────────────────────────────────────────────────
sorted_feats = sorted(pearson_results.items(), key=lambda x: abs(x[1][0]), reverse=True)
names_sorted = [f for f, _ in sorted_feats]
r_vals = [abs(v[0]) for _, v in sorted_feats]
p_vals = [v[1] for _, v in sorted_feats]

bar_colors = ['#22d3ee' if p < 0.05 else '#f87171' for p in p_vals]

fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.patch.set_facecolor('#0f0f1a')
ax2.set_facecolor('#1e1e2e')

bars = ax2.barh(names_sorted[::-1], r_vals[::-1], color=bar_colors[::-1],
                edgecolor='none', height=0.55)

# Value labels
for bar, r in zip(bars, r_vals[::-1]):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{r:.3f}', va='center', ha='left', color='white', fontsize=9)

ax2.axvline(0.3, color='#facc15', lw=1.2, linestyle='--', alpha=0.7,
            label='Threshold |r|=0.3')
ax2.set_xlim(0, 0.65)
ax2.set_xlabel('|Pearson r|  (absolute correlation)', color='#94a3b8')
ax2.set_title('Feature Linearity Strength w.r.t. Target\n'
              '(Cyan = significant  │  Red = not significant)',
              color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='#94a3b8')
ax2.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=9)
for spine in ax2.spines.values():
    spine.set_edgecolor('#334155')

plt.tight_layout()
plt.savefig('linearity_bar.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.show()
print("Saved: linearity_bar.png")
