# draw_dbb_training_vs_inference.py
from graphviz import Digraph

dot = Digraph(name='DBB_Training_vs_Inference', format='png',
              graph_attr={'rankdir': 'LR', 'splines': 'ortho'},
              node_attr={'shape': 'box', 'style': 'filled,rounded', 'fontname': 'Microsoft YaHei'})

# ---------- 左边：Training（多分支相加） ----------
dot.node('in_tr', 'Input\n(Training)', fillcolor='lightgray')
branches = ['origin', 'avg', '1x1', '1x1_kxk']
for b in branches:
    dot.node(b, f'Conv+BN\n({b})', fillcolor='lightgreen')
dot.node('add', '逐元素\n相加', shape='circle', fillcolor='lightgreen')
dot.node('silu_tr', 'SiLU\n(Training)', fillcolor='lightgreen')
dot.node('out_tr', 'Output\n(Training)', fillcolor='lightgreen')

for b in branches:
    dot.edge('in_tr', b, color='green')
dot.edge('add', 'silu_tr')
dot.edge('silu_tr', 'out_tr')

# ---------- 右边：Inference（融合后单卷积） ----------
dot.node('in_inf', 'Input\n(Inference)', fillcolor='lightgray')
dot.node('fused', '融合权重\nConv2D 3×3\n(dbb_reparam)', fillcolor='lightblue')
dot.node('silu_inf', 'SiLU\n(Inference)', fillcolor='lightblue')
dot.node('out_inf', 'Output\n(Inference)', fillcolor='lightgray')

dot.edge('in_inf', 'fused')
dot.edge('fused', 'silu_inf')
dot.edge('silu_inf', 'out_inf')

# ---------- 一次性图注 ----------
dot.node('note', '左边：训练阶段多分支相加；右边：推理阶段单卷积融合（无 BN）',
         shape='none', fontcolor='red', fontsize='11')

dot.render('dbb_training_vs_inference', view=True)   # 生成并打开 dbf_training_vs_inference.png