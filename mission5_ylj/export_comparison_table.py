"""Export an editable text-based SVG table and PNG preview from verified data."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent


def main():
    records=json.loads((BASE/"comparison/comparison.json").read_text(encoding="utf-8"))
    def acc(model,size,shot,lam=None):
        r=next(r for r in records if r["model"]==model and r["relation_pairs"]==size and r["shots"]==shot and (lam is None or r["lambda"]==lam))
        return f"{r['digit_mean_percent']:.3f} ± {r['digit_std_percentage_points']:.3f}"
    rows=[['全监督（100轮）','不使用','全部50,000张',acc('Supervised','none','all')],
          ['基础弱监督','50k','无绝对数字标签',acc('Weakly supervised','50k',0)],
          ['基础弱监督','500k','无绝对数字标签',acc('Weakly supervised','500k',0)]]
    for shot in [1,3]:
        rows.append(['Siamese','50k',f'{shot}-shot（共{shot*10}张）',acc('Siamese','50k',shot)])
    for size in ['50k','500k']:
        for shot in [1,3]:
            rows.append(['联合监督（λ=0.5）',size,f'{shot}-shot（共{shot*10}张）',acc('Joint',size,shot,.5)])
    values=[['模型','关系图片对','绝对数字标签','数字测试准确率（%）']]+rows
    widths=[285,170,325,360]
    xs=[30]
    for w in widths: xs.append(xs[-1]+w)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">',
           '<rect width="1200" height="800" fill="white"/>',
           '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif" fill="#344C7B">',
           '<text x="30" y="48" font-size="30" font-weight="bold">四种监督方案的数字分类结果</text>',
           '<text x="30" y="84" font-size="19">种子42–46的均值 ± 样本标准差；前三行不适用1-shot / 3-shot</text>']
    for i,row in enumerate(values):
        y=110+i*51
        parts.append(f'<rect x="30" y="{y}" width="1140" height="51" fill="{"#DBEAFE" if i==0 else "white"}" stroke="#CBD1D7"/>')
        for j,label in enumerate(row):
            x=xs[j]+14 if j==0 else (xs[j]+xs[j+1])/2
            anchor='start' if j==0 else 'middle'
            parts.append(f'<text x="{x}" y="{y+32}" font-size="22" text-anchor="{anchor}" font-weight="{"bold" if i==0 else "normal"}">{escape(label)}</text>')
    for x in xs[1:-1]: parts.append(f'<line x1="{x}" y1="110" x2="{x}" y2="{110+51*len(values)}" stroke="#CBD1D7"/>')
    parts.extend(['<text x="30" y="714" font-size="18">1-shot / 3-shot 指每类1张 / 3张，共10张 / 30张绝对标签图片。</text>',
                  '<text x="30" y="744" font-size="18">基础弱监督仍使用大小关系标签。“无绝对数字标签”不代表无监督。</text>',
                  '<text x="30" y="774" font-size="18">训练预算不同：全监督100轮；弱/联合5轮；Siamese预训练5轮、分类头100轮。</text>', '</g></svg>'])
    out=BASE/'comparison'
    (out/'comparison_table_editable.svg').write_text('\n'.join(parts),encoding='utf-8')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':13})
    fig,ax=plt.subplots(figsize=(12,8))
    ax.axis('off')
    fig.text(.03,.95,'四种监督方案的数字分类结果',fontsize=21,weight='bold')
    fig.text(.03,.91,'种子42–46的均值 ± 样本标准差；前三行不适用1-shot / 3-shot',fontsize=12)
    t=ax.table(cellText=rows,colLabels=values[0],colWidths=[.25,.149,.285,.316],cellLoc='center',bbox=[0,.11,1,.80])
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#CBD1D7')
        cell.set_linewidth(.6)
        if i==0: cell.set_facecolor('#F1F3F4'); cell.get_text().set_weight('bold')
    fig.text(.03,.095,'1-shot / 3-shot：每类1张 / 3张，共10张 / 30张绝对标签图片。',fontsize=11)
    fig.text(.03,.06,'基础弱监督使用大小关系标签。全监督100轮，弱/联合5轮，Siamese为5轮预训练加100轮分类头训练。',fontsize=10)
    fig.subplots_adjust(left=.03,right=.98,top=.93,bottom=.08)
    fig.savefig(out/'comparison_table_preview.png',dpi=180,facecolor='white')
    plt.close(fig)
    print(out/'comparison_table_editable.svg')


if __name__=='__main__':
    main()
