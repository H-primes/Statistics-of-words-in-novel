import jieba
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
# 显示汉字
from pylab import *
from pypinyin import lazy_pinyin, Style
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

plt.rcParams['font.sans-serif'] = ['SimHei']
# print(int(2.9))
# print(lazy_pinyin('真的吗', style=Style.FIRST_LETTER))

file = open('水浒传.txt', 'r', encoding='UTF-8')
txt = file.read()

# 提取汉字
passage = ''.join(re.findall('[\u4e00-\u9fa5]', txt))

# 汉字计数
countWord = {}
for i in passage:
    if i in countWord:
        continue
    else:
        countWord[i] = passage.count(i)
countWord = sorted(countWord.items(), key=lambda d: d[1], reverse=True)
print(countWord)
print(len(countWord))
print(len(passage))
wordDict = {}
tot = 0
for i in countWord:
    wordDict[i[0]] = i[1]
for i in wordDict.items():
    tot += i[1]

names = ['宋江', '卢俊义', '吴用', '公孙胜', '关胜', '林冲', '秦明', '呼延灼', '花荣', '柴进', '李应', '朱仝', '鲁智深', '武松', '董平', '张清', '杨志', '徐宁',
         '索超', '戴宗', '刘唐', '李逵', '史进', '穆弘', '雷横', '李俊', '阮小二', '张横', '阮小五', '张顺', '阮小七', '杨雄', '石秀', '解珍', '解宝', '燕青',
         '朱武', '黄信', '孙立', '宣赞', '郝思文', '韩滔', '彭', '单廷', '魏定国', '萧让', '裴宣', '欧鹏', '邓飞', '燕顺', '杨林', '凌振', '蒋敬', '吕方',
         '郭盛', '安道全', '皇甫端', '王英', '扈三娘', '鲍旭', '樊瑞', '孔明', '孔亮', '项充', '李衮', '金大坚', '马麟', '童威', '童猛', '孟康', '侯健', '陈达',
         '杨春', '郑天寿', '陶宗旺', '宋清', '乐和', '龚旺', '丁得孙', '穆春', '曹正', '宋万', '杜迁', '薛永', '施恩', '李忠', '周通', '汤隆', '杜兴', '邹渊',
         '邹润', '朱贵', '朱富', '蔡福', '蔡庆', '李立', '李云', '焦挺', '石勇', '孙新', '顾大嫂', '张青', '孙二娘', '王定六', '郁保四', '白胜', '时迁',
         '段景住']
names_count = [(v, passage.count(v)) for v in names]

nicknames = ['呼保义', '玉麒麟', '智多星', '入云龙', '大刀', '豹子头', '霹雳火', '双鞭', '小李广', '小旋风', '扑天雕', '美髯公', '花和尚',
             '行者', '双枪将', '没羽箭', '青面兽', '金枪手', '急先锋', '神行太保', '赤发鬼', '黑旋风', '九纹龙', '没遮拦', '插翅虎', '混江龙',
             '立地太岁', '船火儿', '短命二郎', '浪里白条', '活阎罗', '病关索', '拼命三郎', '两头蛇', '双尾蝎', '浪子', '神机军师', '镇三山', '病尉迟',
             '丑郡马', '井木犴', '百胜将', '天目将', '圣水将', '神火将', '圣手书生', '铁面孔目', '摩云金翅', '火眼狻猊', '锦毛虎', '锦豹子',
             '轰天雷', '神算子', '小温侯', '赛仁贵', '神医', '紫髯伯', '矮脚虎', '一丈青', '丧门神', '混世魔王', '毛头星', '独火星',
             '八臂哪吒', '飞天大圣', '玉臂匠', '铁笛仙', '出洞蛟', '翻江蜃', '玉幡竿', '通臂猿', '跳涧虎', '白花蛇', '白面郎君', '九尾龟',
             '铁扇子', '铁叫子', '花项虎', '中箭虎', '小遮拦', '操刀鬼', '云里金刚', '摸着天', '病大虫', '金眼彪', '打虎将', '小霸王',
             '金钱豹子', '鬼脸儿', '出林龙', '独角龙', '旱地忽律', '笑面虎', '铁臂膊', '一枝花', '催命判官', '青眼虎', '没面目', '石将军',
             '小尉迟', '母大虫', '菜园子', '母夜叉', '活闪婆', '险道神', '白日鼠', '鼓上蚤', '金毛犬']

nicknames_count = [(v, passage.count(v)) for v in nicknames]

starnames = ['天魁星', '天罡星', '天机星', '天闲星', '天勇星', '天雄星', '天猛星', '天威星', '天英星', '天贵星', '天富星', '天满星',
             '天孤星', '天伤星', '天立星', '天捷星', '天暗星', '天佑星', '天空星', '天速星', '天异星', '天杀星', '天微星', '天究星',
             '天退星', '天寿星', '天剑星', '天竟星', '天罪星', '天损星', '天败星', '天牢星', '天慧星', '天暴星', '天哭星', '天巧星',
             '地魁星', '地煞星', '地勇星', '地杰星', '地雄星', '地威星', '地英星', '地奇星', '地猛星', '地文星', '地正星', '地阔星',
             '地阖星', '地强星', '地暗星', '地轴星', '地会星', '地佐星', '地佑星', '地灵星', '地兽星', '地微星', '地慧星', '地暴星',
             '地然星', '地猖星', '地狂星', '地飞星', '地走星', '地巧星', '地明星', '地进星', '地退星', '地满星', '地遂星', '地周星',
             '地隐星', '地异星', '地理星', '地俊星', '地乐星', '地捷星', '地速星', '地镇星', '地嵇星', '地魔星', '地妖星', '地幽星',
             '地伏星', '地僻星', '地空星', '地孤星', '地全星', '地短星', '地角星', '地囚星', '地藏星', '地平星', '地损星', '地奴星',
             '地察星', '地恶星', '地丑星', '地数星', '地阴星', '地刑星', '地壮星', '地劣星', '地健星', '地耗星', '地贼星', '地狗星']
starnames_count = [(v, passage.count(v)) for v in starnames]

# 画人名柱状图----------------------------------------------------------------------------------------------------------

plt.figure(num='人名统计', figsize=(20, 12))

ax1 = plt.subplot(211)

x1 = [v[0] for v in names_count[:36]]
y1 = [[v[1] for v in names_count[:36]],
      [v[1] for v in nicknames_count[:36]],
      [v[1] for v in starnames_count[:36]]]

rects = ax1.bar(x1, y1[0], width=0.7, label='姓名')
ax1.bar(x1, y1[1], width=0.7, label='绰号', bottom=y1[0])
ax1.bar(x1, y1[2], width=0.7, label='星号', bottom=[y1[0][i] + y1[1][i] for i in np.arange(0, len(y1[0]))])

ax1.set_ylabel('次数')
ax1.set_title('三十六天罡星', fontsize=9, color='b')

# 旋转x轴标签
for xtick in ax1.get_xticklabels():
    xtick.set_rotation(50)

for i in np.arange(0, 36):
    rect = rects[i]
    x = rect.get_x()
    height = rect.get_height()
    height += starnames_count[i][1]
    height += nicknames_count[i][1]
    ax1.text(x, 1.01 * height, str(height))
# 显示网格
ax1.grid(True, linestyle='--')
ax1.legend(loc='upper center')

ax2 = plt.subplot(212)
x2 = [v[0] for v in names_count[36:]]
y2 = [[v[1] for v in names_count[36:]],
      [v[1] for v in nicknames_count[36:]],
      [v[1] for v in starnames_count[36:]]]
# print(len(y2[1]))
rects = ax2.bar(x2, y2[0], width=0.7, label='姓名')
ax2.bar(x2, y2[1], width=0.7, label='绰号', bottom=y2[0])
ax2.bar(x2, y2[2], width=0.7, label='星号', bottom=[y2[0][i] + y2[1][i] for i in np.arange(0, len(y2[0]))])

ax2.set_ylabel('次数')
ax2.set_title('七十二地煞星', fontsize=9, color='b')

# 旋转x轴标签
for xtick in ax2.get_xticklabels():
    xtick.set_rotation(50)
    xtick.set_fontsize(8)

for i in np.arange(0, 72):
    rect = rects[i]
    x = rect.get_x()
    height = rect.get_height()
    height += starnames_count[i + 36][1]
    height += nicknames_count[i + 36][1]
    ax2.text(x, 1.01 * height, str(height))
ax2.grid(True, linestyle='--')
ax2.legend(loc='upper center')

# 指定分辨率导出图片
plt.savefig('name_count.jpg', dpi=300, bbox_inches='tight')

# 字数统计---------------------------------------------------------------------------------------------------------------------------------

letters = [chr(i) for i in range(97, 123)]
scales = [str(v) + '000' if v != 0 else '0' for v in np.arange(0, 12)]

# print(scales)
index = 0

def turn_to_check():
    plt.clf()
    global index, tot
    index += 1
    if index > 29:
        index = 0

    if index == 0:
        show_pcolor()
        return

    head = 51 + 144 * (index - 2)
    tail = 51 + 144 * (index - 1)
    if index == 1:
        head = 0

    next_button = Button(plt.axes([0.52, 0.01, 0.03, 0.03]), '->', color='lightgreen', hovercolor='g')
    last_button = Button(plt.axes([0.48, 0.01, 0.03, 0.03]), '<-', color='lightgreen', hovercolor='g')
    next_button.on_clicked(next_fig)
    last_button.on_clicked(last_fig)

    x = [v[0] for v in countWord[head:tail]]
    y = [v[1] * 1000 / tot for v in countWord[head:tail]]
    length = tail - head + 1
    ax3 = plt.subplot(211)
    ax3.bar(x[:int(length / 2)], y[:int(length / 2)], width=0.7)

    ax4 = plt.subplot(212)
    ax4.bar(x[int(length / 2):], y[int(length / 2):], width=0.7)

    ax3.set_title('千分比')
    ser = TextBox(plt.axes([0.6, 0.01, 0.03, 0.03]), '查询', initial='飞')
    ser.on_submit(check)
    plt.show()

def check(text):
    global index
    for i in np.arange(0, len(countWord)):
        if countWord[i][0] == text:
            if i <= 51:
                index = 0
                turn_to_check()
                return
            else:
                index = int((i - 51) / 144 + 1)
                turn_to_check()
                return


def next_fig(event):
    plt.clf()
    global index, tot
    index += 1
    if index > 29:
        index = 0

    if index == 0:
        show_pcolor()
        return

    head = 51 + 144 * (index - 2)
    tail = 51 + 144 * (index - 1)
    if index == 1:
        head = 0

    next_button = Button(plt.axes([0.52, 0.01, 0.03, 0.03]), '->', color='lightgreen', hovercolor='g')
    last_button = Button(plt.axes([0.48, 0.01, 0.03, 0.03]), '<-', color='lightgreen', hovercolor='g')
    next_button.on_clicked(next_fig)
    last_button.on_clicked(last_fig)

    x = [v[0] for v in countWord[head:tail]]
    y = [v[1] * 1000 / tot for v in countWord[head:tail]]
    length = tail - head + 1
    ax3 = plt.subplot(211)
    ax3.bar(x[:int(length / 2)], y[:int(length / 2)], width=0.7)

    ax4 = plt.subplot(212)
    ax4.bar(x[int(length / 2):], y[int(length / 2):], width=0.7)

    ax3.set_title('千分比')
    ser = TextBox(plt.axes([0.6, 0.01, 0.03, 0.03]), '查询', initial='飞')
    ser.on_submit(check)
    plt.show()


def last_fig(event):
    plt.clf()
    global index
    index -= 1
    if index < 0:
        index = 29

    if index == 0:
        show_pcolor()
        return
    head = 51 + 144 * (index - 2)
    tail = 51 + 144 * (index - 1)
    if index == 1:
        head = 0
    next_button = Button(plt.axes([0.52, 0.01, 0.03, 0.03]), '->', color='lightgreen', hovercolor='g')
    last_button = Button(plt.axes([0.48, 0.01, 0.03, 0.03]), '<-', color='lightgreen', hovercolor='g')

    x = [v[0] for v in countWord[head:tail]]
    y = [v[1] * 1000 / tot for v in countWord[head:tail]]
    length = tail - head + 1
    ax3 = plt.subplot(211)
    ax3.bar(x[:int(length / 2)], y[:int(length / 2)], width=0.7)

    ax4 = plt.subplot(212)
    ax4.bar(x[int(length / 2):], y[int(length / 2):], width=0.7)

    ax3.set_title('千分比')
    next_button.on_clicked(next_fig)
    last_button.on_clicked(last_fig)
    ser = TextBox(plt.axes([0.6, 0.01, 0.03, 0.03]), '查询', initial='飞')
    ser.on_submit(check)
    plt.show()


plt.figure(num='汉字统计', figsize=(20, 12))

mat = {}
sarr = []
for v in scales:
    mat[v] = {}
    sarr.append([])
for i in np.arange(0, 12):
    for j in np.arange(0, 26):
        sarr[i].append(0)
# print(sarr[0][0])
for v in countWord:
    c = lazy_pinyin(v[0], style=Style.FIRST_LETTER)[0]
    sarr[int(v[1] / 1000)][ord(c) - ord('a')] += v[1] * 100 / tot
'''
    if c in mat[str(v[1]-(v[1] % 1000))].keys():
        mat[str(v[1]-(v[1] % 1000))][c] += v[1]*100/tot
    else:
        mat[str(v[1] - (v[1] % 1000))][c] = v[1]*100/tot
'''


# print(sarr)

def show_pcolor():
    global sarr, index, passage
    ax3 = plt.subplot(111)

    sarr = np.array(sarr)
    sns.heatmap(sarr, mask=None, ax=ax3, linewidths=0.05, annot=True)
    # sns.heatmap(sarr, mask=sarr <= 0, ax=ax3, linewidths=0.05, annot=True)
    ax3.set_xlabel('首拼')
    ax3.set_ylabel('频数')
    ax3.set_xticklabels(letters)
    ax3.set_yticklabels(scales, rotation=0)

    next_button = Button(plt.axes([0.52, 0.01, 0.03, 0.03]), '->', color='lightgreen', hovercolor='g')
    last_button = Button(plt.axes([0.48, 0.01, 0.03, 0.03]), '<-', color='lightgreen', hovercolor='g')

    next_button.on_clicked(next_fig)
    last_button.on_clicked(last_fig)
    ax3.set_title('全书共有4083种汉字，总计708493个汉字')

    ser = TextBox(plt.axes([0.6, 0.01, 0.03, 0.03]), '查询', initial='飞')
    ser.on_submit(check)
    plt.show()


show_pcolor()


