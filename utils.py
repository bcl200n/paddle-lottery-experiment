import matplotlib.pyplot as plt
import numpy as np
import settings

def sample(preds, temperature=1.0):
    """
    从概率分布中采样
    :param preds: 概率分布数组
    :param temperature: 控制采样分布的温度，越低越倾向于选择最大概率
    :return: 从分布中随机选择的一个号码
    """
    preds = np.asarray(preds).astype(np.float64)
    preds += 1e-9  # 防止log(0)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)  # 确保概率和为1
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def search_award(front_match_num, back_match_num, cache={}):
    if front_match_num == 0 and back_match_num == 0:
        return 0
    award = cache.get((front_match_num, back_match_num), -1)
    if award != -1:
        return award
    award = settings.AWARD_RULES.get((front_match_num, back_match_num), -1)
    if award == -1:
        award = 0
        if front_match_num > 0:
            award = search_award(front_match_num - 1, back_match_num)
        if back_match_num > 0:
            award = max(award, search_award(front_match_num, back_match_num - 1))
    cache[(front_match_num, back_match_num)] = award
    return award


def lotto_calculate(winning_sequence, sequence_selected):
    front_match = len(
        set(winning_sequence[:settings.FRONT_SIZE]).intersection(set(sequence_selected[:settings.FRONT_SIZE])))

    back_match = len(
        set(winning_sequence[settings.FRONT_SIZE:]).intersection(set(sequence_selected[settings.FRONT_SIZE:])))
    award = search_award(front_match, back_match)
    return award


def select_seqs(predicts):
    """
    根据给定的概率分布，随机选择一种彩票序列。
    :param predicts: list[list] 7个球的概率分布组成的列表
    :return: list 包含7个号码的彩票序列
    """
    front_balls = []
    back_balls = []
    
    # 分割前区和后区的概率分布
    front_predicts = predicts[:settings.FRONT_SIZE]
    back_predicts = predicts[settings.FRONT_SIZE:]

    # 1. 选择前区号码 (彼此不重复)
    for predict in front_predicts:
        try_cnt = 0
        while True:
            try_cnt += 1
            ball = sample(predict)
            if ball not in front_balls:
                front_balls.append(ball)
                break
            if try_cnt > 100:
                # 如果超过100次尝试依然没有找到不重复的数字，则重新计算概率并选择
                ball = sample([1. / len(predict) for _ in predict])
                front_balls.append(ball)
                break
    
    # 2. 选择后区号码 (彼此不重复)
    for predict in back_predicts:
        try_cnt = 0
        while True:
            try_cnt += 1
            ball = sample(predict)
            if ball not in back_balls:
                back_balls.append(ball)
                break
            if try_cnt > 100:
                # 如果超过100次尝试依然没有找到不重复的数字，则重新计算概率并选择
                ball = sample([1. / len(predict) for _ in predict])
                back_balls.append(ball)
                break

    # 3. 组合并排序
    final_balls = sorted(front_balls) + sorted(back_balls)
    return final_balls


def draw_graph(y):
    x = list(range(len(y)))
    parameter = np.polyfit(x, y, 1)
    f = np.poly1d(parameter)
    plt.plot(x, f(x), "r--")
    plt.plot(y)
    # 修改为保存图像，而不是显示
    graph_path = 'trend_graph.png'
    plt.savefig(graph_path)
    print(f"Graph saved to {graph_path}")
