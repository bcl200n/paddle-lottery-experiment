# -*- coding: utf-8 -*-
import random
import math
import numpy as np
from collections import Counter, defaultdict

DATA_PATH = "lotto_clean.csv"

FRONT_MAX = 35
BACK_MAX = 12

N_SAMPLES = 30000
TOP_K = 10

random.seed(42)
np.random.seed(42)


def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                issue = parts[0]
                nums = [int(x) for x in parts[1:]]
                rows.append((issue, nums[:5], nums[5:]))

    # 文件一般是新 -> 旧，这里改成旧 -> 新
    rows = rows[::-1]
    return rows


def normalize_counter(counter, max_num, smooth=1.0):
    arr = np.array([counter.get(i, 0) + smooth for i in range(1, max_num + 1)], dtype="float64")
    arr = arr / arr.sum()
    return arr


def get_gap_scores(rows, max_num, part="front"):
    last_seen = {}
    total = len(rows)

    for idx, (_, front, back) in enumerate(rows):
        nums = front if part == "front" else back
        for n in nums:
            last_seen[n] = idx

    gaps = {}
    for n in range(1, max_num + 1):
        if n in last_seen:
            gaps[n] = total - 1 - last_seen[n]
        else:
            gaps[n] = total

    gap_arr = np.array([gaps[i] for i in range(1, max_num + 1)], dtype="float64")

    # 间隔越久，分数略高，但不要过度相信冷号
    gap_arr = np.log1p(gap_arr)
    gap_arr = gap_arr / gap_arr.sum()
    return gap_arr


def weighted_sample_without_replacement(numbers, probs, k):
    numbers = list(numbers)
    probs = np.array(probs, dtype="float64")
    probs = probs / probs.sum()

    selected = []

    for _ in range(k):
        idx = np.random.choice(len(numbers), p=probs)
        selected.append(numbers[idx])

        numbers.pop(idx)
        probs = np.delete(probs, idx)

        if len(probs) > 0:
            probs = probs / probs.sum()

    return sorted(selected)


def calc_history_stats(rows):
    front_counter = Counter()
    back_counter = Counter()

    front_sums = []
    back_sums = []
    odd_counts = []
    big_counts = []

    for _, front, back in rows:
        front_counter.update(front)
        back_counter.update(back)

        front_sums.append(sum(front))
        back_sums.append(sum(back))
        odd_counts.append(sum(1 for x in front if x % 2 == 1))
        big_counts.append(sum(1 for x in front if x >= 18))

    stats = {
        "front_counter": front_counter,
        "back_counter": back_counter,
        "front_sum_mean": np.mean(front_sums),
        "front_sum_std": np.std(front_sums) + 1e-6,
        "back_sum_mean": np.mean(back_sums),
        "back_sum_std": np.std(back_sums) + 1e-6,
        "odd_mean": np.mean(odd_counts),
        "odd_std": np.std(odd_counts) + 1e-6,
        "big_mean": np.mean(big_counts),
        "big_std": np.std(big_counts) + 1e-6,
    }

    return stats


def gaussian_score(x, mean, std):
    z = (x - mean) / std
    return math.exp(-0.5 * z * z)


def combo_score(front, back, stats, front_prob, back_prob, front_gap_prob, back_gap_prob):
    score = 0.0

    # 1. 号码频率分数
    for n in front:
        score += math.log(front_prob[n - 1] + 1e-12)
    for n in back:
        score += math.log(back_prob[n - 1] + 1e-12)

    # 2. 冷热间隔分数
    for n in front:
        score += 0.6 * math.log(front_gap_prob[n - 1] + 1e-12)
    for n in back:
        score += 0.6 * math.log(back_gap_prob[n - 1] + 1e-12)

    # 3. 和值约束
    front_sum = sum(front)
    back_sum = sum(back)
    score += 2.0 * math.log(gaussian_score(front_sum, stats["front_sum_mean"], stats["front_sum_std"]) + 1e-12)
    score += 1.0 * math.log(gaussian_score(back_sum, stats["back_sum_mean"], stats["back_sum_std"]) + 1e-12)

    # 4. 奇偶比例约束
    odd_count = sum(1 for x in front if x % 2 == 1)
    score += 1.5 * math.log(gaussian_score(odd_count, stats["odd_mean"], stats["odd_std"]) + 1e-12)

    # 5. 大小比例约束
    big_count = sum(1 for x in front if x >= 18)
    score += 1.5 * math.log(gaussian_score(big_count, stats["big_mean"], stats["big_std"]) + 1e-12)

    # 6. 连号轻微奖励
    front_sorted = sorted(front)
    consecutive = sum(1 for i in range(len(front_sorted) - 1) if front_sorted[i + 1] - front_sorted[i] == 1)
    if consecutive == 1:
        score += 0.5
    elif consecutive >= 3:
        score -= 1.0

    # 7. 过于极端的组合惩罚
    if max(front) - min(front) < 10:
        score -= 1.0

    return score


def main():
    rows = load_data(DATA_PATH)
    print(f"Loaded draws: {len(rows)}")
    print(f"Earliest: {rows[0][0]}, Latest: {rows[-1][0]}")

    stats = calc_history_stats(rows)

    front_prob = normalize_counter(stats["front_counter"], FRONT_MAX, smooth=1.0)
    back_prob = normalize_counter(stats["back_counter"], BACK_MAX, smooth=1.0)

    front_gap_prob = get_gap_scores(rows, FRONT_MAX, part="front")
    back_gap_prob = get_gap_scores(rows, BACK_MAX, part="back")

    # 混合概率：历史频率 + 冷热间隔
    front_mix = 0.75 * front_prob + 0.25 * front_gap_prob
    back_mix = 0.75 * back_prob + 0.25 * back_gap_prob

    candidates = []

    for _ in range(N_SAMPLES):
        front = weighted_sample_without_replacement(range(1, FRONT_MAX + 1), front_mix, 5)
        back = weighted_sample_without_replacement(range(1, BACK_MAX + 1), back_mix, 2)

        score = combo_score(
            front, back,
            stats,
            front_prob, back_prob,
            front_gap_prob, back_gap_prob
        )

        candidates.append((score, front, back))

    candidates.sort(key=lambda x: x[0], reverse=True)

    print("\nTop candidate sets:")
    used = set()
    count = 0

    for score, front, back in candidates:
        key = tuple(front + back)
        if key in used:
            continue

        used.add(key)
        count += 1

        print(f"Set {count:02d}: {front} + {back} | score={score:.4f}")

        if count >= TOP_K:
            break


if __name__ == "__main__":
    main()
