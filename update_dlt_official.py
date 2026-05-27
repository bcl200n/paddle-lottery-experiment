# -*- coding: utf-8 -*-
import requests
import time

API = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.lottery.gov.cn/kj/kjlb.html?dlt=",
}

all_rows = []

for page in range(1, 40):
    params = {
        "gameNo": "85",
        "provinceId": "0",
        "pageSize": "100",
        "isVerify": "1",
        "pageNo": str(page),
    }

    r = requests.get(API, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = data.get("value", {}).get("list", [])
    if not rows:
        break

    for item in rows:
        issue = item.get("lotteryDrawNum", "").strip()
        nums = item.get("lotteryDrawResult", "").strip().split()
        if issue and len(nums) == 7:
            all_rows.append([issue] + nums)

    print(f"page {page}: {len(rows)} rows")
    time.sleep(0.2)

# 去重 + 按期号倒序
unique = {}
for row in all_rows:
    unique[row[0]] = row

clean = sorted(unique.values(), key=lambda x: x[0], reverse=True)

with open("lotto_clean.csv", "w", encoding="utf-8") as f:
    for row in clean:
        f.write(" ".join(row) + "\n")

with open("lotto.csv", "w", encoding="utf-8") as f:
    for row in clean:
        f.write(" ".join(row) + "\n")

print("更新完成")
print("总期数:", len(clean))
print("最新一期:", " ".join(clean[0]))
print("最早一期:", " ".join(clean[-1]))
