import numpy as np

# 1. 设置上限（需 >= 50）
upper_limit = 75879

# 2. 设定正态分布参数
mean = upper_limit / 2          # 均值在中间
std_dev = upper_limit / 6       # 标准差（控制分布宽度，可调）

# 3. 不断采样直到获得50个不同整数
selected = set()
while len(selected) < 50:
    # 生成一批服从正态分布的随机数
    samples = np.random.normal(loc=mean, scale=std_dev, size=100)
    # 取整并过滤范围外的数
    valid_samples = [int(round(x)) for x in samples if 0 <= x <= upper_limit]
    selected.update(valid_samples)
    # 防止过多迭代
    if len(selected) > 1000:
        break

# 只保留前50个
numbers = sorted(list(selected))[:50]

print("✅ 选取的50个数字（符合近似正态分布）:")
print(numbers)

