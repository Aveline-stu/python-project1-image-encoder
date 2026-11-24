#E1
import numpy as np
A=np.array([[0.20,-0.50,1.10,0.40],
            [0.80,0.30,-0.70,0.60],
            [-0.40,0.90,0.10,-0.20]])
print(A.ndim)
print(A.shape)
print(A.size)
row=A[1,:]
col=A[:,-2:]
print(row)
print(col)
B=A.copy()
B[B<0]=0
print(B)
print(A)
print("")
#E2
import numpy as np
a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])
num=a*b
dot=a@b
print(num,num.shape)
print(dot,dot.shape)
#欧式长度
len_a=np.sqrt(np.sum(a**2))
len_b=np.sqrt(np.sum(b**2))
print(len_a,len_b)
print(np.linalg.norm(a))
print(np.linalg.norm(b))
print("")
#E3
def angle_deg(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 如果是零向量，抛出异常
    if np.allclose(norm_a, 0) or np.allclose(norm_b, 0):
        raise ValueError("输入不能是零向量")

    dot_val = a @ b
    cos_theta = dot_val / (norm_a * norm_b)

    # 钳制到 [-1, 1]，防止浮点误差超出定义域导致 arccos 报错
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    rad = np.arccos(cos_theta)
    deg = np.rad2deg(rad)  # 弧度转角度
    return deg
# 测试样例
a_test = np.array([1.00, -1.00])
b_test = np.array([1.00, 0.00])
print("夹角 =", angle_deg(a_test, b_test), "度")

#E4
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

diff = p - q
print("差向量 p-q：", diff)

euclid_dist = np.sqrt(np.sum(diff ** 2))
print("欧氏距离：", euclid_dist)

manhattan_dist = np.sum(np.abs(diff))
print("曼哈顿距离：", manhattan_dist)

#E5
import numpy as np

def unit_vector(a):
    # 计算向量长度（模）
    length = np.linalg.norm(a)
    # 边界判断：零向量抛 ValueError
    if np.allclose(length, 0):
        raise ValueError("输入为零向量，无法转为单位向量")
    # 除以模，得到单位向量，不修改原数组
    return a / length

# 题目给定数据
a = np.array([-1.20, 0.40, 0.80])
result = unit_vector(a)

print("原向量：", a)
print("单位向量：", result)
# 用 np.allclose 核对长度是否为 1
print("长度是否≈1：", np.allclose(np.linalg.norm(result), 1))
print("")
#E6
import numpy as np
A = np.array([[0.20, -0.50, 1.10],
              [0.80, 0.40, -0.30]])

sum0 = A.sum(axis=0)   # 按列求和
mean1 = A.mean(axis=1) # 按行求平均
max_all = A.max()

print("A.sum(axis=0) =", sum0)
print("sum0 shape:", sum0.shape)
print("A.mean(axis=1) =", mean1)
print("mean1 shape:", mean1.shape)
print("全局最大值 A.max() =", max_all)
#E7
X = np.array([[0.80, -0.60, 0.70],
              [-0.40, 0.50, 0.20]])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])

Y = X * scale + bias

print("Y = ", Y)
print("Y shape:", Y.shape)
#E8
import numpy as np

# 给定数据，单张图像格式：(C, H, W)
a = np.array([
    [[0.20, -0.40], [0.70, 0.10]],
    [[0.30, 0.80], [-0.20, 0.50]],
    [[-0.10, 0.60], [0.40, -0.30]]
])
b = a + 0.10

# 1. np.stack 在第0维堆叠，构造batch
batch = np.stack([a, b], axis=0)
print("batch shape (B,C,H,W):", batch.shape)

# 2. 提取两张图的蓝色通道 C=2
blue_channel = batch[:, 2, :, :]
print("蓝色通道 shape (B,H,W):", blue_channel.shape)
print("蓝色通道数据：", blue_channel)

# 3. 通道顺序转换 (B,C,H,W) → (B,H,W,C)
batch_bhwc = np.transpose(batch, (0, 2, 3, 1))
print("转成(B,H,W,C)之后shape：", batch_bhwc.shape)


import numpy as np

# M1
def angles_to(query, X):
    norm_q = np.linalg.norm(query)
    if np.isclose(norm_q, 0):
        raise ValueError("query 不能为零向量")

    norm_X = np.linalg.norm(X, axis=1, keepdims=True)  # shape (N,1)
    dot_product = X @ query
    denom = norm_X.flatten() * norm_q

    cos_theta = np.zeros_like(dot_product)
    non_zero_mask = ~np.isclose(norm_X.flatten(), 0, atol=1e-9)
    cos_theta[non_zero_mask] = dot_product[non_zero_mask] / denom[non_zero_mask]
    cos_theta[~non_zero_mask] = np.nan

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rad = np.arccos(cos_theta)
    deg = np.rad2deg(rad)
    return deg

# M1测试
query = np.array([0.60, -0.80, 0.00])
X = np.array([
    [0.30, -0.40, 0.00],
    [-0.60, 0.80, 0.00],
    [0.80, 0.60, 0.00],
    [0.00, 0.00, 0.00]
])
print("==== NP-M1 ====")
angles = angles_to(query, X)
print("角度数组：", angles)
print("输出shape：", angles.shape)


# M2
def top_k_cosine(query, gallery, k):
    norm_q = np.linalg.norm(query)
    if np.isclose(norm_q, 0):
        raise ValueError("query不能是零向量")

    norm_g = np.linalg.norm(gallery, axis=1, keepdims=True)
    dot = gallery @ query
    denom = norm_g.flatten() * norm_q

    cos_sim = np.zeros_like(dot)
    non_zero_mask = ~np.isclose(norm_g.flatten(), 0, atol=1e-9)
    cos_sim[non_zero_mask] = dot[non_zero_mask] / denom[non_zero_mask]
    cos_sim[~non_zero_mask] = np.nan
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # 稳定降序排序
    idx_sort = np.argsort(-cos_sim, kind="stable")
    if not isinstance(k, int) or k <= 0 or k > len(gallery):
        raise ValueError("k必须是正整数且不超过样本数量")
    top_idx = idx_sort[:k]
    top_sim = cos_sim[top_idx]
    return top_sim, top_idx

# M2测试
query = np.array([0.60, -0.80, 0.00])
gallery = np.array([
    [0.60, -0.80, 0.00],
    [0.30, -0.40, 0.00],
    [0.80, 0.60, 0.00],
    [-0.60, 0.80, 0.00],
    [0.00, 0.00, 0.00]
])
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
k = 3
print("\n==== NP-M2 ====")
sim, idx = top_k_cosine(query, gallery, k)
print(f"Top{k}相似度：", sim)
print(f"Top{k}索引：", idx)
print(f"Top{k}ID：", ids[idx])


# M3
A = np.array([
    [0.20, -0.40],
    [0.80, 0.30],
    [-0.50, 0.70]
])
B = np.array([
    [0.10, 0.20],
    [-0.60, -0.20]
])
print("\n==== NP-M3 ====")
diff = A[:, None, :] - B[None, :, :]
print("差数组 shape：", diff.shape)
D = np.sqrt(np.sum(diff ** 2, axis=-1))
print("距离矩阵D：\n", D)
print("D shape：", D.shape)
argmin_B = np.argmin(D, axis=1)
print("A每行最近B的索引：", argmin_B)


# M4
X = np.array([
    [0.80, -0.60, 0.70],
    [-0.40, -0.50, 0.90]
])
W1 = np.array([
    [-0.10, 0.20, -0.80],
    [0.30, -0.50, 0.10]
])
b1 = np.array([-0.15, 0.05])
W2 = np.array([
    [0.20, -0.60],
    [-0.80, 0.52]
])
b2 = np.array([0.10, -0.20])

print("\n==== NP-M4 ====")
Z = X @ W1.T + b1
print("Z:\n", Z)
print("Z shape：", Z.shape)
H = np.maximum(Z, 0.0)
print("H:\n", H)
print("H shape：", H.shape)
Y = H @ W2.T + b2
print("Y:\n", Y)
print("Y shape：", Y.shape)