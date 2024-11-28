import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations

'''
将eval_results.json中的结果按照参数组合两两生成热力图，
如果参数只有一个值则不参与画图
热力图中有三个子图分别展示pot_f1, collision_f1, rebound_f1结果
每个子图需要同时展示F1,Precision,Recall三个子子图
'''

# 读取 JSON 文件
with open('eval_results.json', 'r') as f:
    result_data = json.load(f)

# 提取数据
params_list = []
pot_f1 = []
collision_f1 = []
rebound_f1 = []

# 获取第一个 entry 中的所有参数名称，用于后续组合
param_names = list(result_data[0]["params_setting"].keys()) if result_data else []

for entry in result_data:
    params_setting = entry["params_setting"]
    average_results = entry["average_results"]

    # 将参数和指标提取到字典中
    param_dict = params_setting.copy()
    # 确保每个指标都被正确提取
    if "pot" in average_results:
        param_dict["pot_f1"] = average_results["pot"].get("f1", None)
        param_dict["pot_precision"] = average_results["pot"].get("precision", None)
        param_dict["pot_recall"] = average_results["pot"].get("recall", None)

    if "collision" in average_results:
        param_dict["collision_f1"] = average_results["collision"].get("f1", None)
        param_dict["collision_precision"] = average_results["collision"].get("precision", None)
        param_dict["collision_recall"] = average_results["collision"].get("recall", None)

    if "rebound" in average_results:
        param_dict["rebound_f1"] = average_results["rebound"].get("f1", None)
        param_dict["rebound_precision"] = average_results["rebound"].get("precision", None)
        param_dict["rebound_recall"] = average_results["rebound"].get("recall", None)

    params_list.append(param_dict)

# 转换为 DataFrame
df = pd.DataFrame(params_list)

# 检查 DataFrame 中的列名
print("DataFrame 列名：", df.columns)

# 如果 'load_inference_state_path' 只有一个唯一值，并且该值为 None，则跳过与其他参数的比较
if df['load_inference_state_path'].dropna().nunique() == 0:  # 如果删除 None 后，唯一值的数量为 0
    print("'load_inference_state_path' 只有一个值，跳过与其他参数的比较")
    param_names = [param for param in param_names if param != 'load_inference_state_path']

# 获取参数的两两组合
param_combinations = list(combinations(param_names, 2))

# 生成每个参数组合的热力图
for comb in param_combinations:
    # 检查每个参数的唯一值数量
    unique_values_0 = df[comb[0]].nunique()
    unique_values_1 = df[comb[1]].nunique()

    # 如果某个参数的唯一值数量为 1，跳过该组合
    if unique_values_0 == 1 or unique_values_1 == 1:
        print(f"跳过 {comb[0]} 和 {comb[1]} 的比较，因为其一参数只有唯一值")
        continue


    # 创建图表并设置子图
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))  # 3行3列，每行包含 F1, Precision 和 Recall 三个子图
    fig.suptitle(f"Heatmap for {comb[0]} vs {comb[1]}", fontsize=16)

    # 每个指标都生成三个热力图：F1、Precision 和 Recall
    for i, metric in enumerate(['f1', 'precision', 'recall']):
        for j, value_type in enumerate(['pot', 'collision', 'rebound']):
            # 构建对应指标的列名
            column_name = f"{value_type}_{metric}"
            ax = axes[i][j]

            # 创建热力图
            pivot_table = df.pivot_table(index=comb[0], columns=comb[1], values=column_name, aggfunc='mean')
            if not pivot_table.empty:  # 只绘制非空数据
                sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", linewidths=0.5, ax=ax)
                ax.set_title(f"{value_type.capitalize()} {metric.capitalize()}")
                ax.set_xlabel(comb[1])
                ax.set_ylabel(comb[0])

    # # 创建图表并设置子图
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列
    # fig.suptitle(f"Heatmap for {comb[0]} vs {comb[1]}", fontsize=16)
    #
    # # 创建 `pot_f1` 热力图
    # pivot_table = df.pivot_table(index=comb[0], columns=comb[1], values="pot_f1", aggfunc='mean')
    # if not pivot_table.empty:  # 只绘制非空数据
    #     sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", linewidths=0.5, ax=axes[0])
    #     axes[0].set_title("Pot F1")
    #     axes[0].set_xlabel(comb[1])
    #     axes[0].set_ylabel(comb[0])
    #
    # # 创建 `collision_f1` 热力图
    # pivot_table = df.pivot_table(index=comb[0], columns=comb[1], values="collision_f1", aggfunc='mean')
    # if not pivot_table.empty:  # 只绘制非空数据
    #     sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", linewidths=0.5, ax=axes[1])
    #     axes[1].set_title("Collision F1")
    #     axes[1].set_xlabel(comb[1])
    #     axes[1].set_ylabel(comb[0])
    #
    # # 创建 `rebound_f1` 热力图
    # pivot_table = df.pivot_table(index=comb[0], columns=comb[1], values="rebound_f1", aggfunc='mean')
    # if not pivot_table.empty:  # 只绘制非空数据
    #     sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", linewidths=0.5, ax=axes[2])
    #     axes[2].set_title("Rebound F1")
    #     axes[2].set_xlabel(comb[1])
    #     axes[2].set_ylabel(comb[0])

    # 保存图像到文件
    filename = f"f1_heatmap_{comb[0]}_{comb[1]}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 关闭图表
    plt.close()