import joblib
import os
import re

# 读取原始数据
data_path = "data/g1_23/amass_all.pkl"
output_dir = "data/g1_23/separated_motions_amass_all"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data_dump = joblib.load(data_path)

# 遍历字典，逐个存储 motion 数据（保持 {key: motion_data} 结构）
for key, motion_data in data_dump.items():
    # 替换空格和 `-` 为 `_`，并合并多个 `_`
    safe_key = re.sub(r'[-\s]+', '_', key)

    output_file = os.path.join(output_dir, f"{safe_key}.pkl")
    joblib.dump({key: motion_data}, output_file)  # 保持 key: motion_data 结构
    print(f"Saved: {output_file}")
