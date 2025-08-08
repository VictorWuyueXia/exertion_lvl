#!/usr/bin/env python3
import pandas as pd

# 读取标签文件
labels_df = pd.read_csv("data/general_information.csv")

# 统计各类别的数量
exertion_counts = labels_df['Exertion'].value_counts().sort_index()
print("各类别数量:")
for exertion_level, count in exertion_counts.items():
    print(f"  类别 {exertion_level-1}: {count} 个样本")

# 检查类别4的具体会话
class4_sessions = labels_df[labels_df['Exertion'] == 5]['Session Name'].tolist()
print(f"\n类别4的会话数: {len(class4_sessions)}")
print("类别4的会话示例:")
for session in class4_sessions[:5]:
    print(f"  {session}")

