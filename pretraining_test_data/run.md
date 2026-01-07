# 1. 下载数据
python download_data.py --sample_size 10000

# 2. 分析原始数据
python analyze_data.py

# 3. 预处理数据
python preprocess_data.py

# 4. 构建测试数据；stackexchange获取指令格式数据。
python build_test_data.py

# 5. 采样最终数据集
python sample_data.py --sample_size 1000

# 6. 分析最终测试数据
python analyze_test_data.py