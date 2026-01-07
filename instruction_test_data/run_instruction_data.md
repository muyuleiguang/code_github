# 1. Download instruction fine-tuning data

python download_instruction_data.py --max_samples 10000

# 2. Analyze instruction data characteristics

python analyze_instruction_data.py

# 3. Extract instruction features

python extract_instruction_features.py

# 4. Match pre-training data using the extracted features

python match_pretraining_data.py --threshold 0.3

# 5. Build the test dataset

python build_instruction_test_data.py

# Optional: adjust parameters and re-run

python match_pretraining_data.py --threshold 0.5  # Increase the matching threshold
python build_instruction_test_data.py --seed 123   # Change the random seed