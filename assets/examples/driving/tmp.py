import pickle

def load_and_print_shape(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        for i, item in enumerate(data):
            print(f"Item {i} shape: {item.shape}")
            if i==0:
                print(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"Key '{key}' shape: {value}")
    else:
        print(f"Data shape: {data.shape}")

# 使用例
pkl_file_path = 'D:\LivePortrait/10.pkl'
load_and_print_shape(pkl_file_path)
