import yaml
from utils.data_utils import create_structured_npy_dataset

def main():
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting dataset processing...")
    create_structured_npy_dataset(config)
    print("Dataset processing complete.")

if __name__ == '__main__':
    main()