from load_data import load_data
from model import model_training
from preprocess import preprocess_data

def main():
    load_data()
    print("Data loaded successfully.")

    preprocess_data()

    results = model_training()
    print("Results after feature engineering:")
    print(results)

main()
