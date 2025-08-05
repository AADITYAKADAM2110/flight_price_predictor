import pandas as pd

def load_data():
    dataset = pd.read_csv(r'C:\Users\DELL\Desktop\Flight Ticket Prediction Project\dataset\airlines_flights_data.csv')
    return dataset