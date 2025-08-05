from load_data import load_data
from category_encoders import TargetEncoder


def preprocess_data():
    dataset = load_data()
    dataset.drop(columns=['flight'], inplace=True)
    encoder = TargetEncoder()
    dataset['airline'] = encoder.fit_transform(dataset['airline'], dataset['price'])
    dataset['source_city'] = encoder.fit_transform(dataset['source_city'], dataset['price'])
    dataset['destination_city'] = encoder.fit_transform(dataset['destination_city'], dataset['price'])
    dataset['departure_time'] = encoder.fit_transform(dataset['departure_time'], dataset['price'])
    dataset['arrival_time'] = encoder.fit_transform(dataset['arrival_time'], dataset['price'])
    dataset['stops'] = dataset['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})
    dataset['class'] = dataset['class'].map({'Economy': 0, 'Business': 1})
    
    dataset['airline_class'] = dataset['airline'].astype(str) + '_' + dataset['class'].astype(str)
    dataset['duration_stops'] = dataset['duration'].astype(str) + '_' + dataset['stops'].astype(str)

    dataset['airline_class'] = encoder.fit_transform(dataset['airline_class'], dataset['price'])
    dataset['duration_stops'] = encoder.fit_transform(dataset['duration_stops'], dataset['price']) 

    dataset['is_direct_flight'] = dataset['stops'].apply(lambda x: 1 if x == 0 else 0)
    dataset['avg_airline_price'] = dataset.groupby('airline')['price'].transform('mean')

    dataset.drop(['airline', 'class', 'duration', 'stops'], axis=1, inplace=True)
    return dataset