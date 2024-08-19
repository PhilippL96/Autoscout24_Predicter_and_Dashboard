import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import locale

locale.setlocale(locale.LC_ALL, 'de_DE')

# Daten vorbereiten

if 'data' not in st.session_state:
    df = pd.read_csv('autoscout24.csv', delimiter=';')
    df = df.rename(columns={'year of sale': 'year'})
    hp_means = df.groupby(['make', 'model'])['hp'].transform('mean')
    df['hp'] = df['hp'].fillna(hp_means)
    df = df.dropna()

    def group_fuel(x):
        if x == 'Diesel':
            return 'Diesel'
        elif x == 'Gasoline':
            return 'Gasoline'
        else:
            return 'Other'

    df['fuel'] = df['fuel'].apply(group_fuel)

    data = df[(df['make'] == 'Volkswagen') | 
            (df['make'] == 'Opel') | 
            (df['make'] == 'Ford') | 
            (df['make'] == 'Skoda') | 
            (df['make'] == 'Renault')].reset_index()
    data.drop('index', axis=1, inplace=	True)
    st.session_state.data = data


# Machine Learning

if 'pipeline' not in st.session_state:
    data_encoded = pd.get_dummies(st.session_state.data, columns=['make', 'model', 'fuel', 'gear', 'offerType'])
    X, y = data_encoded.drop(['price'], axis=1), data_encoded['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    Mean_error = mean_absolute_error(y_test, y_pred)

    st.session_state.pipeline = pipeline
    st.session_state.mae = Mean_error
    st.session_state.y_pred = y_pred
    st.session_state.y_test = y_test
    st.session_state.X_train = X_train


# Visualisiere MAE

if 'fig' not in st.session_state:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.5)
    ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
    ax.set_xlabel('Tatsächliche Werte')
    ax.set_ylabel('Vorhergesagte Werte')
    ax.set_title('Tatsächliche vs. Vorhergesagte Werte')
    st.session_state.fig = fig

st.header('Autopreis-Predictor')

st.write('')
st.write(f'Mittlerer Fehler der Vorhersage: {locale.format_string("%.2f", st.session_state.mae, grouping=True)} €.')

st.pyplot(st.session_state.fig)


# User Input

make = st.selectbox("Was ist die Automarke?", st.session_state.data['make'].unique())
if make:
    model = st.selectbox("Welches Modell?", st.session_state.data[st.session_state.data['make'] == make]['model'].unique())
fuel = st.selectbox("Welchen Kraftstoff verwendet dein Auto?", st.session_state.data['fuel'].unique())
gear = st.selectbox("Welches Getriebe hat das Auto?", st.session_state.data['gear'].unique())
offerType = st.selectbox("Um welche Art Angebot handelt es sich bei dem Auto?", st.session_state.data['offerType'].unique())
if offerType == 'Used':
    mileage = st.number_input("Wie viele Kilometer ist das betreffende Auto gefahren?", min_value=1, max_value=1000000, value=1, step=1, format="%d")
else:
    mileage = 0
hp = st.number_input("Wie viel PS hat das Auto?", min_value=0, max_value=1000, value=100, step=1, format="%d")
year = st.number_input("Für welches Jahr soll der Preis ermittelt werden?", min_value=2000, max_value=2100, value=2010, step=1, format="%d")


# Prediction

def predict_price(user_input, model, dummy_columns):
    """
    Vorhersage des Preises basierend auf den Nutzereingaben.

    :param user_input: Dictionary mit Nutzereingaben
    :param model: Das trainierte Modell (Pipeline)
    :param dummy_columns: Liste der Dummy-Spaltennamen
    :return: Vorhergesagter Preis
    """
    user_df = pd.DataFrame([user_input])
    
    user_df_encoded = pd.get_dummies(user_df, columns=['make', 'model', 'fuel', 'gear', 'offerType'])
    
    user_df_full = pd.DataFrame(columns=dummy_columns)
    
    user_df_full = pd.concat([user_df_full, user_df_encoded], axis=0)
    
    for col in dummy_columns:
        if col not in user_df_full.columns:
            user_df_full[col] = 0
    
    user_df_full = user_df_full[dummy_columns]
    
    user_df_scaled = model.named_steps['scaler'].transform(user_df_full)
    
    predicted_price = model.named_steps['rf'].predict(user_df_scaled)[0]
    
    return predicted_price

if mileage is not None and make is not None and model is not None and gear is not None and fuel is not None and offerType is not None and hp is not None and year is not None:
    user_input = {
    'mileage': mileage,
    'make': make,
    'model': model,
    'fuel': fuel,
    'gear': gear,
    'offerType': offerType,
    'hp': hp,
    'year': year
    }
    prediction = predict_price(user_input, st.session_state.pipeline,st.session_state.X_train.columns.tolist())

st.markdown(f'### Der vorhergesagte Preis beträgt {locale.format_string("%.2f", prediction, grouping=True)} €.')
