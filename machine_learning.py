import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

def prepare(data_path_order,data_path_products):
    """
    Načíta a pripraví dáta na trénovanie modelu.
    """
    # Načítanie dát
    orders_past = pd.read_csv(data_path_order)
    products = pd.read_csv(data_path_products)
    print(orders_past.columns)


    # Preprocessing
    orders_past['order_date_numeric'] = pd.to_datetime(orders_past['Date Order was placed'],format='%d-%b-%y', errors='coerce').map(datetime.toordinal)
    orders_past['delivery_date_numeric'] = pd.to_datetime(orders_past['Delivery Date'],format='%d-%b-%y', errors='coerce').map(datetime.toordinal)

    # Vstupné a cieľové atribúty
    x = orders_past[['Cost Price Per Unit', 'Total Retail Price for This Order', 'Quantity Ordered', 'Product ID', 'order_date_numeric']]
    y = orders_past['delivery_date_numeric']

    return x, y, products

def train_and_save_model(x, y, model_path):
    """
    Trénuje model a ukladá ho do súboru.
    """

    # Tréning modelu
    model = RandomForestRegressor()
    model.fit(x, y)

    # Uloženie modelu
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model uložený do {model_path}")

def predict_from_input(model_path):
    """
    Načíta model a predikuje dátum doručenia na základe vstupov z konzoly.
    """
    # Načítanie modelu
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Zadanie testovacích dát cez konzolu
    print("Zadajte údaje pre predikciu:")
    cost_price_per_unit = float(input("Cost Price per Unit: "))
    total_retail_price = float(input("Total Retail Price for this Order: "))
    quantity_ordered = int(input("Quantity Ordered: "))
    product_id = int(input("Product ID: "))
    date_order_placed = input("Date Order Was Placed (exp.02-Jan-17): ")

    # Prevod dátumu na numerickú hodnotu
    order_date_numeric = datetime.strptime(date_order_placed, '%d-%b-%y').toordinal()

    # Príprava vstupných dát pre model
    features = [[cost_price_per_unit, total_retail_price, quantity_ordered, product_id, order_date_numeric]]

    # Predikcia
    predicted_date_numeric = model.predict(features)[0]
    predicted_date = datetime.fromordinal(int(predicted_date_numeric)).strftime('%d-%b-%y')
    print(f"Predikovaný dátum doručenia: {predicted_date}")

def main():
    """
    Hlavná funkcia na spustenie prípravy dát, trénovania a predikcie.
    """
    # Cesty k súborom
    #data_path_order = "orders.csv"  # Cesta k datasetu
    #data_path_products = "product-supplier.csv"  # Cesta k datasetu
    model_path = "delivery_date_model.pkl"  # Cesta k modelu

    # Načítanie a príprava dát
    #X, y, products = prepare(data_path_order,data_path_products)

    # Trénovanie a ukladanie modelu toto stači spustiť raz pri provom spusteni a potom to hodiť do komentara.
    #train_and_save_model(X, y, model_path)

    # Predikcia na základe vstupu z konzoly
    predict_from_input(model_path)

if __name__ == "__main__":
    main()
