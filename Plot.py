import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import holidays

# 1. Przygotowanie danych
data = pd.read_csv("policja_info.csv")
data['Data'] = pd.to_datetime(data['Data'], errors='coerce')
data = data.dropna(subset=['Data']).sort_values('Data')

# Konwersja na liczby
for col in ['Kierujacy_Alkohol', 'Wypadki_Drogowe',"Ranni"]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 2. Kalendarz Świąt
pl_holidays = holidays.Poland()

def get_holiday_info(dt):
    # Sprawdzamy, czy data znajduje się w polskim kalendarzu świąt
    # Jeśli tak, zwracamy nazwę święta
    if dt in pl_holidays:
        return pl_holidays.get(dt)
    
    # Jeśli to nie jest święto, zwracamy "Dzień zwykły"
    return "Dzień zwykły"

data['Opis_Dnia'] = data['Data'].apply(get_holiday_info)
data['Czy_Swieto'] = data['Opis_Dnia'] != "Dzień zwykły"

# 3. Tworzenie wykresu
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True, 
    vertical_spacing=0.1,
    subplot_titles=("Zatrzymani po alkoholu", "Liczba wypadków drogowych","Ranni w wypadkach")
)

# Funkcja do dodawania warstw (zwykłe dni i święta osobno dla legendy i kształtów)
def add_layers(col_name, row_num, color_base):
    # WARSTWA 1: Wszystkie dni (Kropki)
    fig.add_trace(go.Scatter(
        x=data['Data'], 
        y=data[col_name],
        mode='lines+markers',
        name='Dni zwykłe/Niedziele',
        marker=dict(color=color_base, size=5, opacity=0.5),
        customdata=data['Opis_Dnia'],
        hovertemplate='<b>Data:</b> %{x|%Y-%m-%d}<br><b>Status:</b> %{customdata}<br><b>Wartość:</b> %{y}<extra></extra>'
    ), row=row_num, col=1)

    # WARSTWA 2: Tylko święta (Diamenty)
    swieta = data[data['Czy_Swieto']]
    fig.add_trace(go.Scatter(
        x=swieta['Data'], 
        y=swieta[col_name],
        mode='markers',
        name='Święta ustawowe',
        marker=dict(
            color='red', 
            size=9, 
            symbol='diamond', 
            line=dict(width=1, color='black')
        ),
        customdata=swieta['Opis_Dnia'],
        hovertemplate='<b>ŚWIĘTO:</b> %{customdata}<br><b>Data:</b> %{x|%Y-%m-%d}<br><b>Wartość:</b> %{y}<extra></extra>'
    ), row=row_num, col=1)

# Dodanie danych do wykresów
add_layers('Kierujacy_Alkohol', 1, 'green')
add_layers('Wypadki_Drogowe', 2, 'blue')
add_layers('Ranni', 3, 'purple')

# 4. Wygląd i funkcje interaktywne
fig.update_layout(
    height=900,
    title_text="Statystyki Policyjne z zaznaczeniem Świąt (bez niedziel)",
    template="plotly_white",
    hovermode="closest",
    showlegend=True
)

# Suwak czasu
fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
fig.show()