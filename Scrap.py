import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_police_hardcore(start_page=0, target_years=10):
    base_url = "https://policja.pl/pol/form/1,Informacja-dzienna.html"
    all_rows = []
    page = start_page
    
    # Udajemy Chrome na Windowsie
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    print(f"🚀 Startujemy od strony {page}. Cel: ok. {target_years * 74} stron.")

    while True:
        try:
            response = requests.get(f"{base_url}?page={page}", headers=headers, timeout=20)
            
            if response.status_code == 503:
                print(f"⚠️ Serwer zmęczony (503) na stronie {page}. Odpoczywam 30 sek...")
                time.sleep(30)
                continue # Próbuje tę samą stronę jeszcze raz

            if response.status_code != 200:
                print(f"Koniec lub błąd: Status {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            if not table: break

            rows = table.find_all('tr')
            new_on_page = 0
            for row in rows:
                cols = [td.text.strip().replace('\xa0', '') for td in row.find_all(['td', 'th'])]
                if len(cols) >= 5 and "Data" not in cols[0]:
                    all_rows.append(cols)
                    new_on_page += 1

            if new_on_page == 0:
                print("Brak nowych danych. Koniec archiwum.")
                break

            print(f"✅ Strona {page} pobrana. Ostatnia data: {all_rows[-1][0]}")
            
            # ZAPISUJEMY CO 10 STRON DO PLIKU (Backup)
            if page % 10 == 0:
                pd.DataFrame(all_rows).to_csv("backup_policja.csv", index=False)

            page += 1
            # LOSOWE OPÓŹNIENIE (Kluczowe, żeby uniknąć 503)
            time.sleep(random.uniform(1.5, 3.0)) 

        except Exception as e:
            print(f"Błąd krytyczny: {e}. Czekam 10s i próbuje dalej...")
            time.sleep(10)
            continue

    # Finalny zapis
    columns = ["Data", "Interwencje", "Zatrz_G_Uczynek", "Zatrz_Poszukiwani", "Alkohol", "Wypadki", "Zabici", "Ranni"]
    df = pd.DataFrame(all_rows)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    df.to_csv("policja_10_lat_full.csv", index=False, encoding='utf-8-sig')
    print(f"🏁 FINAŁ! Zebrano {len(df)} rekordów.")

if __name__ == "__main__":
    # Jeśli wywaliło Cię na 20, możesz tu wpisać start_page=21
    scrape_police_hardcore(start_page=0)