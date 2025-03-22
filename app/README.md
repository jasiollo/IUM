# Dokumentacja mikroserwisu

Stworzony przez nas mikroserwis umożliwia:
- Predykcje przy użyciu dwóch modeli: Model bazowy i Model zaawansowany
- Eksperymenty A/B porównujące wyniki obu modeli


## Wymagania

- Python 3.7+
- Virtualenv lub dowolne środowisko wirtualne

## Struktura katalogów

```
.
├── README.md
├── __init__.py
├── __pycache__
├── app.py
├── input
├── load_data.py
├── models
├── requirements.txt
├── routes.py
├── tests
├── utils
└── venv
```

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone <URL_DO_REPOZYTORIUM>
   cd <NAZWA_KATALOGU>
   ```

2. Środowisko wirtualne:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Zainstaluj wymagane dependencje:
   ```bash
   pip install -r requirements.txt
   ```

## Uruchamianie serwera

Aby uruchomić mikroserwis::
```bash
python app.py
```
Serwer będzie dostępny pod adresem: `http://0.0.0.0:8080`.

## Endpointy API

### Trenowanie modelu bazowego

- **URL:** `/base_model/train`
- **Metoda:** `POST`
- **Opis:** Trenuje model bazowy przy użyciu podanego pliku JSONL.
- **Przykład użycia (curl):**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"file_path": "input/monthly_listening_v3.jsonl"}' http://localhost:8080/base_model/train
  ```
- **Przykładowa odpowiedź:**
  ```json
  {
      "message": "Base model trained successfully."
  }
  ```

### MAE modelu bazowego

- **URL:** `/base_model/mae`
- **Metoda:** `GET`
- **Opis:** Zwraca wartość MAE (Mean Absolute Error) modelu bazowego.
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/base_model/mae
  ```
- **Przykładowa odpowiedź:**
  ```json
  {
      "mae": 5.2
  }
  ```

### Wartości predykcji modelu bazowego

- **URL:** `/base_model/predict`
- **Metoda:** `GET`
- **Opis:** Zwraca rzeczywiste i przewidziane wartości modelu bazowego.
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/base_model/predict
  ```
- **Przykładowa odpowiedź:**
  ```json
  {
      "actual": [10, 15, 20],
      "predicted": [11, 14, 19]
  }
  ```

### Ważność atrybutów modelu bazowego

- **URL:** `/base_model/importance_features`
- **Metoda:** `GET`
- **Opis:** Zwraca ważność atrybutów modelu bazowego.
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/base_model/importance_features
  ```
- **Przykładowa odpowiedź:**
  ```json
  {
      "feature1": 0.45,
      "feature2": 0.35,
      "feature3": 0.20
  }
  ```

## Przykładowy JSON wejściowy w przypadku modelu bazowego

```json
{
    "file_path": "input/monthly_listening_v3.jsonl"
}
```

## Model Zaawansowany

### Trenowanie modelu zaawansowanego

- **URL:** `/advanced_model/train`
- **Metoda:** `POST`
- **Opis:** Trenuje model zaawansowany przy użyciu pliku JSONL.
- **Przykład użycia (curl):**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"file_path": "input/merged_llp.jsonl"}' http://localhost:8080/advanced_model/train

### MAE modelu zaawansowanego

- **URL:** `/advanced_model/mae`
- **Metoda:** `GET`
- **Opis:** Zwraca MAE modelu zaawansowanego
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/advanced_model/mae
  ```

### Predykcja modelu zaawansowanego

- **URL:** `/advanced_model/predict`
- **Metoda:** `GET`
- **Opis:** Zwraca predykcję modelu zaawansowanego
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/advanced_model/predict
  ```
  ### Porównanie atrybutów modelu zaawansowanego

- **URL:** `/advanced_model/importance_features`
- **Metoda:** `GET`
- **Opis:** Zwraca MAE modelu zaawansowanego
- **Przykład użycia (curl):**
  ```bash
  curl -X GET http://localhost:8080/advanced_model/importance_features
  ```

## Test AB

- **URL:**  `/ab_experiment/train_and_compare`
- **Metoda:** `POST`
- **Opis:** Zwraca przygotowane dane do oceny jakości modeli
- **Przykład użycia (curl):**
  ```bash
    curl -X POST http://127.0.0.1:8080/ab_experiment/train_and_compare -H "Content-Type: application/json" -d '{
      "base_file_path": "input/monthly_listening_v3.jsonl",
      "advanced_file_path": "input/merged_llp.jsonl"
  }'
  ```




## Testy

Uruchamiamy serwer

```bash
python app.py
```

a potem w katalogu tests/

```bash
pytest NAZWA_TESTU
```

## Obsługa skryptu

Uruchamiamy serwer 

```bash
python app.py
```

a potem w katalogu ium/

```bash
python3 skrypt_flask
```