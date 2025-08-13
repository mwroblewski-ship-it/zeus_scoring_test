#!/bin/bash

# Skrypt uruchamiający lokalny test validatora Zeus
# 
# Użycie:
#   chmod +x run_local_test.sh
#   ./run_local_test.sh

echo "Lokalny Tester Validatora Zeus"
echo "=============================="

# Sprawdź czy środowisko conda jest aktywne
if [[ "$CONDA_DEFAULT_ENV" != "zeus" ]]; then
    echo "Aktywowanie środowiska conda 'zeus'..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate zeus
fi

# Sprawdź czy plik validator.env istnieje
if [ ! -f "validator.env" ]; then
    echo "BŁĄD: Nie znaleziono pliku validator.env"
    echo "Skopiuj i skonfiguruj plik validator.env z odpowiednimi kluczami API"
    exit 1
fi

# Sprawdź czy zeus jest zainstalowany
python -c "import zeus" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "BŁĄD: Pakiet zeus nie jest zainstalowany"
    echo "Uruchom: pip install -e ."
    exit 1
fi

# Sprawdź zmienne środowiskowe
echo "Sprawdzanie konfiguracji..."
source validator.env

if [ -z "$CDS_API_KEY" ]; then
    echo "UWAGA: Brak CDS_API_KEY - dane ERA5 mogą być niedostępne"
fi

if [ -z "$OPEN_METEO_API_KEY" ]; then
    echo "UWAGA: Brak OPEN_METEO_API_KEY - używam darmowego API (ograniczenia)"
fi

echo "Uruchamianie lokalnego testera..."
python testing.py

echo "Test zakończony."