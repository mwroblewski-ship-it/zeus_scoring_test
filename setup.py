# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import os
import codecs
import pathlib
from os import path
from io import open
from setuptools import setup, find_packages
from pkg_resources import parse_requirements


def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            # For git or other VCS links
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    # You may decide to raise an exception here,
                    # if you want to ensure every VCS link has an #egg=<package_name> at the end
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements


requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(os.path.join(here, "zeus/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="Zeus",
    version=version_string,
    description="Zeus subnet by Orpheus A.I. (Incubated by BitMind)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Orpheus-AI/Zeus",
    author="orpheus-ai.nl",
    packages=find_packages(),
    include_package_data=True,
    author_email="eric@orpheus-ai.nl",
    license="MIT",
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)



import cdsapi
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple, Dict, Optional
import xarray as xr
import zipfile
import os

class CopernicusERA5WeatherFetcher:
    """
    Klasa do pobierania danych pogodowych z Copernicus ERA5
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicjalizacja klienta CDS API
        
        Args:
            api_key: Klucz API (jeśli None, będzie próbował użyć domyślnej konfiguracji)
        """
        self.client = cdsapi.Client()
        
    def fetch_era5_data(self, 
                       latitudes: List[float], 
                       longitudes: List[float],
                       start_datetime: str,
                       end_datetime: str,
                       output_file: str = "era5_data.nc") -> str:
        """
        Pobiera dane ERA5 dla podanych współrzędnych i przedziału czasowego
        
        Args:
            latitudes: Lista szerokości geograficznych
            longitudes: Lista długości geograficznych  
            start_datetime: Data/czas początkowy (format: 'YYYY-MM-DDTHH:MM')
            end_datetime: Data/czas końcowy (format: 'YYYY-MM-DDTHH:MM')
            output_file: Nazwa pliku wyjściowego
            
        Returns:
            Nazwa pliku z pobranymi danymi
        """
        
        # Konwersja datetime do formatu wymaganego przez CDS
        start_dt = datetime.fromisoformat(start_datetime.replace('%3A', ':'))
        end_dt = datetime.fromisoformat(end_datetime.replace('%3A', ':'))
        
        # Przygotowanie listy lat, miesięcy, dni i godzin
        years = list(set([start_dt.year, end_dt.year]))
        months = []
        days = []
        times = []
        
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.month not in months:
                months.append(current_dt.month)
            if current_dt.day not in days:
                days.append(current_dt.day)
            time_str = f"{current_dt.hour:02d}:00"
            if time_str not in times:
                times.append(time_str)
            current_dt += timedelta(hours=1)
        
        # Określenie obszaru (bounding box)
        north = max(latitudes)
        south = min(latitudes)
        east = max(longitudes)
        west = min(longitudes)
        
        # Parametry żądania do CDS API
        request_params = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature',
                'total_precipitation',
                '100m_u_component_of_wind',
                '100m_v_component_of_wind',
            ],
            'year': [str(year) for year in years],
            'month': [f"{month:02d}" for month in sorted(months)],
            'day': [f"{day:02d}" for day in sorted(days)],
            'time': sorted(times),
            'area': [north, west, south, east],  # North, West, South, East
        }
        
        print("Pobieranie danych z Copernicus ERA5...")
        print(f"Obszar: {north}°N, {west}°W, {south}°S, {east}°E")
        print(f"Okres: {start_datetime} - {end_datetime}")
        
        # Pobranie danych
        self.client.retrieve(
            'reanalysis-era5-single-levels',
            request_params,
            output_file
        )
        
        print(f"Dane zostały pobrane do pliku: {output_file}")
        
        # Sprawdź czy to plik ZIP i rozpakuj jeśli potrzeba
        if zipfile.is_zipfile(output_file):
            print("Plik jest archiwum ZIP, rozpakowuję...")
            zip_dir = output_file.replace('.nc', '_extracted')
            os.makedirs(zip_dir, exist_ok=True)
            
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(zip_dir)
                
            # Znajdź plik NetCDF w rozpakowanym folderze
            nc_files = [f for f in os.listdir(zip_dir) if f.endswith('.nc')]
            if nc_files:
                extracted_file = os.path.join(zip_dir, nc_files[0])
                print(f"Rozpakowano plik NetCDF: {extracted_file}")
                return extracted_file
            else:
                raise Exception("Nie znaleziono pliku NetCDF w archiwum ZIP")
        
        return output_file
    
    def process_data_for_coordinates(self, 
                                   netcdf_file: str,
                                   latitudes: List[float], 
                                   longitudes: List[float]) -> pd.DataFrame:
        """
        Przetwarza dane NetCDF i ekstraktuje wartości dla konkretnych współrzędnych
        
        Args:
            netcdf_file: Ścieżka do pliku NetCDF
            latitudes: Lista szerokości geograficznych
            longitudes: Lista długości geograficznych
            
        Returns:
            DataFrame z danymi pogodowymi dla wszystkich punktów
        """
        
        # Wczytanie danych NetCDF
        try:
            # Próbujemy najpierw netcdf4 (najczęściej używany dla plików z CDS)
            ds = xr.open_dataset(netcdf_file, engine='netcdf4')
        except Exception as e1:
            try:
                # Alternatywnie h5netcdf
                ds = xr.open_dataset(netcdf_file, engine='h5netcdf')
            except Exception as e2:
                try:
                    # W ostateczności scipy (tylko NetCDF3)
                    ds = xr.open_dataset(netcdf_file, engine='scipy')
                except Exception as e3:
                    print(f"Nie udało się otworzyć pliku NetCDF:")
                    print(f"netcdf4 engine: {e1}")
                    print(f"h5netcdf engine: {e2}")
                    print(f"scipy engine: {e3}")
                    raise e1
        
        # Wyświetl informacje o strukturze danych
        print("Struktura danych NetCDF:")
        print("Wymiary:", dict(ds.dims))
        print("Zmienne:", list(ds.data_vars))
        print("Współrzędne:", list(ds.coords))
        print("Pierwsze znaczniki czasowe:", ds.coords.get('time', ds.coords.get('valid_time', 'brak')))
        
        # Sprawdź nazwy zmiennych czasowych
        time_var = None
        for potential_time in ['time', 'valid_time', 'forecast_time']:
            if potential_time in ds.coords:
                time_var = potential_time
                break
        
        if time_var is None:
            print("Dostępne współrzędne:", list(ds.coords))
            raise ValueError("Nie znaleziono zmiennej czasowej w danych")
        
        results = []
        
        # Przetwarzanie dla każdej pary współrzędnych
        for lat, lon in zip(latitudes, longitudes):
            print(f"Przetwarzanie danych dla punktu: {lat}°, {lon}°")
            
            # Znajdowanie najbliższych punktów siatki
            point_data = ds.sel(latitude=lat, longitude=lon, method='nearest')
            
            # Konwersja do DataFrame
            df = point_data.to_dataframe().reset_index()
            
            # Użyj poprawnej nazwy zmiennej czasowej
            if time_var != 'time' and time_var in df.columns:
                df = df.rename(columns={time_var: 'time'})
            
            # Dodanie informacji o współrzędnych
            df['requested_lat'] = lat
            df['requested_lon'] = lon
            df['actual_lat'] = float(point_data.latitude.values)
            df['actual_lon'] = float(point_data.longitude.values)
            
            results.append(df)
        
        # Połączenie wszystkich wyników
        final_df = pd.concat(results, ignore_index=True)
        
        # Sortowanie według współrzędnych, potem według czasu
        final_df = final_df.sort_values(['requested_lat', 'requested_lon', 'time']).reset_index(drop=True)
        
        return final_df
    
    def format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formatuje dane wyjściowe w czytelny sposób
        
        Args:
            df: DataFrame z surowymi danymi
            
        Returns:
            Sformatowany DataFrame
        """
        
        # Sprawdź jakie zmienne są dostępne
        available_vars = [col for col in df.columns if col in ['t2m', 'tp', 'u100', 'v100']]
        print(f"Dostępne zmienne meteorologiczne: {available_vars}")
        
        # Podstawowe kolumny
        base_cols = ['time', 'requested_lat', 'requested_lon', 'actual_lat', 'actual_lon']
        output_cols = base_cols.copy()
        
        # Dynamicznie dodaj dostępne zmienne
        for var in available_vars:
            output_cols.append(var)
        
        # Wybór dostępnych kolumn
        available_cols = [col for col in output_cols if col in df.columns]
        output_df = df[available_cols].copy()
        
        # Przemianowanie kolumn
        rename_dict = {
            'time': 'datetime',
            'requested_lat': 'lat_requested', 
            'requested_lon': 'lon_requested',
            'actual_lat': 'lat_actual', 
            'actual_lon': 'lon_actual'
        }
        
        if 't2m' in output_df.columns:
            rename_dict['t2m'] = 'temperature_2m_K'
        if 'tp' in output_df.columns:
            rename_dict['tp'] = 'total_precipitation_m'
        if 'u100' in output_df.columns:
            rename_dict['u100'] = 'wind_u_100m_ms'
        if 'v100' in output_df.columns:
            rename_dict['v100'] = 'wind_v_100m_ms'
            
        output_df = output_df.rename(columns=rename_dict)
        
        # Konwersja jednostek - tylko jeśli zmienne istnieją
        if 'temperature_2m_K' in output_df.columns:
            output_df['temperature_2m_C'] = output_df['temperature_2m_K'] - 273.15
        
        if 'total_precipitation_m' in output_df.columns:
            output_df['total_precipitation_mm'] = output_df['total_precipitation_m'] * 1000
        
        # Obliczenie prędkości i kierunku wiatru - tylko jeśli obie składowe istnieją
        if 'wind_u_100m_ms' in output_df.columns and 'wind_v_100m_ms' in output_df.columns:
            output_df['wind_speed_100m_ms'] = np.sqrt(output_df['wind_u_100m_ms']**2 + output_df['wind_v_100m_ms']**2)
            output_df['wind_direction_100m_deg'] = (np.arctan2(output_df['wind_u_100m_ms'], output_df['wind_v_100m_ms']) * 180/np.pi + 180) % 360
        
        return output_df

def parse_coordinates_from_url(url: str) -> Tuple[List[float], List[float], str, str]:
    """
    Parsuje współrzędne i daty z URL w stylu Open-Meteo
    
    Args:
        url: URL zawierający parametry latitude, longitude, start_hour, end_hour
        
    Returns:
        Tuple zawierający listy szerokości i długości geograficznych oraz daty start i end
    """
    from urllib.parse import urlparse, parse_qs
    
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)
    
    latitudes = [float(lat) for lat in params.get('latitude', [])]
    longitudes = [float(lon) for lon in params.get('longitude', [])]
    
    # Parsowanie dat
    start_hour = params.get('start_hour', [''])[0]
    end_hour = params.get('end_hour', [''])[0]
    
    # Dekodowanie URL encoding (%3A = :)
    start_datetime = start_hour.replace('%3A', ':') if start_hour else None
    end_datetime = end_hour.replace('%3A', ':') if end_hour else None
    
    return latitudes, longitudes, start_datetime, end_datetime

def main():
    """
    Przykład użycia
    """
    
    # URL z przykładu - używamy nowego URL podanego przez użytkownika
    example_url = "https://api.open-meteo.com/v1/forecast?latitude=24.25&latitude=24.25&latitude=24.25&latitude=24.5&latitude=24.5&latitude=24.5&latitude=24.75&latitude=24.75&latitude=24.75&longitude=25.0&longitude=25.25&longitude=25.5&longitude=25.0&longitude=25.25&longitude=25.5&longitude=25.0&longitude=25.25&longitude=25.5&hourly=temperature_2m&start_hour=2025-08-05T00%3A00&end_hour=2025-08-05T08%3A00"
    
    # Parsowanie współrzędnych i dat z URL
    latitudes, longitudes, start_datetime, end_datetime = parse_coordinates_from_url(example_url)
    
    print(f"Znalezione współrzędne:")
    print(f"Szerokości geograficzne: {latitudes}")
    print(f"Długości geograficzne: {longitudes}")
    print(f"Okres: {start_datetime} - {end_datetime}")
    
    # Sprawdzenie czy udało się sparsować daty
    if not start_datetime or not end_datetime:
        print("Nie udało się sparsować dat z URL, używam domyślnych...")
        start_datetime = "2024-08-04T15:00"
        end_datetime = "2024-08-04T23:00"
    
    # Dla dat w przyszłości (ERA5 to dane historyczne), użyj poprzedniego roku
    try:
        start_dt = datetime.fromisoformat(start_datetime)
        if start_dt.year >= 2025:
            print(f"UWAGA: ERA5 to dane historyczne. Zmieniam rok z {start_dt.year} na {start_dt.year-1}")
            start_datetime = start_datetime.replace(str(start_dt.year), str(start_dt.year-1))
            end_datetime = end_datetime.replace(str(start_dt.year), str(start_dt.year-1))
    except:
        pass
    
    # Inicjalizacja fetchera
    fetcher = CopernicusERA5WeatherFetcher()
    
    try:
        # Pobranie danych
        netcdf_file = fetcher.fetch_era5_data(
            latitudes=latitudes,
            longitudes=longitudes,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_file="weather_data.nc"
        )
        
        # Przetworzenie danych
        raw_df = fetcher.process_data_for_coordinates(netcdf_file, latitudes, longitudes)
        
        # Formatowanie wyników
        final_df = fetcher.format_output(raw_df)
        
        # Zapisanie do CSV
        output_csv = "weather_data_processed.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"Przetworzone dane zapisane do: {output_csv}")
        
        # Wyświetlenie pierwszych kilku wierszy
        print("\nPierwsze 5 wierszy danych:")
        print(final_df.head())
        
        # Statystyki
        print(f"\nStatystyki:")
        print(f"Liczba punktów: {len(final_df['lat_requested'].unique())}")
        print(f"Liczba znaczników czasowych: {len(final_df['datetime'].unique())}")
        if 'temperature_2m_C' in final_df.columns:
            print(f"Zakres temperatur: {final_df['temperature_2m_C'].min():.1f}°C - {final_df['temperature_2m_C'].max():.1f}°C")
        if 'wind_speed_100m_ms' in final_df.columns:
            print(f"Maksymalna prędkość wiatru: {final_df['wind_speed_100m_ms'].max():.1f} m/s")
        
    except Exception as e:
        print(f"Błąd podczas pobierania lub przetwarzania danych: {e}")
        print("\nUpewnij się, że:")
        print("1. Masz zainstalowane wymagane biblioteki: pip install cdsapi xarray pandas numpy")
        print("2. Masz skonfigurowany dostęp do CDS API (klucz API w ~/.cdsapirc)")
        print("3. Masz aktywne konto w Copernicus Climate Data Store")
        print("4. ERA5 to dane historyczne - dla przyszłych dat użyj poprzedniego roku")

if __name__ == "__main__":
    main()