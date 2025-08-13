#!/usr/bin/env python3
"""
Local Validator Tester for Zeus Subnet
=======================================

Ten program testuje lokalnie logikę validatora bez łączenia z testnetem,
używając historycznych danych ERA5 do natychmiastowego scoringu.

Uruchomienie:
    python local_validator_test.py

Wymagania:
    - Zainstalowany pakiet zeus (pip install -e .)
    - Klucz API dla OpenMeteo (opcjonalny)
    - Klucz API dla CDS (dla danych ERA5)
"""

import asyncio
import os
import sys
import time
import logging
import calendar
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import xarray as xr
from dotenv import load_dotenv

try:
    import cdsapi
except ImportError:
    cdsapi = None

# Importy z Zeus
try:
    from zeus.protocol import TimePredictionSynapse
    from zeus.data.sample import Era5Sample
    from zeus.data.loaders.era5_cds import Era5CDSLoader
    from zeus.data.loaders.openmeteo import OpenMeteoLoader
    from zeus.data.difficulty_loader import DifficultyLoader
    from zeus.validator.reward import set_penalties, set_rewards, rmse
    from zeus.validator.miner_data import MinerData
    from zeus.validator.constants import (
        ERA5_DATA_VARS, 
        REWARD_IMPROVEMENT_MIN_DELTA,
        ERA5_AREA_SAMPLE_RANGE
    )
    from zeus.utils.coordinates import get_grid, bbox_to_str
    from zeus.utils.time import timestamp_to_str, get_today
    from zeus.data.converter import get_converter
    from zeus import __version__ as zeus_version
except ImportError as e:
    print(f"Błąd importu Zeus: {e}")
    print("Upewnij się, że pakiet zeus jest zainstalowany: pip install -e .")
    sys.exit(1)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ImprovedEra5Loader:
    """Ulepszona implementacja ERA5 loader bazująca na działającym kodzie"""
    
    def __init__(self, cache_dir: str = "era5_cache"):
        if cdsapi is None:
            raise RuntimeError("Brak modułu 'cdsapi'. Zainstaluj: pip install cdsapi")
        self.client = cdsapi.Client()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Mapowanie zmiennych Zeus -> ERA5 (ERA5 używa skróconych nazw!)
        self.variable_map = {
            "2m_temperature": "t2m",  # 2-metre temperature
            "total_precipitation": "tp",  # Total precipitation
            "100m_u_component_of_wind": "u100",  # 100m u-component of wind
            "100m_v_component_of_wind": "v100",  # 100m v-component of wind
            "mean_sea_level_pressure": "msl",  # Mean sea level pressure
            "10m_u_component_of_wind": "u10",  # 10m u-component of wind
            "10m_v_component_of_wind": "v10"  # 10m v-component of wind
        }
    
    def daterange_months(self, start_date, end_date):
        """Generator miesięcy w zakresie dat"""
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        
        y, m = start.year, start.month
        while (y < end.year) or (y == end.year and m <= end.month):
            yield y, m
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
    
    def month_bounds_in_range(self, year, month, start_date, end_date):
        """Znajdź granice miesiąca w zakresie dat"""
        first_day = pd.Timestamp(year=year, month=month, day=1).date()
        last_day = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1]).date()
        s = max(first_day, start_date)
        e = min(last_day, end_date)
        return s, e
    
    def fetch_era5_to_netcdf(self, sample: Era5Sample) -> List[str]:
        """Pobierz dane ERA5 do plików NetCDF"""
        start_date = pd.Timestamp(sample.start_timestamp, unit='s').date()
        end_date = pd.Timestamp(sample.end_timestamp, unit='s').date()
        
        era5_variable = self.variable_map.get(sample.variable, sample.variable)
        
        # Rozszerz obszar o mały margines dla interpolacji
        bbox = sample.get_bbox()
        lat_min, lat_max = bbox[0] - 0.25, bbox[1] + 0.25
        lon_min, lon_max = bbox[2] - 0.25, bbox[3] + 0.25
        
        nc_files = []
        
        for year, month in self.daterange_months(start_date, end_date):
            s, e = self.month_bounds_in_range(year, month, start_date, end_date)
            
            fname = f"era5_{era5_variable.replace(' ','_')}_{lat_min:.2f}_{lat_max:.2f}_{lon_min:.2f}_{lon_max:.2f}_{s}_{e}.nc"
            fpath = self.cache_dir / fname
            
            if fpath.exists():
                logger.info(f"Używam cached file: {fpath.name}")
                nc_files.append(str(fpath))
                continue
            
            logger.info(f"Pobieram ERA5 {era5_variable} dla {s}..{e}")
            
            request = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": era5_variable,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(s.day, e.day + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [lat_max, lon_min, lat_min, lon_max],  # north, west, south, east
                "grid": [0.25, 0.25],
            }
            
            tmpfile = str(fpath) + ".tmp"
            
            try:
                self.client.retrieve("reanalysis-era5-single-levels", request, tmpfile)
                os.replace(tmpfile, fpath)
                nc_files.append(str(fpath))
                logger.info(f"Pobrano pomyślnie: {fpath.name}")
            except Exception as e:
                logger.error(f"Błąd pobierania ERA5: {e}")
                if os.path.exists(tmpfile):
                    os.remove(tmpfile)
                raise
        
        return nc_files
    
    def load_era5_grid(self, nc_files: List[str], sample: Era5Sample) -> torch.Tensor:
        """Załaduj dane ERA5 jako grid tensor"""
        try:
            # Otwórz pliki NetCDF
            ds = xr.open_mfdataset(nc_files, combine="by_coords", engine="netcdf4", parallel=False)
            
            era5_variable = self.variable_map.get(sample.variable, sample.variable)
            
            if era5_variable not in ds.variables:
                available_vars = list(ds.variables.keys())
                logger.error(f"Zmienna {era5_variable} nie znaleziona w ERA5. Dostępne: {available_vars}")
                raise ValueError(f"Brak zmiennej {era5_variable} w danych ERA5")
            
            # Wyciągnij dane dla zmiennej
            data_var = ds[era5_variable]
            
            # Filtruj czas
            start_time = pd.Timestamp(sample.start_timestamp, unit='s')
            end_time = pd.Timestamp(sample.end_timestamp, unit='s')
            
            # Uwaga: ERA5 może mieć różne nazwy dla czasu
            time_dim = None
            for dim in ['time', 'valid_time', 'datetime']:
                if dim in data_var.dims:
                    time_dim = dim
                    break
            
            if time_dim is None:
                logger.error(f"Nie znaleziono wymiaru czasu w danych ERA5. Dostępne wymiary: {data_var.dims}")
                raise ValueError("Brak wymiaru czasu w danych ERA5")
            
            # Filtruj czasowo
            time_mask = (data_var[time_dim] >= start_time) & (data_var[time_dim] <= end_time)
            data_filtered = data_var.where(time_mask, drop=True)
            
            if len(data_filtered[time_dim]) == 0:
                logger.error(f"Brak danych czasowych dla zakresu {start_time} -> {end_time}")
                logger.info(f"Dostępny zakres czasu: {data_var[time_dim].min().values} -> {data_var[time_dim].max().values}")
                raise ValueError("Brak danych czasowych w wymaganym zakresie")
            
            # Interpoluj do gridu sample
            grid_lats = sample.x_grid[..., 0].cpu().numpy()
            grid_lons = sample.x_grid[..., 1].cpu().numpy()
            
            # Sprawdź czy grid jest regularny
            unique_lats = np.unique(grid_lats)
            unique_lons = np.unique(grid_lons)
            
            if len(unique_lats) * len(unique_lons) != grid_lats.size:
                logger.error("Grid nie jest regularny - użyj point interpolation")
                # Fallback do interpolacji punktowej
                interpolated_data = []
                for t in range(len(data_filtered[time_dim])):
                    time_slice = data_filtered.isel({time_dim: t})
                    points = []
                    for i in range(grid_lats.shape[0]):
                        for j in range(grid_lats.shape[1]):
                            lat, lon = grid_lats[i, j], grid_lons[i, j]
                            point_val = time_slice.interp(latitude=lat, longitude=lon, method="linear")
                            points.append(float(point_val.values))
                    interpolated_data.append(np.array(points).reshape(grid_lats.shape))
                
                result = torch.tensor(np.array(interpolated_data), dtype=torch.float32)
            else:
                # Regularna interpolacja
                interpolated = data_filtered.interp(
                    latitude=unique_lats, 
                    longitude=unique_lons, 
                    method="linear"
                )
                
                # Konwertuj do torch tensor
                result = torch.tensor(interpolated.values, dtype=torch.float32)
                
                # Sprawdź kolejność wymiarów i dostosuj
                if result.dim() == 3:  # time, lat, lon
                    pass  # Już w dobrym formacie
                elif result.dim() == 4:  # Możliwe dodatkowe wymiary
                    result = result.squeeze()
                else:
                    logger.error(f"Nieoczekiwany kształt danych: {result.shape}")
            
            # Konwersje jednostek jeśli potrzebne
            if sample.variable == "total_precipitation":
                # ERA5 'tp' w m/s, konwertuj do m/h jeśli potrzeba
                result = result * 3600  # m/s -> m/h
            elif sample.variable == "2m_temperature":
                # ERA5 't2m' już w Kelvinach
                pass
            
            logger.info(f"ERA5 pomyślnie załadowane - shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Błąd ładowania ERA5 grid: {e}")
            raise
        finally:
            if 'ds' in locals():
                ds.close()
    
    def get_output(self, sample: Era5Sample) -> torch.Tensor:
        """Główna metoda do pobierania danych ERA5"""
        try:
            logger.info(f"Pobieranie ERA5 dla zmiennej: {sample.variable}")
            logger.info(f"Bbox: {bbox_to_str(sample.get_bbox())}")
            logger.info(f"Czas: {timestamp_to_str(sample.start_timestamp)} -> {timestamp_to_str(sample.end_timestamp)}")
            logger.info(f"Grid shape: {sample.x_grid.shape}")
            
            # Sprawdź czy data nie jest zbyt świeża (ERA5 ma ~5 dni opóźnienia)
            end_time = pd.Timestamp(sample.end_timestamp, unit='s')
            if end_time > pd.Timestamp.now() - pd.Timedelta(days=5):
                logger.warning("Dane mogą być zbyt świeże dla ERA5 (< 5 dni)")
            
            # Pobierz pliki NetCDF
            nc_files = self.fetch_era5_to_netcdf(sample)
            
            if not nc_files:
                logger.error("Brak plików NetCDF do załadowania")
                return torch.empty(0, *sample.x_grid.shape[:2])
            
            # Załaduj jako grid
            result = self.load_era5_grid(nc_files, sample)
            
            return result
            
        except Exception as e:
            logger.error(f"Błąd w ImprovedEra5Loader.get_output: {e}")
            import traceback
            traceback.print_exc()
            # Zwróć pusty tensor zamiast None
            return torch.empty(0, *sample.x_grid.shape[:2])


class LocalMinerSimulator:
    """Symulator minera - zwraca predykcje z różnych źródeł"""
    
    def __init__(self):
        self.has_openmeteo = os.getenv("OPEN_METEO_API_KEY") is not None
        if self.has_openmeteo:
            try:
                self.open_meteo_loader = OpenMeteoLoader()
                logger.info("OpenMeteo API dostępne - używam prawdziwych predykcji")
            except Exception as e:
                logger.warning(f"Błąd inicjalizacji OpenMeteo: {e}")
                self.has_openmeteo = False
        else:
            logger.warning("Brak OpenMeteo API - używam symulowanych danych")
        
    def generate_realistic_temperature(self, sample: Era5Sample) -> torch.Tensor:
        """Generuje realistyczne dane temperatury bez API"""
        coords_shape = sample.x_grid.shape[:2]
        
        # Bazowa temperatura zależna od szerokości geograficznej
        base_temp = 15 - abs(sample.x_grid[..., 0]) * 0.6  # Zimniej na biegunach
        
        # Dodaj sezonowość (uproszczoną)
        day_of_year = pd.Timestamp(sample.start_timestamp, unit='s').dayofyear
        seasonal_variation = 10 * np.cos(2 * np.pi * (day_of_year - 80) / 365)
        
        # Dodaj dzienną cykliczność
        predictions = []
        for hour in range(sample.predict_hours):
            hour_of_day = (pd.Timestamp(sample.start_timestamp, unit='s').hour + hour) % 24
            daily_variation = 5 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
            
            # Dodaj losowy szum
            noise = torch.randn(coords_shape) * 2
            
            # Temperatura w Kelvinach
            temp = base_temp + seasonal_variation + daily_variation + noise + 273.15
            predictions.append(temp)
        
        return torch.stack(predictions)
    
    def generate_realistic_precipitation(self, sample: Era5Sample) -> torch.Tensor:
        """Generuje realistyczne dane opadów"""
        coords_shape = sample.x_grid.shape[:2]
        
        # Większość godzin bez opadów, czasem deszcz
        predictions = []
        for hour in range(sample.predict_hours):
            # 10% szans na opady
            if np.random.random() < 0.1:
                precip = torch.abs(torch.randn(coords_shape)) * 0.001  # m/h
            else:
                precip = torch.zeros(coords_shape)
            predictions.append(precip)
        
        return torch.stack(predictions)
    
    def generate_realistic_wind(self, sample: Era5Sample) -> torch.Tensor:
        """Generuje realistyczne dane wiatru"""
        coords_shape = sample.x_grid.shape[:2]
        
        predictions = []
        for hour in range(sample.predict_hours):
            # Wiatr 2-15 m/s z losowym kierunkiem
            wind_speed = torch.rand(coords_shape) * 13 + 2  # 2-15 m/s
            wind_direction = torch.rand(coords_shape) * 2 * np.pi - np.pi
            
            if "u_component" in sample.variable:
                wind = wind_speed * torch.sin(wind_direction)
            else:  # v_component
                wind = wind_speed * torch.cos(wind_direction)
            
            predictions.append(wind)
        
        return torch.stack(predictions)
        
    async def predict(self, synapse: TimePredictionSynapse) -> torch.Tensor:
        """Symuluje odpowiedź minera"""
        try:
            # Konwertuj synapse do Era5Sample
            coordinates = torch.tensor(synapse.locations)
            sample = Era5Sample(
                variable=synapse.variable,
                start_timestamp=synapse.start_time,
                end_timestamp=synapse.end_time,
                lat_start=coordinates[0, 0, 0].item(),
                lat_end=coordinates[-1, 0, 0].item(),
                lon_start=coordinates[0, 0, 1].item(),
                lon_end=coordinates[0, -1, 1].item(),
                predict_hours=synapse.requested_hours
            )
            
            # Spróbuj użyć OpenMeteo jeśli dostępne
            if self.has_openmeteo:
                try:
                    prediction = self.open_meteo_loader.get_output(sample)
                    logger.info(f"Użyto OpenMeteo API - kształt: {prediction.shape}")
                    return prediction
                except Exception as e:
                    logger.warning(f"OpenMeteo API nie działa: {e}, używam symulowanych danych")
            
            # Fallback do symulowanych danych
            if sample.variable == "2m_temperature":
                prediction = self.generate_realistic_temperature(sample)
            elif sample.variable == "total_precipitation":
                prediction = self.generate_realistic_precipitation(sample)
            elif "wind" in sample.variable:
                prediction = self.generate_realistic_wind(sample)
            else:
                # Ogólny fallback
                coords_shape = sample.x_grid.shape[:2]
                prediction = torch.randn(sample.predict_hours, *coords_shape)
            
            logger.info(f"Użyto symulowanych danych - kształt: {prediction.shape}")
            return prediction
            
        except Exception as e:
            logger.error(f"Błąd w symulatorze minera: {e}")
            # Ostateczny fallback - losowe dane
            coords_shape = torch.tensor(synapse.locations).shape
            return torch.randn(synapse.requested_hours, coords_shape[0], coords_shape[1])


class LocalValidatorTester:
    """Lokalny tester validatora"""
    
    def __init__(self):
        # Załaduj zmienne środowiskowe
        load_dotenv("validator.env")
        
        # Sprawdź klucze API
        self.check_api_keys()
        
        # Inicjalizuj komponenty z fallbackami
        self.has_cds = os.getenv("CDS_API_KEY") is not None
        
        if self.has_cds:
            try:
                # Użyj ulepszonego loadera zamiast oryginalnego
                self.cds_loader = ImprovedEra5Loader()
                logger.info("CDS API dostępne - używam ulepszonego ERA5 loader")
            except Exception as e:
                logger.error(f"Błąd inicjalizacji ulepszonego CDS: {e}")
                # Fallback do oryginalnego
                try:
                    self.cds_loader = Era5CDSLoader()
                    logger.warning("Używam oryginalnego Era5CDSLoader")
                except Exception as e2:
                    logger.error(f"Błąd inicjalizacji oryginalnego CDS: {e2}")
                    self.has_cds = False
        
        if not self.has_cds:
            logger.warning("Brak CDS API - używam symulowanych ground truth")
            self.cds_loader = None
        
        try:
            self.difficulty_loader = DifficultyLoader()
        except Exception as e:
            logger.warning(f"Nie udało się załadować difficulty loader: {e}")
            self.difficulty_loader = None
            
        self.miner_simulator = LocalMinerSimulator()
        
        logger.info("Lokalny tester validatora zainicjalizowany")
    
    def create_small_sample(self, days_back: int = 10) -> Era5Sample:
        """Stwórz mały sample (max 4x4 punkty) dla szybkich testów API"""
        return self.create_historical_sample(days_back, max_grid_size=4)
    
    def generate_fake_ground_truth(self, sample: Era5Sample) -> torch.Tensor:
        """Generuje sztuczne ground truth bez API"""
        coords_shape = sample.x_grid.shape[:2]
        
        if sample.variable == "2m_temperature":
            # Realistyczna temperatura
            base_temp = 15 - abs(sample.x_grid[..., 0]) * 0.5 + 273.15
            return base_temp.unsqueeze(0).repeat(sample.predict_hours, 1, 1) + torch.randn(sample.predict_hours, *coords_shape)
        elif sample.variable == "total_precipitation":
            # Minimalne opady
            return torch.abs(torch.randn(sample.predict_hours, *coords_shape)) * 0.0001
        else:
            # Inne zmienne - losowe ale realistyczne
            return torch.randn(sample.predict_hours, *coords_shape) * 5
    
    def check_api_keys(self):
        """Sprawdź dostępność kluczy API"""
        cds_key = os.getenv("CDS_API_KEY")
        if not cds_key:
            logger.warning("Brak CDS_API_KEY - dane ERA5 mogą być niedostępne")
        else:
            logger.info("CDS_API_KEY znaleziony")
        
        om_key = os.getenv("OPEN_METEO_API_KEY")
        if not om_key:
            logger.warning("Brak OPEN_METEO_API_KEY - używam darmowego API (ograniczenia)")
        else:
            logger.info("OPEN_METEO_API_KEY znaleziony")
    
    def create_historical_sample(self, days_back: int = 10, max_grid_size: int = 6) -> Era5Sample:
        """
        Stwórz sample z danych historycznych (gdzie ERA5 jest już dostępne)
        
        Args:
            days_back: Ile dni wstecz od dzisiaj
            max_grid_size: Maksymalny rozmiar gridu (w punktach 0.25°) dla API
        """
        # Użyj daty sprzed kilku dni, gdzie ERA5 jest już dostępne (min 7 dni wstecz)
        min_days_back = max(7, days_back)
        base_date = get_today("h") - pd.Timedelta(days=min_days_back)
        
        # Ograniczony rozmiar gridu dla API (2-4 punkty)
        grid_size_lat = min(max_grid_size, np.random.randint(2, 5))
        grid_size_lon = min(max_grid_size, np.random.randint(2, 5))
        
        # Losuj parametry z ograniczonym rozmiarem i zaokrągleniem do 0.25°
        lat_center = np.random.uniform(-60, 60)  # Unikaj biegunów
        lon_center = np.random.uniform(-160, 160)
        
        # Zaokrągl do 0.25° gridu
        lat_start = round(lat_center * 4) / 4
        lon_start = round(lon_center * 4) / 4
        
        # Dodaj grid size z zachowaniem 0.25° resolution
        lat_end = lat_start + ((grid_size_lat - 1) * 0.25)
        lon_end = lon_start + ((grid_size_lon - 1) * 0.25)
        
        logger.info(f"Grid size: {grid_size_lat}x{grid_size_lon} punktów ({lat_end-lat_start:.2f}°x{lon_end-lon_start:.2f}°)")
        
        # Predict hours (1-12 dla szybszych testów)
        predict_hours = np.random.randint(1, 13)
        
        start_time = base_date + pd.Timedelta(hours=np.random.randint(0, 24))
        end_time = start_time + pd.Timedelta(hours=predict_hours - 1)
        
        # Wybierz losową zmienną
        variables = list(ERA5_DATA_VARS.keys())
        variable = np.random.choice(variables)
        
        sample = Era5Sample(
            variable=variable,
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            predict_hours=predict_hours
        )
        
        # Debug info
        logger.info(f"Utworzono historyczny sample:")
        logger.info(f"  Zmienna: {variable}")
        logger.info(f"  Bbox: {bbox_to_str(sample.get_bbox())}")
        logger.info(f"  Czas: {timestamp_to_str(sample.start_timestamp)} -> {timestamp_to_str(sample.end_timestamp)}")
        logger.info(f"  Godziny predykcji: {predict_hours}")
        logger.info(f"  Grid shape expected: {sample.x_grid.shape}")
        
        return sample
    
    async def run_test_challenge(self, sample: Optional[Era5Sample] = None) -> dict:
        """
        Uruchom jeden test challenge z natychmiastowym scoringiem
        
        Args:
            sample: Opcjonalny sample do testowania (jeśli None, zostanie wygenerowany)
        
        Returns:
            dict: Wyniki testu zawierające metryki
        """
        if sample is None:
            sample = self.create_historical_sample()
        
        logger.info("=" * 60)
        logger.info("ROZPOCZYNANIE TESTU CHALLENGE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. Pobierz ground truth z ERA5 lub wygeneruj
        logger.info("Pobieranie ground truth...")
        try:
            if self.has_cds and self.cds_loader:
                ground_truth = self.cds_loader.get_output(sample)
                
                if ground_truth is None or ground_truth.shape[0] == 0:
                    logger.error("ERA5 zwróciło pusty tensor - używam symulowanych danych")
                    ground_truth = self.generate_fake_ground_truth(sample)
                else:
                    logger.info(f"Użyto prawdziwych danych ERA5 - shape: {ground_truth.shape}")
                    
                    # Sprawdź czy shape się zgadza z oczekiwaniami
                    expected_shape = (sample.predict_hours, *sample.x_grid.shape[:2])
                    if ground_truth.shape != expected_shape:
                        logger.warning(f"Niezgodny kształt ERA5: {ground_truth.shape}, oczekiwano: {expected_shape}")
                        # Spróbuj dopasować rozmiar
                        if ground_truth.shape[0] == sample.predict_hours:
                            logger.info("Próbuję interpolować do właściwego rozmiaru gridu")
                            # TODO: Dodaj interpolację jeśli potrzebna
                        else:
                            logger.error("Błędna liczba godzin - używam symulowanych danych")
                            ground_truth = self.generate_fake_ground_truth(sample)
            else:
                logger.info("Używam symulowanych ground truth")
                ground_truth = self.generate_fake_ground_truth(sample)
                
            sample.output_data = ground_truth
            
        except Exception as e:
            logger.error(f"Błąd pobierania ground truth: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Używam symulowanych danych")
            ground_truth = self.generate_fake_ground_truth(sample)
            sample.output_data = ground_truth
        
        # 2. Pobierz baseline z OpenMeteo
        logger.info("Pobieranie baseline z OpenMeteo...")
        baseline = None
        try:
            if self.miner_simulator.has_openmeteo:
                baseline = self.miner_simulator.open_meteo_loader.get_output(sample)
                logger.info(f"Baseline shape: {baseline.shape}")
                
                # Sprawdź zgodność kształtów
                if baseline.shape != ground_truth.shape:
                    logger.warning(f"Niezgodny kształt baseline: {baseline.shape} vs ground truth: {ground_truth.shape}")
                    # Spróbuj dopasować lub zrezygnuj
                    if baseline.shape[0] == ground_truth.shape[0]:
                        # Tylko różnica w grid size - można interpolować
                        logger.info("Różne rozmiary gridu - dopasowuję baseline")
                        # TODO: Dodaj interpolację
                    else:
                        logger.warning("Rezygnuję z baseline z powodu niezgodnych kształtów")
                        baseline = None
        except Exception as e:
            logger.warning(f"Błąd pobierania OpenMeteo baseline: {e}")
            baseline = None
        
        # 3. Symuluj odpowiedzi minerów
        logger.info("Symulowanie odpowiedzi minerów...")
        num_miners = 5  # Symuluj 5 minerów
        miner_predictions = []
        
        for i in range(num_miners):
            try:
                if i == 0 and baseline is not None:
                    # Pierwszy miner zwraca baseline (OpenMeteo)
                    if baseline.shape == ground_truth.shape:
                        prediction = baseline
                    else:
                        prediction = torch.randn_like(ground_truth)
                elif i == 1:
                    # Drugi miner zwraca ground truth + szum (symuluje dobrego minera)
                    noise = torch.randn_like(ground_truth) * 0.5
                    prediction = ground_truth + noise
                elif i == 2:
                    # Trzeci miner zwraca złą odpowiedź (zły kształt)
                    prediction = torch.randn(5, 5, 5)  # Zły kształt
                else:
                    # Pozostali minerzy zwracają losowe predykcje
                    prediction = torch.randn_like(ground_truth) * 10 + ground_truth.mean()
                
                miner_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Błąd w symulacji minera {i}: {e}")
                miner_predictions.append(torch.randn_like(ground_truth))
        
        # 4. Stwórz MinerData
        miners_data = []
        for i, prediction in enumerate(miner_predictions):
            miner_data = MinerData(
                uid=i,
                hotkey=f"fake_hotkey_{i}",
                prediction=prediction
            )
            miners_data.append(miner_data)
        
        # 5. Oblicz penalties
        logger.info("Obliczanie penalties...")
        try:
            miners_data = set_penalties(ground_truth, miners_data)
        except Exception as e:
            logger.error(f"Błąd obliczania penalties: {e}")
            # Ręczne penalties
            for miner_data in miners_data:
                if miner_data.prediction.shape != ground_truth.shape:
                    miner_data.shape_penalty = True
                    miner_data.rmse = -1.0
                    miner_data.reward = 0.0
        
        # 6. Oblicz rewards
        logger.info("Obliczanie rewards...")
        try:
            if self.difficulty_loader:
                difficulty_grid = self.difficulty_loader.get_difficulty_grid(sample)
            else:
                # Fallback difficulty grid - średnia trudność
                coords_shape = sample.x_grid.shape[:2]
                difficulty_grid = np.ones(coords_shape) * 0.5
                logger.info("Używam domyślnej trudności (brak difficulty loader)")
                
            miners_data = set_rewards(
                output_data=ground_truth,
                miners_data=miners_data,
                baseline_data=baseline,
                difficulty_grid=difficulty_grid,
                min_sota_delta=REWARD_IMPROVEMENT_MIN_DELTA.get(sample.variable, 0.1)
            )
        except Exception as e:
            logger.error(f"Błąd obliczania rewards: {e}")
            import traceback
            traceback.print_exc()
            # Ustaw podstawowe nagrody w przypadku błędu
            for miner_data in miners_data:
                if not getattr(miner_data, 'shape_penalty', False):
                    try:
                        miner_data.rmse = rmse(ground_truth, miner_data.prediction)
                        miner_data.reward = max(0, 1.0 - miner_data.rmse / 10.0)
                    except Exception as e2:
                        logger.error(f"Błąd obliczania RMSE dla minera {miner_data.uid}: {e2}")
                        miner_data.rmse = float('nan')
                        miner_data.reward = 0.0
        
        # 7. Podsumowanie wyników
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("WYNIKI TESTU")
        logger.info("=" * 60)
        
        baseline_rmse = "N/A"
        if baseline is not None:
            try:
                baseline_rmse = rmse(ground_truth, baseline)
            except Exception as e:
                logger.warning(f"Błąd obliczania baseline RMSE: {e}")
                baseline_rmse = "ERROR"
        
        logger.info(f"Baseline RMSE (OpenMeteo): {baseline_rmse}")
        logger.info(f"Czas wykonania: {total_time:.2f}s")
        logger.info("")
        
        results = {
            "sample": {
                "variable": sample.variable,
                "bbox": sample.get_bbox(),
                "start_time": timestamp_to_str(sample.start_timestamp),
                "end_time": timestamp_to_str(sample.end_timestamp),
                "predict_hours": sample.predict_hours,
                "ground_truth_shape": list(ground_truth.shape),
                "baseline_shape": list(baseline.shape) if baseline is not None else None
            },
            "baseline_rmse": baseline_rmse,
            "execution_time": total_time,
            "miners": []
        }
        
        for miner in miners_data:
            miner_result = {
                "uid": miner.uid,
                "hotkey": miner.hotkey,
                "prediction_shape": list(miner.prediction.shape),
                "rmse": getattr(miner, 'rmse', None),
                "reward": getattr(miner, 'reward', None),
                "shape_penalty": getattr(miner, 'shape_penalty', False),
                "baseline_improvement": getattr(miner, 'baseline_improvement', None)
            }
            results["miners"].append(miner_result)
            
            rmse_val = getattr(miner, 'rmse', None)
            reward_val = getattr(miner, 'reward', None)
            shape_penalty = getattr(miner, 'shape_penalty', False)
            
            logger.info(f"Miner {miner.uid}:")
            if shape_penalty:
                logger.info(f"  RMSE: {rmse_val} (shape penalty)")
                logger.info(f"  Reward: {reward_val}")
            else:
                logger.info(f"  RMSE: {rmse_val:.4f}" if rmse_val and not np.isnan(rmse_val) else f"  RMSE: {rmse_val}")
                logger.info(f"  Reward: {reward_val:.4f}" if reward_val and not np.isnan(reward_val) else f"  Reward: {reward_val}")
            
            logger.info(f"  Shape penalty: {shape_penalty}")
            if hasattr(miner, 'baseline_improvement') and miner.baseline_improvement:
                logger.info(f"  Baseline improvement: {miner.baseline_improvement:.4f}")
        
        return results
    
    async def run_continuous_test(self, num_tests: int = 5, delay: int = 10):
        """
        Uruchom ciągłe testy
        
        Args:
            num_tests: Liczba testów do uruchomienia
            delay: Opóźnienie między testami w sekundach
        """
        logger.info(f"Rozpoczynanie {num_tests} testów z opóźnieniem {delay}s...")
        
        all_results = []
        
        for test_num in range(num_tests):
            logger.info(f"\n{'='*20} TEST {test_num + 1}/{num_tests} {'='*20}")
            
            try:
                result = await self.run_test_challenge()
                all_results.append(result)
                
                if test_num < num_tests - 1:  # Nie czekaj po ostatnim teście
                    logger.info(f"Czekanie {delay}s do następnego testu...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Błąd w teście {test_num + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        # Podsumowanie wszystkich testów
        self.print_summary(all_results)
    
    def print_summary(self, results: List[dict]):
        """Wydrukuj podsumowanie wszystkich testów"""
        if not results:
            logger.info("Brak wyników do podsumowania")
            return
        
        logger.info("\n" + "="*60)
        logger.info("PODSUMOWANIE WSZYSTKICH TESTÓW")
        logger.info("="*60)
        
        # Średnie czasy wykonania
        avg_time = np.mean([r["execution_time"] for r in results if "execution_time" in r])
        logger.info(f"Średni czas wykonania: {avg_time:.2f}s")
        
        # Statystyki RMSE
        all_rmses = []
        all_rewards = []
        for result in results:
            if "miners" in result:
                for miner in result["miners"]:
                    rmse_val = miner.get("rmse")
                    reward_val = miner.get("reward")
                    if rmse_val and not np.isnan(rmse_val) and rmse_val > 0:
                        all_rmses.append(rmse_val)
                    if reward_val and not np.isnan(reward_val):
                        all_rewards.append(reward_val)
        
        if all_rmses:
            logger.info(f"RMSE - min: {min(all_rmses):.4f}, max: {max(all_rmses):.4f}, śr: {np.mean(all_rmses):.4f}")
        if all_rewards:
            logger.info(f"Rewards - min: {min(all_rewards):.4f}, max: {max(all_rewards):.4f}, śr: {np.mean(all_rewards):.4f}")
        
        # Zapisz wyniki do pliku
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Wyniki zapisane do: {filename}")
        except Exception as e:
            logger.error(f"Nie udało się zapisać wyników: {e}")


class InteractiveMenu:
    """Interaktywne menu dla testowania"""
    
    def __init__(self):
        self.tester = LocalValidatorTester()
    
    async def run(self):
        """Uruchom interaktywne menu"""
        while True:
            self.print_menu()
            choice = input("\nWybierz opcję (1-5): ").strip()
            
            try:
                if choice == "1":
                    await self.single_test()
                elif choice == "2":
                    await self.continuous_test()
                elif choice == "3":
                    await self.custom_test()
                elif choice == "4":
                    await self.quick_test()
                elif choice == "5":
                    logger.info("Zakończenie programu...")
                    break
                else:
                    print("Nieprawidłowy wybór. Spróbuj ponownie.")
            except KeyboardInterrupt:
                logger.info("\nOperacja przerwana przez użytkownika")
            except Exception as e:
                logger.error(f"Błąd: {e}")
                import traceback
                traceback.print_exc()
    
    def print_menu(self):
        """Wydrukuj menu opcji"""
        print("\n" + "="*50)
        print("LOKALNY TESTER VALIDATORA ZEUS")
        print("="*50)
        print("1. Pojedynczy test")
        print("2. Testy ciągłe")
        print("3. Test z własnymi parametrami")
        print("4. Szybki test (mały grid, bez API)")
        print("5. Wyjście")
        print("="*50)
    
    async def single_test(self):
        """Uruchom pojedynczy test"""
        logger.info("Uruchamianie pojedynczego testu...")
        await self.tester.run_test_challenge()
    
    async def continuous_test(self):
        """Uruchom testy ciągłe"""
        try:
            num_tests = int(input("Liczba testów (domyślnie 5): ").strip() or "5")
            delay = int(input("Opóźnienie między testami w sekundach (domyślnie 10): ").strip() or "10")
            await self.tester.run_continuous_test(num_tests, delay)
        except ValueError:
            logger.error("Nieprawidłowe wartości liczbowe")
    
    async def custom_test(self):
        """Uruchom test z własnymi parametrami"""
        try:
            print("\nParametry custom testu:")
            days_back = int(input("Dni wstecz (domyślnie 15): ").strip() or "15")
            variable = input(f"Zmienna {list(ERA5_DATA_VARS.keys())} (domyślnie 2m_temperature): ").strip() or "2m_temperature"
            
            if variable not in ERA5_DATA_VARS:
                logger.error(f"Nieznana zmienna: {variable}")
                return
            
            # Dodaj opcję ograniczenia rozmiaru gridu
            max_grid_size = int(input("Maksymalny rozmiar gridu w punktach (domyślnie 4 dla mniejszych requestów API): ").strip() or "4")
            
            use_custom_coords = input("Użyć własnych współrzędnych? (y/N): ").strip().lower() == 'y'
            
            if use_custom_coords:
                lat_start = float(input("Latitude start: ").strip())
                lat_end = float(input("Latitude end: ").strip())
                lon_start = float(input("Longitude start: ").strip())
                lon_end = float(input("Longitude end: ").strip())
                
                # Zaokrągl do 0.25° gridu
                lat_start = round(lat_start * 4) / 4
                lat_end = round(lat_end * 4) / 4
                lon_start = round(lon_start * 4) / 4
                lon_end = round(lon_end * 4) / 4
                
                # Sprawdź rozmiar gridu
                grid_lat_size = int((lat_end - lat_start) / 0.25) + 1
                grid_lon_size = int((lon_end - lon_start) / 0.25) + 1
                
                if grid_lat_size > max_grid_size or grid_lon_size > max_grid_size:
                    logger.warning(f"Grid będzie duży: {grid_lat_size}x{grid_lon_size} punktów")
                    confirm = input("Kontynuować? (y/N): ").strip().lower()
                    if confirm != 'y':
                        return
            else:
                # Automatycznie wygeneruj małe współrzędne
                lat_center = np.random.uniform(-60, 60)
                lon_center = np.random.uniform(-160, 160)
                
                lat_start = round(lat_center * 4) / 4
                lon_start = round(lon_center * 4) / 4
                lat_end = lat_start + ((max_grid_size - 1) * 0.25)
                lon_end = lon_start + ((max_grid_size - 1) * 0.25)
            
            predict_hours = int(input("Godziny predykcji (1-24, domyślnie 6): ").strip() or "6")
            
            # Sprawdź zakres
            if not (1 <= predict_hours <= 24):
                logger.error("Godziny predykcji muszą być między 1 a 24")
                return
            
            # Stwórz custom sample z bezpieczną datą (min 7 dni wstecz)
            min_days_back = max(7, days_back)
            base_date = get_today("h") - pd.Timedelta(days=min_days_back)
            start_time = base_date + pd.Timedelta(hours=np.random.randint(0, 24))
            end_time = start_time + pd.Timedelta(hours=predict_hours - 1)
            
            sample = Era5Sample(
                variable=variable,
                start_timestamp=start_time.timestamp(),
                end_timestamp=end_time.timestamp(),
                lat_start=lat_start,
                lat_end=lat_end,
                lon_start=lon_start,
                lon_end=lon_end,
                predict_hours=predict_hours
            )
            
            logger.info(f"Utworzono custom sample:")
            logger.info(f"  Grid: {sample.x_grid.shape}")
            logger.info(f"  Bbox: {bbox_to_str(sample.get_bbox())}")
            
            await self.tester.run_test_challenge(sample)
            
        except ValueError:
            logger.error("Nieprawidłowe wartości liczbowe")
        except Exception as e:
            logger.error(f"Błąd w custom test: {e}")
            import traceback
            traceback.print_exc()
    
    async def quick_test(self):
        """Uruchom szybki test z małym gridem i bez API"""
        logger.info("Uruchamianie szybkiego testu (bez API, mały grid)...")
        # Wymuś użycie małego gridu i symulowanych danych
        original_has_cds = self.tester.has_cds
        original_has_om = self.tester.miner_simulator.has_openmeteo
        
        self.tester.has_cds = False
        self.tester.miner_simulator.has_openmeteo = False
        
        try:
            sample = self.tester.create_small_sample()
            await self.tester.run_test_challenge(sample)
        finally:
            # Przywróć oryginalne ustawienia
            self.tester.has_cds = original_has_cds
            self.tester.miner_simulator.has_openmeteo = original_has_om


def main():
    """Główna funkcja programu"""
    print("Lokalny Tester Validatora Zeus")
    print("==============================")
    print(f"Zeus version: {zeus_version}")
    print(f"Torch version: {torch.__version__}")
    print(f"Current time: {datetime.now()}")
    print("")
    
    # Sprawdź czy jesteśmy w odpowiednim katalogu
    if not os.path.exists("zeus"):
        print("Błąd: Nie znaleziono katalogu 'zeus'")
        print("Upewnij się, że uruchamiasz program z głównego katalogu Zeus")
        sys.exit(1)
    
    # Sprawdź czy .env istnieje
    if not os.path.exists("validator.env"):
        logger.warning("Nie znaleziono pliku validator.env")
        logger.info("Tworząc przykładowy plik validator.env...")
        with open("validator.env", "w") as f:
            f.write("# API Keys dla Zeus Validator\n")
            f.write("CDS_API_KEY=your_cds_api_key_here\n")
            f.write("OPEN_METEO_API_KEY=your_openmeteo_api_key_here\n")
        logger.info("Edytuj validator.env i dodaj swoje klucze API")
    
    try:
        menu = InteractiveMenu()
        asyncio.run(menu.run())
    except KeyboardInterrupt:
        logger.info("\nProgram przerwany przez użytkownika")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()