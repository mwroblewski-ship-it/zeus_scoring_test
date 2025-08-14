import os
import openmeteo_requests
import numpy as np
import torch
import bittensor as bt
from zeus.data.sample import Era5Sample
from zeus.data.converter import get_converter
from zeus.utils.time import to_timestamp
from zeus.validator.constants import (
    OPEN_METEO_URL
)

class OpenMeteoLoader:
    def __init__(
        self,
        open_meteo_url = OPEN_METEO_URL,
    ) -> None:
       
        # Twoje lokalne API nie potrzebuje klucza
        self.open_meteo_url = "https://api.open-meteo.com/v1/forecast"
        self.open_meteo_api = openmeteo_requests.Client()

    def get_output(self, sample: Era5Sample) -> torch.Tensor:
        start_time = to_timestamp(sample.start_timestamp)
        end_time = to_timestamp(sample.end_timestamp)
        
        latitudes, longitudes = sample.x_grid.view(-1, 2).T
        converter = get_converter(sample.variable)
        
        # Używaj formatu dat który działa z Twoim API
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
        }

        # Debugowanie URL-a i parametrów
        self._log_api_request(self.open_meteo_url, params)

        try:
            responses = self.open_meteo_api.weather_api(
                self.open_meteo_url, params=params
            )
            
            # Zbierz wszystkie dane z odpowiedzi
            all_data = []
            for r in responses:
                location_data = []
                for i in range(r.Hourly().VariablesLength()):
                    variable_data = r.Hourly().Variables(i).ValuesAsNumpy()
                    location_data.append(variable_data)
                all_data.append(np.stack(location_data, axis=-1))
            
            # Stack wszystkie lokalizacje
            raw_data = np.stack(all_data, axis=1)  # (time_total, locations, variables)
            
            bt.logging.info(f"Raw data shape from API: {raw_data.shape}")
            bt.logging.info(f"Expected final shape: {(sample.predict_hours, *sample.x_grid.shape[:2])}")
            
            # Oblicz godzinę początkową w dniu
            start_hour = start_time.hour
            
            # Wytnij tylko potrzebne godziny
            if raw_data.shape[0] >= start_hour + sample.predict_hours:
                # Mamy wystarczająco danych - wytnij potrzebny zakres
                sliced_data = raw_data[start_hour:start_hour + sample.predict_hours]
                bt.logging.success(f"Successfully sliced data to shape: {sliced_data.shape}")
            else:
                bt.logging.warning(f"Not enough data: have {raw_data.shape[0]} hours, need {start_hour + sample.predict_hours}")
                # Użyj tyle ile mamy, ewentualnie wypełnij zerami
                available_hours = min(sample.predict_hours, raw_data.shape[0] - start_hour)
                if available_hours > 0:
                    sliced_data = raw_data[start_hour:start_hour + available_hours]
                    # Dopełnij zerami jeśli trzeba
                    if available_hours < sample.predict_hours:
                        padding_shape = (sample.predict_hours - available_hours, *sliced_data.shape[1:])
                        padding = np.zeros(padding_shape)
                        sliced_data = np.concatenate([sliced_data, padding], axis=0)
                else:
                    # Fallback - użyj pierwszych dostępnych godzin
                    sliced_data = raw_data[:sample.predict_hours]
            
            # Reshape do oczekiwanego formatu gridu
            try:
                output = torch.tensor(sliced_data).reshape(
                    sample.predict_hours, *sample.x_grid.shape[:2], -1
                )
                
                # [time, lat, lon] in case of single variable output
                output = output.squeeze(dim=-1)
                
                bt.logging.success(f"Successfully reshaped to: {output.shape}")
                
                # Wypełnij NaN zerami
                if torch.isnan(output).any():
                    nan_count = torch.isnan(output).sum().item()
                    total_count = output.numel()
                    bt.logging.warning(f"⚠️  API returned {nan_count}/{total_count} NaN values - filling with zeros")
                    
                    # Zastąp NaN zerami
                    output = torch.nan_to_num(output, nan=0.0)
                    
                    # Sprawdź czy zostały jakieś NaN
                    remaining_nans = torch.isnan(output).sum().item()
                    if remaining_nans == 0:
                        bt.logging.success(f"✅ Successfully filled all NaN values with zeros")
                    else:
                        bt.logging.error(f"❌ Still have {remaining_nans} NaN values after filling")
                else:
                    bt.logging.success(f"✅ Local API returned valid data for {sample.variable}")
                
                # Convert variable(s) to ERA5 units, combines variables for windspeed
                output = converter.om_to_era5(output)
                
                return output
                
            except Exception as reshape_error:
                bt.logging.error(f"Reshape failed: {reshape_error}")
                bt.logging.error(f"Sliced data shape: {sliced_data.shape}")
                bt.logging.error(f"Expected grid shape: {sample.x_grid.shape}")
                raise reshape_error
            
        except Exception as e:
            bt.logging.error(f"Error in local OpenMeteo API call: {e}")
            # Zwróć tensor wypełniony zerami zamiast NaN
            fallback_shape = (sample.predict_hours, *sample.x_grid.shape[:2])
            bt.logging.warning(f"⚠️  Using zero fallback with shape: {fallback_shape}")
            return torch.zeros(fallback_shape)

    def _log_api_request(self, url: str, params: dict):
        """Loguj szczegóły żądania API bez API key"""
        try:
            import urllib.parse
            
            bt.logging.info(f"🌐 === LOCAL API REQUEST ===")
            bt.logging.info(f"   URL: {url}")
            
            # Pokaż parametry
            for key, value in params.items():
                if isinstance(value, list) and len(value) > 5:
                    bt.logging.info(f"   {key}: [{value[0]}, {value[1]}, {value[2]}, ...] ({len(value)} total)")
                else:
                    bt.logging.info(f"   {key}: {value}")
            
            # Stwórz pełny URL dla debugowania
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query_string}"
            
            bt.logging.info(f"   Full URL: {full_url}")
            bt.logging.info(f"🌐 === END REQUEST ===")
            
        except Exception as e:
            bt.logging.warning(f"Could not log API request details: {e}")
            bt.logging.info(f"🌐 LOCAL API REQUEST: {url}")