import os
import pandas as pd
import torch
import numpy as np
import bittensor as bt
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs
import time
import json
from pathlib import Path

# Import z głównego katalogu projektu
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from fetch_era5_from_url import CopernicusERA5WeatherFetcher, parse_coordinates_from_url

from zeus.data.sample import Era5Sample
from zeus.data.converter import get_converter
from zeus.utils.time import to_timestamp, timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.validator.miner_data import MinerData
from zeus.validator.reward import rmse

class EnhancedERA5Validator:
    """
    Rozszerzony validator który może weryfikować wyniki minerów
    przeciwko prawdziwym danym ERA5 pobranym z URL-a w czasie rzeczywistym
    """
    
    def __init__(self, cache_dir: str = "era5_ground_truth_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.era5_fetcher = CopernicusERA5WeatherFetcher()
        
        # Cache dla ground truth data żeby nie pobierać wielokrotnie
        self.ground_truth_cache: Dict[str, torch.Tensor] = {}
        
    def create_verification_url(self, sample: Era5Sample) -> str:
        """
        Tworzy URL compatible z Open-Meteo format na podstawie Era5Sample
        """
        start_time = to_timestamp(sample.start_timestamp)
        end_time = to_timestamp(sample.end_timestamp)
        
        # Pobierz lokalizacje z gridu
        locations = sample.x_grid
        latitudes = locations[..., 0].flatten().tolist()
        longitudes = locations[..., 1].flatten().tolist()
        
        # Utwórz URL w formacie Open-Meteo
        base_url = "https://api.open-meteo.com/v1/forecast"
        
        # Parametry URL
        lat_params = '&'.join([f'latitude={lat}' for lat in latitudes])
        lon_params = '&'.join([f'longitude={lon}' for lon in longitudes])
        
        url = (f"{base_url}?"
               f"{lat_params}&"
               f"{lon_params}&"
               f"hourly=temperature_2m&"
               f"start_hour={start_time.strftime('%Y-%m-%dT%H:%M')}&"
               f"end_hour={end_time.strftime('%Y-%m-%dT%H:%M')}")
        
        return url
        
    def get_era5_ground_truth(self, sample: Era5Sample) -> Optional[torch.Tensor]:
        """
        Pobiera prawdziwe dane ERA5 dla danego sampla
        """
        # Utwórz unikalne ID dla cache
        cache_key = f"{sample.variable}_{sample.start_timestamp}_{sample.end_timestamp}_{hash(str(sample.get_bbox()))}"
        
        if cache_key in self.ground_truth_cache:
            bt.logging.info(f"📋 Using cached ERA5 ground truth for {cache_key}")
            return self.ground_truth_cache[cache_key]
            
        try:
            # Utwórz URL dla weryfikacji
            verification_url = self.create_verification_url(sample)
            bt.logging.info(f"🌍 Fetching ERA5 ground truth from: {verification_url[:100]}...")
            
            # Parse coordinates z URL
            latitudes, longitudes, start_datetime, end_datetime = parse_coordinates_from_url(verification_url)
            
            if not start_datetime or not end_datetime:
                bt.logging.error("❌ Nie udało się sparsować dat z URL")
                return None
                
            # Pobierz dane ERA5
            bt.logging.info(f"📡 Downloading ERA5 data...")
            netcdf_file = self.era5_fetcher.fetch_era5_data(
                latitudes=latitudes,
                longitudes=longitudes,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_file=f"{self.cache_dir}/ground_truth_{cache_key}.nc"
            )
            
            # Przetwórz dane
            bt.logging.info(f"⚙️ Processing downloaded data...")
            raw_df = self.era5_fetcher.process_data_for_coordinates(netcdf_file, latitudes, longitudes)
            final_df = self.era5_fetcher.format_output(raw_df)
            
            # Konwertuj do tensora w odpowiednim formacie
            ground_truth_tensor = self._convert_df_to_tensor(final_df, sample)
            
            if ground_truth_tensor is not None:
                # Cache wynik
                self.ground_truth_cache[cache_key] = ground_truth_tensor
                bt.logging.success(f"✅ ERA5 ground truth loaded: shape {list(ground_truth_tensor.shape)}")
                
                # Zapisz backup do CSV
                csv_path = f"{self.cache_dir}/ground_truth_{cache_key}.csv"
                final_df.to_csv(csv_path, index=False)
                bt.logging.info(f"💾 Saved ground truth to {csv_path}")
                
            return ground_truth_tensor
            
        except Exception as e:
            bt.logging.error(f"❌ Error fetching ERA5 ground truth: {e}")
            import traceback
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _convert_df_to_tensor(self, df: pd.DataFrame, sample: Era5Sample) -> Optional[torch.Tensor]:
        """
        Konwertuje DataFrame z danymi ERA5 do tensora w formacie [time, lat, lon]
        """
        try:
            # Grupuj według czasu
            time_groups = df.groupby('datetime')
            
            # Sprawdź czy mamy odpowiednią liczbę timestepów
            unique_times = df['datetime'].unique()
            if len(unique_times) != sample.predict_hours:
                bt.logging.warning(f"⚠️ Time mismatch: expected {sample.predict_hours}, got {len(unique_times)}")
            
            # Pobierz zmienną do konwersji
            converter = get_converter(sample.variable)
            
            # Wybierz odpowiednią kolumnę
            if sample.variable == "2m_temperature":
                value_column = 'temperature_2m_K'
            elif sample.variable == "total_precipitation":
                value_column = 'total_precipitation_m'
            elif sample.variable == "100m_u_component_of_wind":
                value_column = 'wind_u_100m_ms'
            elif sample.variable == "100m_v_component_of_wind":
                value_column = 'wind_v_100m_ms'
            else:
                bt.logging.error(f"❌ Unsupported variable: {sample.variable}")
                return None
            
            if value_column not in df.columns:
                bt.logging.error(f"❌ Column {value_column} not found in data")
                bt.logging.error(f"Available columns: {list(df.columns)}")
                return None
            
            # Utwórz tensor
            time_tensors = []
            for time_step in sorted(unique_times):
                time_data = df[df['datetime'] == time_step]
                
                # Sortuj według lat/lon
                time_data = time_data.sort_values(['lat_requested', 'lon_requested'])
                values = time_data[value_column].values
                
                # Reshape do grid format
                grid_shape = sample.x_grid.shape[:2]  # [lat, lon]
                
                if len(values) == np.prod(grid_shape):
                    grid_values = values.reshape(grid_shape)
                    time_tensors.append(torch.tensor(grid_values, dtype=torch.float32))
                else:
                    bt.logging.warning(f"⚠️ Value count mismatch: expected {np.prod(grid_shape)}, got {len(values)}")
                    # Fallback: użyj zeros
                    time_tensors.append(torch.zeros(grid_shape, dtype=torch.float32))
            
            if time_tensors:
                result = torch.stack(time_tensors, dim=0)  # [time, lat, lon]
                bt.logging.success(f"✅ Converted DataFrame to tensor: {list(result.shape)}")
                return result
            else:
                bt.logging.error("❌ No time tensors created")
                return None
                
        except Exception as e:
            bt.logging.error(f"❌ Error converting DataFrame to tensor: {e}")
            import traceback
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    def enhanced_validation(self, sample: Era5Sample, miners_data: List[MinerData], 
                          baseline_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Ulepszona walidacja która porównuje wyniki minerów z prawdziwymi danymi ERA5
        """
        bt.logging.info("🔍" * 40)
        bt.logging.info("🎯 ENHANCED VALIDATION - PRAWDZIWE DANE ERA5")
        bt.logging.info("🔍" * 40)
        
        # Pobierz ground truth z ERA5
        ground_truth = self.get_era5_ground_truth(sample)
        
        validation_results = {
            "ground_truth_available": ground_truth is not None,
            "baseline_vs_ground_truth_rmse": None,
            "miners_vs_ground_truth": {},
            "miners_vs_baseline": {},
            "best_miner_improvement": 0.0
        }
        
        if ground_truth is None:
            bt.logging.warning("❌ Nie udało się pobrać ground truth z ERA5")
            return validation_results
        
        # Statystyki ground truth
        gt_stats = {
            "mean": ground_truth.mean().item(),
            "std": ground_truth.std().item(),
            "min": ground_truth.min().item(),
            "max": ground_truth.max().item()
        }
        bt.logging.info(f"📏 ERA5 Ground Truth Stats:")
        bt.logging.info(f"   Shape: {list(ground_truth.shape)}")
        bt.logging.info(f"   Mean: {gt_stats['mean']:.4f}, Std: {gt_stats['std']:.4f}")
        bt.logging.info(f"   Range: [{gt_stats['min']:.4f}, {gt_stats['max']:.4f}]")
        
        # Porównaj baseline z ground truth
        if baseline_data is not None and baseline_data.shape == ground_truth.shape:
            baseline_rmse = rmse(ground_truth, baseline_data)
            validation_results["baseline_vs_ground_truth_rmse"] = baseline_rmse
            
            bt.logging.info(f"🌐 Baseline vs ERA5 Ground Truth:")
            bt.logging.info(f"   RMSE: {baseline_rmse:.4f}")
            
            # Sprawdź correlation
            baseline_flat = baseline_data.flatten()
            gt_flat = ground_truth.flatten()
            correlation = np.corrcoef(baseline_flat.numpy(), gt_flat.numpy())[0, 1]
            bt.logging.info(f"   Correlation: {correlation:.4f}")
        
        # Porównaj każdego minera z ground truth i baseline
        valid_miners = [m for m in miners_data if not m.shape_penalty]
        
        if not valid_miners:
            bt.logging.warning("❌ Brak prawidłowych minerów do walidacji")
            return validation_results
        
        bt.logging.info(f"⛏️ MINERS ENHANCED VALIDATION ({len(valid_miners)} miners):")
        
        best_gt_rmse = float('inf')
        best_miner_id = None
        
        for miner in valid_miners:
            # RMSE vs ground truth
            gt_rmse = rmse(ground_truth, miner.prediction)
            validation_results["miners_vs_ground_truth"][miner.uid] = gt_rmse
            
            # RMSE vs baseline
            if baseline_data is not None:
                baseline_rmse_miner = rmse(baseline_data, miner.prediction)
                validation_results["miners_vs_baseline"][miner.uid] = baseline_rmse_miner
            
            # Aktualizuj najlepszego minera
            if gt_rmse < best_gt_rmse:
                best_gt_rmse = gt_rmse
                best_miner_id = miner.uid
            
            # Loguj wyniki
            status = "🟢" if gt_rmse < 1.0 else "🟡" if gt_rmse < 3.0 else "🔴"
            bt.logging.info(f"   {status} UID {miner.uid}:")
            bt.logging.info(f"      RMSE vs ERA5: {gt_rmse:.4f}")
            
            if baseline_data is not None:
                improvement_vs_baseline = validation_results["baseline_vs_ground_truth_rmse"] - gt_rmse
                validation_results["best_miner_improvement"] = max(
                    validation_results["best_miner_improvement"], 
                    improvement_vs_baseline
                )
                bt.logging.info(f"      RMSE vs Baseline: {baseline_rmse_miner:.4f}")
                bt.logging.info(f"      Improvement over baseline: {improvement_vs_baseline:.4f}")
        
        # Podsumowanie
        avg_gt_rmse = np.mean(list(validation_results["miners_vs_ground_truth"].values()))
        
        bt.logging.info(f"🏆 Enhanced Validation Summary:")
        bt.logging.info(f"   Best miner UID: {best_miner_id}")
        bt.logging.info(f"   Best RMSE vs ERA5: {best_gt_rmse:.4f}")
        bt.logging.info(f"   Average RMSE vs ERA5: {avg_gt_rmse:.4f}")
        bt.logging.info(f"   Best improvement over baseline: {validation_results['best_miner_improvement']:.4f}")
        
        if validation_results["baseline_vs_ground_truth_rmse"]:
            miners_beating_baseline = sum(1 for rmse_val in validation_results["miners_vs_ground_truth"].values() 
                                        if rmse_val < validation_results["baseline_vs_ground_truth_rmse"])
            bt.logging.info(f"   Miners beating baseline: {miners_beating_baseline}/{len(valid_miners)}")
        
        bt.logging.info("🔍" * 40)
        return validation_results
    
    def save_validation_report(self, sample: Era5Sample, validation_results: Dict, 
                             miners_data: List[MinerData], baseline_data: Optional[torch.Tensor] = None):
        """
        Zapisuje szczegółowy raport walidacji do pliku
        """
        timestamp = int(time.time())
        report_path = self.cache_dir / f"validation_report_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "sample_info": {
                "variable": sample.variable,
                "start_timestamp": sample.start_timestamp,
                "end_timestamp": sample.end_timestamp,
                "predict_hours": sample.predict_hours,
                "bbox": sample.get_bbox(),
                "start_time_str": timestamp_to_str(sample.start_timestamp),
                "end_time_str": timestamp_to_str(sample.end_timestamp),
                "bbox_str": bbox_to_str(sample.get_bbox())
            },
            "validation_results": validation_results,
            "miners_data": [
                {
                    "uid": m.uid,
                    "hotkey": m.hotkey,
                    "reward": m.reward,
                    "rmse": m.rmse,
                    "shape_penalty": m.shape_penalty,
                    "baseline_improvement": getattr(m, 'baseline_improvement', None),
                    "prediction_stats": {
                        "mean": m.prediction.mean().item(),
                        "std": m.prediction.std().item(),
                        "min": m.prediction.min().item(),
                        "max": m.prediction.max().item()
                    }
                }
                for m in miners_data
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        bt.logging.info(f"📄 Validation report saved to: {report_path}")
        return report_path