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
from fetch_era5_from_url import CopernicusERA5WeatherFetcher

from zeus.data.sample import Era5Sample
from zeus.data.converter import get_converter
from zeus.utils.time import to_timestamp, timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.validator.miner_data import MinerData
from zeus.validator.reward import rmse

class EnhancedERA5Validator:
    """
    Rozszerzony validator który może weryfikować wyniki minerów
    przeciwko prawdziwym danym ERA5 pobranym w czasie rzeczywistym
    """
    
    def __init__(self, cache_dir: str = "era5_ground_truth_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.era5_fetcher = CopernicusERA5WeatherFetcher()
        
        # Cache dla ground truth data
        self.ground_truth_cache: Dict[str, torch.Tensor] = {}
        
        # Mapowanie zmiennych Zeus -> ERA5
        self.variable_mapping = {
            "2m_temperature": "2m_temperature",
            "total_precipitation": "total_precipitation", 
            "100m_u_component_of_wind": "100m_u_component_of_wind",
            "100m_v_component_of_wind": "100m_v_component_of_wind"
        }
        
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
            # Konwertuj timestamps na datetime
            start_dt = pd.Timestamp(sample.start_timestamp, unit='s')
            end_dt = pd.Timestamp(sample.end_timestamp, unit='s')
            
            bt.logging.info(f"🌍 Fetching ERA5 ground truth...")
            bt.logging.info(f"   Variable: {sample.variable}")
            bt.logging.info(f"   Time range: {start_dt} -> {end_dt}")
            bt.logging.info(f"   Location: {bbox_to_str(sample.get_bbox())}")
            
            # **KLUCZOWA POPRAWKA**: Sprawdź czy daty nie są w przyszłości
            now = pd.Timestamp.now()
            if start_dt > now:
                bt.logging.warning(f"❌ Cannot fetch ERA5 for future dates!")
                bt.logging.warning(f"   Requested: {start_dt}")
                bt.logging.warning(f"   Current: {now}")
                bt.logging.info(f"💡 Adjusting dates to historical period for testing...")
                
                # Przesuń daty do przeszłości (np. rok wcześniej)
                days_diff = (start_dt - now).days
                start_dt = start_dt - pd.Timedelta(days=days_diff + 30)  # Dodaj bufor
                end_dt = end_dt - pd.Timedelta(days=days_diff + 30)
                
                bt.logging.info(f"   Adjusted to: {start_dt} -> {end_dt}")
            
            # Pobierz współrzędne z grid
            locations = sample.x_grid
            latitudes = locations[..., 0].flatten().tolist()
            longitudes = locations[..., 1].flatten().tolist()
            
            bt.logging.info(f"📍 Grid locations: {len(latitudes)} points")
            bt.logging.info(f"   Lat range: [{min(latitudes):.2f}, {max(latitudes):.2f}]")
            bt.logging.info(f"   Lon range: [{min(longitudes):.2f}, {max(longitudes):.2f}]")
            
            # **POPRAWKA**: Użyj poprawnych formatów dat
            start_datetime = start_dt.strftime('%Y-%m-%dT%H:%M')
            end_datetime = end_dt.strftime('%Y-%m-%dT%H:%M')
            
            bt.logging.info(f"📡 Downloading ERA5 data...")
            bt.logging.info(f"   Start: {start_datetime}")
            bt.logging.info(f"   End: {end_datetime}")
            
            # Pobierz dane ERA5
            netcdf_file = self.era5_fetcher.fetch_era5_data(
                latitudes=latitudes,
                longitudes=longitudes,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                output_file=f"{self.cache_dir}/ground_truth_{cache_key}.nc"
            )
            
            if not os.path.exists(netcdf_file):
                bt.logging.error(f"❌ ERA5 file not downloaded: {netcdf_file}")
                return None
            
            # Przetwórz dane
            bt.logging.info(f"⚙️ Processing downloaded data...")
            raw_df = self.era5_fetcher.process_data_for_coordinates(netcdf_file, latitudes, longitudes)
            
            if raw_df.empty:
                bt.logging.error(f"❌ No data returned from ERA5 processing")
                return None
                
            final_df = self.era5_fetcher.format_output(raw_df)
            
            bt.logging.info(f"📊 ERA5 DataFrame info:")
            bt.logging.info(f"   Shape: {final_df.shape}")
            bt.logging.info(f"   Columns: {list(final_df.columns)}")
            bt.logging.info(f"   Unique times: {len(final_df['datetime'].unique())}")
            bt.logging.info(f"   Unique locations: {len(final_df['lat_requested'].unique())}")
            
            # Konwertuj do tensora w odpowiednim formacie
            ground_truth_tensor = self._convert_df_to_tensor(final_df, sample)
            
            if ground_truth_tensor is not None:
                # Cache wynik
                self.ground_truth_cache[cache_key] = ground_truth_tensor
                bt.logging.success(f"✅ ERA5 ground truth loaded: shape {list(ground_truth_tensor.shape)}")
                
                # Sprawdź basic stats
                gt_stats = {
                    "mean": ground_truth_tensor.mean().item(),
                    "std": ground_truth_tensor.std().item(),
                    "min": ground_truth_tensor.min().item(),
                    "max": ground_truth_tensor.max().item()
                }
                bt.logging.info(f"📏 Ground Truth Stats: mean={gt_stats['mean']:.4f}, std={gt_stats['std']:.4f}, range=[{gt_stats['min']:.4f}, {gt_stats['max']:.4f}]")
                
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
            bt.logging.info(f"🔄 Converting DataFrame to tensor...")
            
            # **POPRAWKA**: Lepsze mapowanie kolumn
            column_mapping = {
                "2m_temperature": "temperature_2m_K",
                "total_precipitation": "total_precipitation_m", 
                "100m_u_component_of_wind": "wind_u_100m_ms",
                "100m_v_component_of_wind": "wind_v_100m_ms"
            }
            
            if sample.variable not in column_mapping:
                bt.logging.error(f"❌ Unsupported variable: {sample.variable}")
                bt.logging.error(f"Available variables: {list(column_mapping.keys())}")
                return None
                
            value_column = column_mapping[sample.variable]
            
            if value_column not in df.columns:
                bt.logging.error(f"❌ Column {value_column} not found in data")
                bt.logging.error(f"Available columns: {list(df.columns)}")
                return None
            
            # Sortuj DataFrame
            df_sorted = df.sort_values(['datetime', 'lat_requested', 'lon_requested']).copy()
            
            # Grupuj według czasu
            unique_times = sorted(df_sorted['datetime'].unique())
            unique_lats = sorted(df_sorted['lat_requested'].unique()) 
            unique_lons = sorted(df_sorted['lon_requested'].unique())
            
            bt.logging.info(f"   Time steps: {len(unique_times)}")
            bt.logging.info(f"   Unique lats: {len(unique_lats)}")
            bt.logging.info(f"   Unique lons: {len(unique_lons)}")
            bt.logging.info(f"   Expected shape: [{len(unique_times)}, {len(unique_lats)}, {len(unique_lons)}]")
            bt.logging.info(f"   Sample shape: {list(sample.x_grid.shape)}")
            
            # **POPRAWKA**: Zbuduj tensor step by step
            time_tensors = []
            
            for i, time_step in enumerate(unique_times):
                time_data = df_sorted[df_sorted['datetime'] == time_step].copy()
                
                if time_data.empty:
                    bt.logging.warning(f"⚠️ No data for time step {time_step}")
                    continue
                
                # Stwórz grid dla tego timestep
                grid_values = np.full((len(unique_lats), len(unique_lons)), np.nan)
                
                for _, row in time_data.iterrows():
                    try:
                        lat_idx = unique_lats.index(row['lat_requested'])
                        lon_idx = unique_lons.index(row['lon_requested'])
                        grid_values[lat_idx, lon_idx] = row[value_column]
                    except (ValueError, KeyError) as e:
                        bt.logging.warning(f"⚠️ Skipping point due to error: {e}")
                        continue
                
                # Sprawdź czy mamy kompletne dane
                nan_count = np.isnan(grid_values).sum()
                total_points = grid_values.size
                
                if nan_count > 0:
                    bt.logging.warning(f"⚠️ Time {i}: {nan_count}/{total_points} points are NaN")
                    # Wypełnij NaN średnią
                    valid_values = grid_values[~np.isnan(grid_values)]
                    if len(valid_values) > 0:
                        mean_val = valid_values.mean()
                        grid_values[np.isnan(grid_values)] = mean_val
                        bt.logging.info(f"   Filled NaN with mean: {mean_val:.4f}")
                
                time_tensors.append(torch.tensor(grid_values, dtype=torch.float32))
            
            if not time_tensors:
                bt.logging.error("❌ No valid time tensors created")
                return None
            
            result = torch.stack(time_tensors, dim=0)  # [time, lat, lon]
            
            bt.logging.success(f"✅ Successfully converted to tensor: {list(result.shape)}")
            
            # Validacja końcowa
            if torch.isnan(result).any():
                nan_count = torch.isnan(result).sum().item()
                bt.logging.warning(f"⚠️ Result tensor contains {nan_count} NaN values")
                result = torch.nan_to_num(result, nan=0.0)
                bt.logging.info(f"   Replaced NaN with zeros")
            
            return result
                
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
        
        # **POPRAWKA**: Zawsze próbuj pobrać ground truth
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
            bt.logging.info("🔄 Enhanced validation will fall back to standard method")
            return validation_results
        
        bt.logging.success(f"✅ ERA5 Ground Truth successfully loaded!")
        
        # **POPRAWKA**: Aktualizuj sample z prawdziwym ground truth
        original_gt = sample.output_data
        sample.output_data = ground_truth
        
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
        
        # Porównaj każdego minera z ground truth
        valid_miners = [m for m in miners_data if not m.shape_penalty]
        
        if not valid_miners:
            bt.logging.warning("❌ Brak prawidłowych minerów do walidacji")
            return validation_results
        
        bt.logging.info(f"⛏️ MINERS ENHANCED VALIDATION ({len(valid_miners)} miners):")
        
        best_gt_rmse = float('inf')
        best_miner_id = None
        
        for miner in valid_miners:
            # **POPRAWKA**: Sprawdź shape compatibility
            if miner.prediction.shape != ground_truth.shape:
                bt.logging.warning(f"⚠️ UID {miner.uid}: Shape mismatch {list(miner.prediction.shape)} vs {list(ground_truth.shape)}")
                continue
                
            # RMSE vs ground truth
            gt_rmse = rmse(ground_truth, miner.prediction)
            validation_results["miners_vs_ground_truth"][miner.uid] = gt_rmse
            
            # RMSE vs baseline
            if baseline_data is not None and miner.prediction.shape == baseline_data.shape:
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
            
            if baseline_data is not None and validation_results["baseline_vs_ground_truth_rmse"]:
                improvement_vs_baseline = validation_results["baseline_vs_ground_truth_rmse"] - gt_rmse
                validation_results["best_miner_improvement"] = max(
                    validation_results["best_miner_improvement"], 
                    improvement_vs_baseline
                )
                bt.logging.info(f"      Improvement over baseline: {improvement_vs_baseline:.4f}")
        
        # Podsumowanie
        if validation_results["miners_vs_ground_truth"]:
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