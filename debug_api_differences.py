#!/usr/bin/env python3
"""
Debug script to find why miner and validator get different OpenMeteo results
"""

import time
import torch
import numpy as np
import pandas as pd
import openmeteo_requests
import bittensor as bt
from typing import List

def debug_api_differences():
    """
    Sprawdź dlaczego miner i validator dostają różne wyniki z OpenMeteo
    """
    print("🔍 DEBUGGING API DIFFERENCES")
    print("=" * 50)
    
    # Test parameters z twoich logów
    test_cases = [
        {
            "name": "Wind Request (z logów minera)",
            "latitudes": [30.75, 30.75, 30.75, 31.0, 31.0, 31.0, 31.25, 31.25, 31.25],
            "longitudes": [-113.0, -112.75, -112.5, -113.0, -112.75, -112.5, -113.0, -112.75, -112.5],
            "hourly": ["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m"],
            "start_hour": "2025-08-06T03:00",
            "end_hour": "2025-08-06T16:00",
            "variable": "100m_v_component_of_wind"
        },
        {
            "name": "Precipitation Request (z logów validatora)",  
            "latitudes": [38.5, 38.5, 38.5, 38.5, 38.75, 38.75, 38.75, 38.75, 39.0, 39.0, 39.0, 39.0, 39.25, 39.25, 39.25, 39.25],
            "longitudes": [-132.75, -132.5, -132.25, -132.0, -132.75, -132.5, -132.25, -132.0, -132.75, -132.5, -132.25, -132.0, -132.75, -132.5, -132.25, -132.0],
            "hourly": ["precipitation"],
            "start_hour": "2025-08-06T13:00", 
            "end_hour": "2025-08-07T02:00",
            "variable": "total_precipitation"
        }
    ]
    
    api_client = openmeteo_requests.Client()
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}: {test_case['name']}")
        print("-" * 40)
        
        params = {
            "latitude": test_case["latitudes"],
            "longitude": test_case["longitudes"], 
            "hourly": test_case["hourly"],
            "start_hour": test_case["start_hour"],
            "end_hour": test_case["end_hour"],
        }
        
        print(f"📍 Locations: {len(test_case['latitudes'])} points")
        print(f"⏰ Time: {test_case['start_hour']} -> {test_case['end_hour']}")
        print(f"📊 Variables: {test_case['hourly']}")
        
        # Call API multiple times to check consistency
        results = []
        
        for attempt in range(3):
            print(f"\n  🔄 Attempt {attempt + 1}/3...")
            
            try:
                start_time = time.time()
                responses = api_client.weather_api(
                    "https://api.open-meteo.com/v1/forecast", 
                    params=params
                )
                duration = time.time() - start_time
                
                print(f"    ✅ API responded in {duration:.2f}s")
                print(f"    📦 Received {len(responses)} location responses")
                
                # Process responses like miner does
                all_data = []
                for r in responses:
                    location_data = []
                    for var_idx in range(r.Hourly().VariablesLength()):
                        variable_data = r.Hourly().Variables(var_idx).ValuesAsNumpy()
                        location_data.append(variable_data)
                    all_data.append(np.stack(location_data, axis=-1))
                
                raw_data = np.stack(all_data, axis=1)  # (time, locations, variables)
                
                print(f"    📏 Raw data shape: {raw_data.shape}")
                
                # Podstawowe statystyki
                stats = {
                    "mean": raw_data.mean(),
                    "std": raw_data.std(),
                    "min": raw_data.min(),
                    "max": raw_data.max(),
                    "nan_count": np.isnan(raw_data).sum(),
                    "inf_count": np.isinf(raw_data).sum()
                }
                
                print(f"    📊 Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                print(f"    📏 Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"    🔍 NaN: {stats['nan_count']}, Inf: {stats['inf_count']}")
                
                # Pokaż sample values
                if raw_data.size > 0:
                    sample_vals = raw_data[0, :min(3, raw_data.shape[1]), :].flatten()[:5]
                    print(f"    🎯 Sample values: {sample_vals.tolist()}")
                
                results.append({
                    "attempt": attempt + 1,
                    "duration": duration,
                    "shape": raw_data.shape,
                    "stats": stats,
                    "raw_data": raw_data.copy()
                })
                
                # Krótka przerwa między requestami
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ API call failed: {e}")
                results.append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        # Porównaj wyniki między attempts
        print(f"\n  📈 Consistency Analysis:")
        
        successful_results = [r for r in results if "error" not in r]
        
        if len(successful_results) >= 2:
            # Porównaj pierwsze 2 successful results
            data1 = successful_results[0]["raw_data"]
            data2 = successful_results[1]["raw_data"]
            
            if data1.shape == data2.shape:
                rmse = np.sqrt(np.mean((data1 - data2)**2))
                max_diff = np.abs(data1 - data2).max()
                
                print(f"    🔄 RMSE between attempts: {rmse:.6f}")
                print(f"    📊 Max difference: {max_diff:.6f}")
                
                if rmse > 0.001:
                    print(f"    ⚠️  API results are NOT consistent!")
                    print(f"       This could explain different RMSE values")
                    
                    # Find where differences occur
                    diff_mask = np.abs(data1 - data2) > 0.001
                    if diff_mask.any():
                        diff_locations = np.where(diff_mask)
                        print(f"    🎯 Differences at positions: {list(zip(diff_locations[0][:5], diff_locations[1][:5]))}")
                else:
                    print(f"    ✅ API results are consistent")
            else:
                print(f"    ❌ Different shapes: {data1.shape} vs {data2.shape}")
        else:
            print(f"    ❌ Not enough successful results to compare")
        
        print(f"\n  🕐 Time Analysis:")
        for r in results:
            if "duration" in r:
                print(f"    Attempt {r['attempt']}: {r['duration']:.2f}s")
    
    print(f"\n🎯 POTENTIAL CAUSES OF RMSE DIFFERENCES:")
    print("1. 🕐 Timing differences - API data updates between miner/validator calls")
    print("2. 🔄 API inconsistency - same request returns different results")  
    print("3. ⚙️  Processing differences - different conversion/reshaping logic")
    print("4. 🌐 Network/caching - different data served to different clients")
    print("5. 📐 Floating point precision - small rounding differences")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Run this script multiple times to check API consistency")
    print("2. Compare exact same request from miner vs validator at same time")
    print("3. Check if validator/miner use different time zones or formats")
    print("4. Verify both use identical conversion logic (om_to_era5)")

def test_wind_conversion():
    """Test wind conversion specifically since that's where RMSE diff occurs"""
    print(f"\n🌪️ TESTING WIND CONVERSION")
    print("=" * 40)
    
    # Mock wind data like from API (4 variables: speed_80m, dir_80m, speed_120m, dir_120m)
    mock_wind_data = torch.tensor([
        [10.0, 45.0, 12.0, 50.0],  # Location 1: speeds in km/h, directions in degrees
        [8.0, 30.0, 10.0, 35.0],   # Location 2
        [15.0, 60.0, 18.0, 65.0],  # Location 3
    ]).unsqueeze(0)  # Add time dimension: [1, 3, 4]
    
    print(f"🔢 Mock wind data shape: {mock_wind_data.shape}")
    print(f"📊 Sample data: {mock_wind_data[0, 0].tolist()}")
    
    # Test conversion like in miner
    try:
        from zeus.data.converter import get_converter
        
        converter = get_converter("100m_v_component_of_wind")
        print(f"🔧 Using converter: {converter.__class__.__name__}")
        print(f"📝 Input variables: {converter.om_name}")
        
        # Convert to ERA5 format
        converted = converter.om_to_era5(mock_wind_data)
        print(f"✅ Conversion successful")
        print(f"📏 Output shape: {converted.shape}")
        print(f"📊 Sample output: {converted[0, :3].tolist()}")
        
        # Check if conversion is deterministic
        converted2 = converter.om_to_era5(mock_wind_data)
        conversion_diff = torch.abs(converted - converted2).max()
        print(f"🔄 Conversion consistency: max diff = {conversion_diff:.8f}")
        
    except Exception as e:
        print(f"❌ Wind conversion test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_api_differences()
    test_wind_conversion()
    
    print(f"\n🏁 SUMMARY")
    print("=" * 50)
    print("This script helps identify why miner and validator get different RMSE")
    print("when they should be using the same OpenMeteo API.")
    print("")
    print("🔍 Check the consistency analysis above:")
    print("- If RMSE between attempts > 0.001 → API is inconsistent")
    print("- If RMSE = 0.000000 → API is consistent, problem is elsewhere")
    print("")
    print("💡 If API is consistent, the difference might be in:")
    print("- Request timing (validator calls API at different time than miner)")
    print("- Data processing pipeline differences")  
    print("- Coordinate ordering or reshaping logic")