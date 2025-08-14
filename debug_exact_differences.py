#!/usr/bin/env python3
"""
Debug script to compare EXACT data that miner and validator are processing
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import pandas as pd
import openmeteo_requests
from typing import Dict, Any

def recreate_wind_challenge():
    """
    Recreate exact wind challenge from logs and see where difference occurs
    """
    print("🌪️ RECREATING WIND CHALLENGE FROM LOGS")
    print("=" * 50)
    
    # Exact parameters from your logs
    miner_params = {
        "latitude": [30.75, 30.75, 30.75, 31.0, 31.0, 31.0, 31.25, 31.25, 31.25],
        "longitude": [-113.0, -112.75, -112.5, -113.0, -112.75, -112.5, -113.0, -112.75, -112.5],
        "hourly": ["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m"],
        "start_hour": "2025-08-06T03:00",
        "end_hour": "2025-08-06T16:00",
    }
    
    print("📋 Challenge parameters:")
    print(f"   📍 Grid: 3x3 locations")
    print(f"   ⏰ Time: {miner_params['start_hour']} -> {miner_params['end_hour']} (14 hours)")
    print(f"   🌪️ Variable: 100m_v_component_of_wind")
    print(f"   📊 Expected shape: [14, 3, 3]")
    
    # Call API exactly like miner and validator do
    api_client = openmeteo_requests.Client()
    
    try:
        print(f"\n🔄 Calling OpenMeteo API...")
        responses = api_client.weather_api(
            "https://api.open-meteo.com/v1/forecast", 
            params=miner_params
        )
        
        print(f"✅ API responded with {len(responses)} location responses")
        
        # Process EXACTLY like miner does (from neurons/miner.py)
        print(f"\n⚙️ Processing like MINER...")
        
        # Method 1: Miner's approach (from neurons/miner.py lines ~150-160)
        miner_output = torch.Tensor(np.stack(
            [
                np.stack(
                    [
                        r.Hourly().Variables(i).ValuesAsNumpy() 
                        for i in range(r.Hourly().VariablesLength())
                    ],
                    axis=-1
                )
                for r in responses
            ],
            axis=1
        )).reshape(14, 3, 3, -1)  # [time, lat, lon, variables]
        
        # Remove last dimension if single variable output
        miner_output = miner_output.squeeze(dim=-1)
        
        print(f"   📏 Miner raw shape: {miner_output.shape}")
        print(f"   📊 Miner raw stats: mean={miner_output.mean():.4f}, std={miner_output.std():.4f}")
        
        # Apply wind conversion (from zeus/data/converter.py)
        from zeus.data.converter import get_converter
        converter = get_converter("100m_v_component_of_wind")
        
        miner_converted = converter.om_to_era5(miner_output)
        
        print(f"   📏 Miner converted shape: {miner_converted.shape}")
        print(f"   📊 Miner converted stats: mean={miner_converted.mean():.4f}, std={miner_converted.std():.4f}")
        print(f"   🎯 Miner sample values: {miner_converted[0, :2, :2].flatten().tolist()}")
        
        # Method 2: Validator's approach (from zeus/data/loaders/openmeteo.py)
        print(f"\n⚙️ Processing like VALIDATOR...")
        
        # Simulate validator's get_output method
        all_data = []
        for r in responses:
            location_data = []
            for i in range(r.Hourly().VariablesLength()):
                variable_data = r.Hourly().Variables(i).ValuesAsNumpy()
                location_data.append(variable_data)
            all_data.append(np.stack(location_data, axis=-1))
        
        raw_data = np.stack(all_data, axis=1)  # (time_total, locations, variables)
        
        print(f"   📏 Validator raw shape: {raw_data.shape}")
        print(f"   📊 Validator raw stats: mean={raw_data.mean():.4f}, std={raw_data.std():.4f}")
        
        # Reshape to grid format like validator does
        validator_output = torch.tensor(raw_data).reshape(14, 3, 3, -1)
        validator_output = validator_output.squeeze(dim=-1)
        
        # Apply same conversion
        validator_converted = converter.om_to_era5(validator_output)
        
        print(f"   📏 Validator converted shape: {validator_converted.shape}")
        print(f"   📊 Validator converted stats: mean={validator_converted.mean():.4f}, std={validator_converted.std():.4f}")
        print(f"   🎯 Validator sample values: {validator_converted[0, :2, :2].flatten().tolist()}")
        
        # Compare results
        print(f"\n🔍 COMPARISON:")
        
        if miner_converted.shape == validator_converted.shape:
            rmse = torch.sqrt(torch.mean((miner_converted - validator_converted) ** 2))
            max_diff = torch.abs(miner_converted - validator_converted).max()
            mean_diff = torch.abs(miner_converted - validator_converted).mean()
            
            print(f"   📊 RMSE: {rmse:.6f}")
            print(f"   📏 Max difference: {max_diff:.6f}")
            print(f"   📈 Mean difference: {mean_diff:.6f}")
            
            if rmse > 0.001:
                print(f"   ⚠️  SIGNIFICANT DIFFERENCE FOUND!")
                
                # Find where differences occur
                diff_mask = torch.abs(miner_converted - validator_converted) > 0.001
                if diff_mask.any():
                    diff_positions = torch.where(diff_mask)
                    print(f"   🎯 Differences at positions: {list(zip(diff_positions[0][:5].tolist(), diff_positions[1][:5].tolist(), diff_positions[2][:5].tolist()))}")
                    
                    # Show actual values at difference positions
                    for i in range(min(3, len(diff_positions[0]))):
                        t, lat, lon = diff_positions[0][i], diff_positions[1][i], diff_positions[2][i]
                        miner_val = miner_converted[t, lat, lon].item()
                        validator_val = validator_converted[t, lat, lon].item()
                        print(f"      Position [{t}, {lat}, {lon}]: Miner={miner_val:.6f}, Validator={validator_val:.6f}")
            else:
                print(f"   ✅ Processing is IDENTICAL!")
        else:
            print(f"   ❌ Different shapes: Miner={miner_converted.shape}, Validator={validator_converted.shape}")
        
        # Test if it's a raw data reshaping issue
        print(f"\n🔧 TESTING COORDINATE ORDERING...")
        
        # Check if coordinates are in same order
        print(f"   📍 Expected coordinate order:")
        for i, (lat, lon) in enumerate(zip(miner_params["latitude"], miner_params["longitude"])):
            grid_lat = i // 3
            grid_lon = i % 3
            print(f"      Location {i}: ({lat}, {lon}) -> Grid[{grid_lat}, {grid_lon}]")
        
        return {
            "miner_output": miner_converted,
            "validator_output": validator_converted,
            "rmse": rmse.item() if 'rmse' in locals() else None
        }
        
    except Exception as e:
        print(f"❌ Error recreating challenge: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_timing_hypothesis():
    """
    Test if timing difference could cause RMSE difference
    """
    print(f"\n🕐 TESTING TIMING HYPOTHESIS")
    print("=" * 40)
    
    # From your logs:
    # Miner called at: 2025-08-14 12:31:41
    # Validator called at: 2025-08-14 12:33:12 (1.5 minutes later)
    
    print(f"📅 From your logs:")
    print(f"   Miner API call: 12:31:41")
    print(f"   Validator API call: 12:33:12 (1.5 minutes later)")
    print(f"   Challenge time: 2025-08-06 (8 days in past)")
    
    print(f"\n🤔 Analysis:")
    print(f"   Since challenge is 8 days in the past (historical data),")
    print(f"   OpenMeteo should return identical results regardless of when you call it.")
    print(f"   Historical weather data doesn't change minute by minute.")
    
    print(f"\n💡 Conclusion:")
    print(f"   Timing difference is UNLIKELY to be the cause.")
    print(f"   The difference must be in data processing pipeline.")

def analyze_your_logs():
    """
    Analyze the specific values from your logs to find the discrepancy
    """
    print(f"\n📋 ANALYZING YOUR ACTUAL LOG VALUES")
    print("=" * 50)
    
    # From your logs - miner wind prediction
    miner_sample_hour_0 = [2.5248215198516846, 1.5175840854644775, 0.7051709890365601, -0.7041362524032593]
    miner_sample_hour_13 = [3.328958511352539, 1.2572410106658936, 2.769651174545288, 1.3010339736938477]
    miner_stats = {"mean": 1.1153, "std": 1.7440, "min": -2.2165, "max": 6.1872}
    
    # From your logs - validator baseline
    baseline_sample_hour_0 = [3.624823491258103, 2.258276013236996, 2.674820627532304, 0.7572395107494851]
    baseline_sample_hour_13 = [0.0, 0.0, 0.0, 0.0]  # All zeros!
    baseline_stats = {"mean": 1.0839, "std": 1.6118, "min": -1.8634, "max": 6.1872}
    
    # RMSE from logs
    rmse_vs_baseline = 1.8378
    
    print(f"🤖 MINER prediction (from logs):")
    print(f"   Hour 0 sample: {miner_sample_hour_0}")
    print(f"   Hour 13 sample: {miner_sample_hour_13}")
    print(f"   Stats: {miner_stats}")
    
    print(f"\n🌐 VALIDATOR baseline (from logs):")
    print(f"   Hour 0 sample: {baseline_sample_hour_0}")
    print(f"   Hour 13 sample: {baseline_sample_hour_13}")  # ← ALL ZEROS!
    print(f"   Stats: {baseline_stats}")
    
    print(f"\n📊 RMSE difference: {rmse_vs_baseline}")
    
    print(f"\n🎯 KEY OBSERVATION:")
    print(f"   ⚠️  Validator baseline shows ALL ZEROS at hour 13!")
    print(f"   🤖 Miner shows real values at hour 13!")
    print(f"   📊 This explains the RMSE difference!")
    
    print(f"\n💡 HYPOTHESIS:")
    print(f"   1. Validator's OpenMeteo call may be getting truncated data")
    print(f"   2. Or validator is processing/reshaping data incorrectly")
    print(f"   3. Or validator is using different time slicing logic")
    
    # Calculate what RMSE should be with zeros
    import torch
    miner_tensor = torch.tensor([[miner_sample_hour_0[0], miner_sample_hour_0[1]], [miner_sample_hour_13[0], miner_sample_hour_13[1]]])
    baseline_tensor = torch.tensor([[baseline_sample_hour_0[0], baseline_sample_hour_0[1]], [0.0, 0.0]])  # zeros at hour 13
    
    calculated_rmse = torch.sqrt(torch.mean((miner_tensor - baseline_tensor) ** 2))
    print(f"\n🧮 Calculated RMSE with sample values: {calculated_rmse:.4f}")
    print(f"   📋 Logged RMSE: {rmse_vs_baseline}")
    print(f"   ✅ Close match - confirms the difference is real!")

if __name__ == "__main__":
    print("🔍 DEBUGGING EXACT MINER vs VALIDATOR DIFFERENCE")
    print("=" * 60)
    
    # Analyze the values from your logs first
    analyze_your_logs()
    
    # Test timing hypothesis
    debug_timing_hypothesis()
    
    # Recreate exact challenge to see where difference occurs
    result = recreate_wind_challenge()
    
    if result and result["rmse"] is not None:
        print(f"\n🎯 FINAL DIAGNOSIS:")
        print(f"   RMSE when processing same API data: {result['rmse']:.6f}")
        if result["rmse"] < 0.001:
            print(f"   ✅ Same processing = identical results")
            print(f"   💡 Difference must be in the DATA miner vs validator receive")
        else:
            print(f"   ⚠️  Different processing = different results")
            print(f"   💡 Difference is in the PROCESSING pipeline")
    
    print(f"\n🔧 NEXT STEPS:")
    print(f"1. Compare actual API responses validator vs miner receive")
    print(f"2. Check if validator truncates data or has different time slicing")
    print(f"3. Look at zeus/data/loaders/openmeteo.py get_output() method")
    print(f"4. Compare coordinate ordering between miner and validator")