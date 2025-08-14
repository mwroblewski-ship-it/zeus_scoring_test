#!/usr/bin/env python3
"""
Test script do debugowania pobierania ground truth z ERA5
"""

import os
import sys
import torch
import pandas as pd
import bittensor as bt

# Dodaj ścieżki
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from zeus.data.sample import Era5Sample
from zeus.utils.coordinates import get_grid
from zeus.validator.enhanced_validator import EnhancedERA5Validator

def test_ground_truth_fetch():
    """
    Test pobierania ground truth dla prostego przypadku
    """
    print("🧪 TESTING ENHANCED GROUND TRUTH FETCHING")
    print("=" * 60)
    
    # Stwórz prosty sample z danymi historycznymi (żeby na pewno były dostępne)
    # Użyj dat z przeszłości
    start_time = pd.Timestamp('2024-08-01 12:00:00')
    end_time = pd.Timestamp('2024-08-01 15:00:00')  # 4 godziny
    
    # Prosty grid 2x2
    lat_start, lat_end = 50.0, 50.25
    lon_start, lon_end = 10.0, 10.25
    
    sample = Era5Sample(
        start_timestamp=start_time.timestamp(),
        end_timestamp=end_time.timestamp(),
        lat_start=lat_start,
        lat_end=lat_end,
        lon_start=lon_start,
        lon_end=lon_end,
        variable="2m_temperature",
        predict_hours=4
    )
    
    print(f"📋 Test Sample Info:")
    print(f"   Variable: {sample.variable}")
    print(f"   Time: {start_time} -> {end_time}")
    print(f"   Location: lat[{lat_start}, {lat_end}], lon[{lon_start}, {lon_end}]")
    print(f"   Grid shape: {list(sample.x_grid.shape)}")
    print(f"   Predict hours: {sample.predict_hours}")
    
    # Test enhanced validator
    try:
        print(f"\n🚀 Testing Enhanced Validator...")
        validator = EnhancedERA5Validator()
        
        ground_truth = validator.get_era5_ground_truth(sample)
        
        if ground_truth is not None:
            print(f"✅ SUCCESS! Ground truth loaded")
            print(f"   Shape: {list(ground_truth.shape)}")
            print(f"   Mean: {ground_truth.mean().item():.4f}")
            print(f"   Std: {ground_truth.std().item():.4f}")
            print(f"   Range: [{ground_truth.min().item():.4f}, {ground_truth.max().item():.4f}]")
            
            # Pokaż kilka przykładowych wartości
            print(f"   Sample values (hour 0, all points): {ground_truth[0].flatten().tolist()}")
            
            # Test czy możemy używać tego jako ground truth
            print(f"\n✅ Enhanced validation będzie działać!")
            return True
        else:
            print(f"❌ FAILED to get ground truth")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_simple_validation():
    """
    Test całego procesu enhanced validation
    """
    print(f"\n🧪 TESTING COMPLETE ENHANCED VALIDATION")
    print("=" * 60)
    
    # Stwórz sample
    start_time = pd.Timestamp('2024-07-15 06:00:00')  # Starsze dane
    end_time = pd.Timestamp('2024-07-15 08:00:00')    # 3 godziny
    
    sample = Era5Sample(
        start_timestamp=start_time.timestamp(),
        end_timestamp=end_time.timestamp(),
        lat_start=45.0,
        lat_end=45.5,
        lon_start=5.0,
        lon_end=5.5,
        variable="2m_temperature",
        predict_hours=3
    )
    
    # Stwórz fake miner data
    from zeus.validator.miner_data import MinerData
    
    # Fake prediction (podobny do prawdziwego)
    fake_prediction = torch.randn(3, 3, 3) * 5 + 288  # Temperatura około 288K
    
    miners_data = [
        MinerData(uid=117, hotkey="fake_hotkey", prediction=fake_prediction)
    ]
    
    # Fake baseline
    baseline_data = torch.randn(3, 3, 3) * 3 + 290
    
    print(f"📋 Test setup:")
    print(f"   Sample: {sample.variable}, {start_time} -> {end_time}")
    print(f"   Fake miner prediction shape: {list(fake_prediction.shape)}")
    print(f"   Baseline shape: {list(baseline_data.shape)}")
    
    try:
        validator = EnhancedERA5Validator()
        
        validation_results = validator.enhanced_validation(
            sample, miners_data, baseline_data
        )
        
        print(f"\n📊 Validation Results:")
        for key, value in validation_results.items():
            print(f"   {key}: {value}")
        
        if validation_results["ground_truth_available"]:
            print(f"✅ Enhanced validation SUCCESSFUL!")
            
            # Test zapisywania raportu
            report_path = validator.save_validation_report(
                sample, validation_results, miners_data, baseline_data
            )
            print(f"📄 Report saved to: {report_path}")
            
            return True
        else:
            print(f"❌ Enhanced validation failed to get ground truth")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in validation: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔧 ENHANCED VALIDATOR DEBUG TESTS")
    print("=" * 80)
    
    # Test 1: Podstawowe pobieranie ground truth
    test1_success = test_ground_truth_fetch()
    
    # Test 2: Pełna enhanced validation
    test2_success = test_simple_validation()
    
    print(f"\n🏁 SUMMARY:")
    print(f"   Test 1 (Ground Truth Fetch): {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Test 2 (Enhanced Validation): {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 Enhanced validator is ready to use!")
        print(f"   Your validators should now get real ERA5 ground truth data")
        print(f"   for more accurate miner comparisons.")
    else:
        print(f"\n🔧 Some issues found - check the error messages above")
        print(f"   Make sure you have:")
        print(f"   1. Valid CDS_API_KEY in validator.env")
        print(f"   2. Internet connection")
        print(f"   3. All required packages installed")