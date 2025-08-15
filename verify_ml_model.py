#!/usr/bin/env python3
"""
ML Model Deployment Verification Script
This script checks if all required ML model files are accessible in the current environment.
"""

import os
import sys

def verify_ml_model_files():
    """Verify that all required ML model files are accessible"""
    print("🔍 ML Model Deployment Verification")
    print("=" * 50)
    
    # Possible model directories
    possible_dirs = [
        os.path.join(os.path.dirname(__file__), 'ml_model'),  # Current directory
        '/opt/render/project/src/ml_model',  # Render production
        '/app/ml_model',  # Alternative production path
        os.path.join(os.getcwd(), 'ml_model'),  # Current working directory
    ]
    
    # Required files
    required_files = [
        'model.pkl',
        'encoders.pkl', 
        'categorical_cols.pkl',
        'feature_columns.pkl',
        'numeric_cols.pkl'
    ]
    
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Script location: {os.path.dirname(__file__)}")
    print()
    
    # Check each possible directory
    for i, model_dir in enumerate(possible_dirs):
        print(f"🔍 Checking directory {i+1}: {model_dir}")
        
        if os.path.exists(model_dir):
            print(f"   ✅ Directory exists")
            
            # Check each required file
            all_files_present = True
            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ {file} - {file_size:,} bytes")
                else:
                    print(f"   ❌ {file} - NOT FOUND")
                    all_files_present = False
            
            if all_files_present:
                print(f"   🎉 All files present in {model_dir}")
                print(f"   📊 Total directory size: {sum(os.path.getsize(os.path.join(model_dir, f)) for f in required_files):,} bytes")
                return True
            else:
                print(f"   ⚠️ Some files missing in {model_dir}")
        else:
            print(f"   ❌ Directory does not exist")
        
        print()
    
    print("❌ No valid ML model directory found with all required files")
    return False

def test_model_loading():
    """Test if the model can actually be loaded"""
    print("🧪 Testing Model Loading")
    print("=" * 30)
    
    try:
        # Try to import and load the model
        sys.path.append(os.path.dirname(__file__))
        from landing.views import load_ml_model
        
        print("✅ Successfully imported landing.views")
        
        if load_ml_model():
            print("✅ ML model loaded successfully")
            return True
        else:
            print("❌ ML model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 ML Model Deployment Verification")
    print("=" * 50)
    
    # Check file accessibility
    files_ok = verify_ml_model_files()
    print()
    
    if files_ok:
        # Test model loading
        model_ok = test_model_loading()
        
        if model_ok:
            print("\n🎉 SUCCESS: ML model is fully accessible and functional!")
            sys.exit(0)
        else:
            print("\n⚠️ WARNING: Files exist but model loading failed")
            sys.exit(1)
    else:
        print("\n❌ ERROR: Required ML model files are not accessible")
        sys.exit(1)
