# tests/test_api.py
import pytest
import sys
from pathlib import Path

# Añadir el directorio src al path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

def test_api_import():
    """Test that the API module can be imported without errors"""
    try:
        from src.api import app, PassengerRequest, PredictionResponse
        assert app is not None
        assert PassengerRequest is not None
        assert PredictionResponse is not None
        print("✅ API modules imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import API modules: {e}")

def test_pydantic_models():
    """Test that Pydantic models work correctly"""
    from src.api import PassengerRequest, PredictionResponse
    
    # Test PassengerRequest
    passenger_data = {
        "Pclass": 1,
        "Name": "John Doe",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST",
        "Fare": 50.0
    }
    
    passenger = PassengerRequest(**passenger_data)
    assert passenger.Pclass == 1
    assert passenger.Sex == "male"
    assert passenger.Age == 30.0
    
    # Test PredictionResponse - CORREGIDO: USAR MAYÚSCULAS
    prediction_data = {
        "passenger_id": "test",
        "prediction": 1,          # ← P MAYÚSCULA (como en API)
        "probability": 0.85,      # ← P MAYÚSCULA (como en API)
        "survival_status": "Survived",
        "message": "Test message"
    }
    
    prediction = PredictionResponse(**prediction_data)
    assert prediction.Prediction == 1        # ← Acceder con mayúscula
    assert prediction.Probability == 0.85    # ← Acceder con mayúscula

def test_data_preprocessing_functions():
    """Test that preprocessing functions can be imported and used"""
    try:
        from src.data_preprocessing import clean_data, encode_data
        import pandas as pd
        
        # Create test data
        test_df = pd.DataFrame({
            'Pclass': [1, 3],
            'Name': ['Test1', 'Test2'],
            'Sex': ['male', 'female'],
            'Age': [30, 25],
            'SibSp': [0, 1],
            'Parch': [0, 0],
            'Ticket': ['T1', 'T2'],
            'Fare': [50.0, 10.0],
            'Cabin': [None, 'C123'],
            'Embarked': ['S', 'C']
        })
        
        # Test functions
        cleaned = clean_data(test_df)
        assert cleaned is not None
        assert 'Has_Cabin' in cleaned.columns
        
        encoded = encode_data(cleaned)
        assert encoded is not None
        
        print("✅ Data preprocessing functions work correctly")
        
    except ImportError as e:
        pytest.fail(f"Data preprocessing import failed: {e}")

if __name__ == "__main__":
    test_api_import()
    test_pydantic_models()
    test_data_preprocessing_functions()
    print("✅ All unit tests passed!")# tests/test_api.py
import pytest
import sys
from pathlib import Path

# Añadir el directorio src al path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

def test_api_import():
    """Test that the API module can be imported without errors"""
    try:
        from src.api import app, PassengerRequest, PredictionResponse
        assert app is not None
        assert PassengerRequest is not None
        assert PredictionResponse is not None
        print("API modules imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import API modules: {e}")

def test_pydantic_models():
    """Test that Pydantic models work correctly"""
    from src.api import PassengerRequest, PredictionResponse
    
    # Test PassengerRequest
    passenger_data = {
        "Pclass": 1,
        "Name": "John Doe",
        "Sex": "male",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST",
        "Fare": 50.0
    }
    
    passenger = PassengerRequest(**passenger_data)
    assert passenger.Pclass == 1
    assert passenger.Sex == "male"
    assert passenger.Age == 30.0
    
    # Test PredictionResponse
    prediction_data = {
        "passenger_id": "test",
        "prediction": 1,
        "probability": 0.85,
        "survival_status": "Survived",
        "message": "Test message"
    }
    
    prediction = PredictionResponse(**prediction_data)
    assert prediction.prediction == 1
    assert prediction.probability == 0.85

def test_data_preprocessing_functions():
    """Test that preprocessing functions can be imported and used"""
    try:
        from src.data_preprocessing import clean_data, encode_data
        import pandas as pd
        
        # Create test data
        test_df = pd.DataFrame({
            'Pclass': [1, 3],
            'Name': ['Test1', 'Test2'],
            'Sex': ['male', 'female'],
            'Age': [30, 25],
            'SibSp': [0, 1],
            'Parch': [0, 0],
            'Ticket': ['T1', 'T2'],
            'Fare': [50.0, 10.0],
            'Cabin': [None, 'C123'],
            'Embarked': ['S', 'C']
        })
        
        # Test functions
        cleaned = clean_data(test_df)
        assert cleaned is not None
        assert 'Has_Cabin' in cleaned.columns
        
        encoded = encode_data(cleaned)
        assert encoded is not None
        
        print("Data preprocessing functions work correctly")
        
    except ImportError as e:
        pytest.fail(f"Data preprocessing import failed: {e}")

if __name__ == "__main__":
    test_api_import()
    test_pydantic_models()
    test_data_preprocessing_functions()
    print("All unit tests passed!")