import pandas as pd
import numpy as np
from app import calculate_pcr

def test_pcr_calculation_handles_nan():
    """
    Tests that the PCR calculation correctly handles NaN values
    that arise from 0/0 division by replacing them with 0.
    """
    # 1. Arrange: Create a sample DataFrame that will produce NaN
    data = {
        "Strike Price": [100, 200, 300],
        "CE_OI": [1000, 0, 500],
        "PE_OI": [800, 0, 600],
        "CE_Chng_OI": [200, 0, -100],
        "PE_Chng_OI": [150, 0, 50]
    }
    df = pd.DataFrame(data)

    # 2. Act: Call the function from app.py
    df = calculate_pcr(df)

    # 3. Assert: Check that NaN has been replaced by 0
    assert df.loc[1, "PCR_OI"] == 0, "PCR_OI should be 0, not NaN"
    assert df.loc[1, "PCR_Chng_OI"] == 0, "PCR_Chng_OI should be 0, not NaN"

def test_pcr_calculation_handles_inf():
    """
    Tests that the PCR calculation correctly handles infinite values
    that arise from division by zero, replacing them with 0.
    """
    # 1. Arrange: Create a sample DataFrame that will produce inf
    data = {
        "Strike Price": [100],
        "CE_OI": [0],
        "PE_OI": [800],
        "CE_Chng_OI": [0],
        "PE_Chng_OI": [150]
    }
    df = pd.DataFrame(data)

    # 2. Act: Call the function from app.py
    df = calculate_pcr(df)

    # 3. Assert: Check that inf has been replaced by 0
    assert df.loc[0, "PCR_OI"] == 0, "PCR_OI should be 0, not inf"
    assert df.loc[0, "PCR_Chng_OI"] == 0, "PCR_Chng_OI should be 0, not inf"
