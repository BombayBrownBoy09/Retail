import pytest
from models.retail_model import RetailModel

def test_retail_model_init():
    params = {"example_param": 42}
    model = RetailModel(params)
    assert model.params == params
