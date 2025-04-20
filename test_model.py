import numpy as np
from app import train_and_predict, get_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada 
    przewidywanej liczbie próbek testowych.
    """
    preds, _ = train_and_predict()
    assert len(preds) > 0, "Predictions length should be greater than 0."
    # Assuming the model predicts on the entire dataset in train_and_predict
    # You might need to adjust this based on your actual implementation
    assert len(preds) == 114, "Predictions length should match the number of test samples."

def test_predictions_value_range():
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie:
    Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    assert all(0 <= pred <= 1 for pred in preds), "Predictions should be within the range [0, 1]."

def test_model_accuracy():
    """
    Test 4 (na maksymalną ocenę 5): Sprawdza, czy model osiąga co najmniej 70% dokładności (przykładowy 
    warunek, można dostosować do potrzeb).
    """
    _, accuracy = train_and_predict()
    assert accuracy >= 0.7, "Model accuracy should be at least 70%."
