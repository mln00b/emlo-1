import os
import csv

def test_data_not_uploaded():
    assert not os.path.isdir("data"), "Data exists. This should not be uploaded"

def test_model_not_uploaded():
    assert not os.path.isfile("model.pth"), "Model exists. This should not be uploaded"

def test_accuracy():
    if os.path.isfile("Metrics.csv"):
        with open("Metrics.csv") as f:
            cr = csv.reader(f)
            last = list(cr)[-1]
            val_acc = float(last[3])
            assert val_acc > 0.7, f"Val accuracy < 70%: {val_acc}"

def test_per_class_accuracy():
    if os.path.isfile("Metrics.csv"):
        with open("Metrics.csv") as f:
            cr = csv.reader(f)
            last = list(cr)[-1]
            class_1 = float(last[-2])
            class_2 = float(last[-1])
            assert class_1 > 0.7, f"Class 1 accuracy < 70%: {class_1}"
            assert class_2 > 0.7, f"Class 1 accuracy < 70%: {class_2}"