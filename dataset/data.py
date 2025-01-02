import os
from dataset.dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderLow


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)

def get_unpair_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderLow(rgb_dir, img_options)
