import pytest
from unittest.mock import patch, mock_open, MagicMock
from data.datasets.printed.generate import GenerateSyntheticPrintedDataset


@pytest.fixture
def dataset_generator():
    return GenerateSyntheticPrintedDataset()


@patch('os.path.exists')
@patch('os.listdir')
@patch('builtins.open', new_callable=mock_open, read_data='font1,handwritten\nfont2,serif\nfont3,sans-serif\n')
def test_get_google_fonts(mock_open, mock_listdir, mock_exists, dataset_generator):
    # Mock the responses
    mock_exists.side_effect = lambda path: path in ["datasets/fonts", "datasets/fonts/font-info.csv",
                                                    "datasets/fonts/font3"]
    mock_listdir.return_value = ["font3_Regular.ttf"]

    # Call the method
    fonts = dataset_generator.get_google_fonts()

    # Assertions
    assert len(fonts) == 1
    assert fonts[0] == ("font3", "datasets/fonts/font3/font3_Regular.ttf")
