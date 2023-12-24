from sys import argv

from utils.predict import get_predictions
from utils.save_csv import save_to_csv

if __name__ == "__main__":
    # _, dir_path, save_path = argv

    res = get_predictions(directory_path='orig_imgs', save_path="save_imgs",
                          device="cuda:0")
    save_to_csv(res, save_path="save_imgs")
