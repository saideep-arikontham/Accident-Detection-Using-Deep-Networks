# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from collections import defaultdict

def count_images_per_class(base_path, sets, classes):
    data_summary = defaultdict(dict)
    for ds in sets:
        for cls in classes:
            folder = os.path.join(base_path, ds, cls)
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            data_summary[ds][cls] = len(files)
    return pd.DataFrame(data_summary).T


def display_image_counts(df_counts):
    df_counts.columns.name = "Class"
    df_counts.index.name = "Dataset"
    print("Image Count per Class per Dataset:\n")
    print(df_counts, "\n")


def show_sample_images(base_path, sets, classes, n_samples=5):
    for ds in sets:
        for cls in classes:
            folder = os.path.join(base_path, ds, cls)
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue
            sample_files = random.sample(files, min(n_samples, len(files)))

            plt.figure(figsize=(15, 3))
            for i, file in enumerate(sample_files):
                img_path = os.path.join(folder, file)
                img = mpimg.imread(img_path)
                plt.subplot(1, n_samples, i + 1)
                plt.imshow(img)
                plt.title(cls)
                plt.axis('off')
            plt.suptitle(f"{ds.upper()} - {cls} Samples", fontsize=14)
            plt.tight_layout()
            plt.show()


def main():
    # Set dataset path
    base_path = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data"
    
    # Dataset splits and classes
    sets = ['train', 'val', 'test']
    classes = ['Accident', 'Non Accident']

    # Count and display image distribution
    df_counts = count_images_per_class(base_path, sets, classes)
    display_image_counts(df_counts)

    # Show random image samples
    show_sample_images(base_path, sets, classes)


if __name__ == "__main__":
    main()
