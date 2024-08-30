import os
import csv
import argparse
import numpy as np
import skimage
import skimage.color as color
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu, threshold_yen
import warnings

# Suppress specific UserWarnings related to skimage's remove_small_holes function
warnings.filterwarnings(
    "ignore",
    message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?",
)


def get_back_foreground_pixels(image, plot_path):
    """
    Convert the input image to grayscale, apply Otsu's thresholding to separate
    the foreground from the background, and calculate the number of foreground
    and background pixels. Save the binary image as a plot.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        plot_path (str): The file path to save the plot showing the binary image.

    Returns:
        tuple: A tuple containing the number of foreground pixels, background pixels, and total pixels.
    """
    # Convert to grayscale
    gray_image = color.rgb2gray(image)

    # Apply a threshold to separate background (white) and foreground
    # Using Otsu's method to determine a suitable threshold
    threshold_value = threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value

    # Calculate the number of white (background) and black (foreground) pixels
    background_pixels = np.sum(binary_image)  # True values (white pixels)
    foreground_pixels = binary_image.size - background_pixels  # False values (black pixels)

    total_pixels = binary_image.size

    # Plot only the binary image
    plt.figure(figsize=(5, 5))
    plt.imshow(binary_image, cmap="gray")
    # plt.title("Binary Image (Foreground/Background)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

    return foreground_pixels, background_pixels, total_pixels


def analyze_image(full_file_path, output_dir_path):
    """
    Analyze an image to calculate the percentage of foreground pixels and save the results.

    Args:
        full_file_path (str): The full path to the input image file.
        output_dir_path (str): The directory path to save output files.

    Returns:
        list: A list containing the filename and the calculated foreground percentage.
    """
    table_row = None
    file = os.path.basename(full_file_path)

    if full_file_path.endswith(".jpg") and "_bw.jpg" not in full_file_path:
        # Step 1: Load the entire image
        image = skimage.io.imread(full_file_path)

        # Step 2: Foreground, background, total pixels
        foreground_pixels, background_pixels, total_pixels = get_back_foreground_pixels(
            image, output_dir_path / file.replace(".jpg", "_bw.jpg")
        )

        # Calculate the percentage of foreground pixels
        foreground_percentage = (foreground_pixels / total_pixels) * 100

        table_row = [file, f"{foreground_percentage:.2f}"]

    return table_row


def new_or_update_csv(output_dir_path, all_table_rows):
    """
    Create or update a CSV file with the results of foreground pixel calculations.

    Args:
        output_dir_path (str): The directory path where the CSV file will be saved.
        all_table_rows (list): A list of rows containing the results.
    """
    csv_file = os.path.join(output_dir_path, "percentage_foreground.csv")
    existing_data = {}

    # Read existing CSV file if it exists
    if os.path.exists(csv_file):
        with open(csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                filename = row[0]
                existing_data[filename] = row[1:]

    # Update or add rows based on all_table_rows
    for row in all_table_rows:
        filename = row[0]
        existing_data[filename] = row[1:]

    # Create a new header based on the maximum number of cells
    header = ["filename", "Foreground percentage"]

    # Write the updated data back to the CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for filename, data in existing_data.items():
            writer.writerow([filename] + data)


def main(args):
    """
    Main function to process images and calculate foreground pixel percentages.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Set output directory path
    if not args.output_dir:
        if os.path.isdir(args.input):
            args.output_dir = Path(args.input)
        else:
            args.output_dir = os.path.dirname(args.input)

    output_dir_path = Path(args.output_dir)

    # Create the output directory if it does not exist
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    all_table_rows = []

    # Process all images in the input directory
    if os.path.isdir(args.input):
        input_dir_path = Path(args.input)
        for file in os.listdir(input_dir_path):
            print(f"Image {file} is currently being processed...")
            full_file_path = os.path.join(input_dir_path, file)
            table_row = analyze_image(full_file_path, output_dir_path)
            if table_row:
                all_table_rows.append(table_row)
    else:
        table_row = analyze_image(args.input, output_dir_path)
        if table_row:
            all_table_rows.append(table_row)

    # Update or create the CSV file with the collected data
    new_or_update_csv(output_dir_path, all_table_rows)


if __name__ == "__main__":
    # Argument Parsing
    description = "Calculates percentage of foreground against the white background (counting the white vs none white pixels)"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        help="Path to the directory which contains all .jpg files to be analyzed (should not contain any other .jpg files) or to a single input file",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        help="Path to the directory where the percentage_foreground.csv and black and white .jpgs will be saved (default: input_dir)",
        required=False,
    )

    args = parser.parse_args()
    main(args)
