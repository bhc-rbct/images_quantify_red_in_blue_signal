import os
import csv
import copy
import random
import argparse
import numpy as np
import skimage
from pathlib import Path
from skimage.draw import polygon
from skimage import measure, morphology, exposure
from skimage.filters import threshold_otsu, threshold_yen, gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from collections import defaultdict
from statistics import mean
import warnings

# Suppress specific UserWarnings related to skimage's remove_small_holes function
warnings.filterwarnings(
    "ignore",
    message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?",
)


def get_filtered_cell_props(image, blue_area_threshold):
    """
    Filter and retrieve cell properties based on the blue channel of the image.

    Args:
        image (ndarray): The input image containing cells.
        blue_area_threshold (int): The maximum size of a hole to be filled.

    Returns:
        tuple: A tuple containing:
            - filtered_cell_props (list): A list of filtered cell properties.
            - filtered_cell_labels (ndarray): The labeled image after filtering out border cells.
    """
    blue_channel = image[:, :, 2]
    # Optionally, smooth the image to reduce noise
    blue_channel = gaussian(blue_channel, sigma=2)

    # Thresholding to binarize the image
    threshold = threshold_otsu(blue_channel)
    thresholded_image = (blue_channel > threshold) * 255

    # Remove small holes in the thresholded image
    cells_image_blue = morphology.remove_small_holes(
        thresholded_image, connectivity=1, area_threshold=blue_area_threshold
    )
    cells_image_blue = morphology.remove_small_objects(cells_image_blue)

    # Detect connected components (cells)
    cell_labels = measure.label(cells_image_blue, connectivity=2)

    # Get properties of the labeled regions
    cell_props = measure.regionprops(cell_labels)

    # Initialize a mask to remove edge cells
    mask = np.ones(cell_labels.shape, dtype=bool)
    filtered_cell_props = []

    for prop in cell_props:
        min_row, min_col, max_row, max_col = prop.bbox
        # Filter out cells that touch the image border
        if (
            min_row > 0
            and min_col > 0
            and max_row < cells_image_blue.shape[0]
            and max_col < cells_image_blue.shape[1]
        ):
            filtered_cell_props.append(prop)
        else:
            # Remove edge cells from the mask
            mask[cell_labels == prop.label] = False

    # Apply the mask to remove edge cells from the labeled image
    filtered_cell_labels = cell_labels * mask
    return filtered_cell_props, filtered_cell_labels


def get_props_red(image, min_red_int, sigma):
    """
    Retrieve properties of red dots in the image.

    Args:
        image (ndarray): The input image containing cells.
        min_red_int (int): Minimum red intensity to consider for thresholding.
        sigma (float): Sigma value for Gaussian smoothing to reduce noise.

    Returns:
        tuple: A tuple containing:
            - props_red (list): A list of properties of the red dots.
            - thresholded_image (ndarray): The thresholded binary image of the red dots.
    """
    red_channel = image[:, :, 0]

    # Smooth the image to reduce noise
    smoothed_image = gaussian(red_channel, sigma=sigma)

    # Create a mask where the red channel is the maximum and above a minimum intensity
    red_min_val_mask = image[:, :, 0] > 0
    red_is_max = np.argmax(image, axis=2) == 0
    max_is_red_and_has_min_val = red_min_val_mask & red_is_max

    # Apply the mask to the smoothed red channel
    red_channel_where_max_is_red_and_has_min_val = smoothed_image * max_is_red_and_has_min_val

    # Thresholding (Otsu or Yen method)
    threshold = threshold_otsu(red_channel_where_max_is_red_and_has_min_val)
    threshold = (
        threshold
        if threshold > min_red_int
        else max(threshold_yen(red_channel_where_max_is_red_and_has_min_val), min_red_int)
    )

    thresholded_image = (red_channel_where_max_is_red_and_has_min_val > threshold) * 255

    # Detect connected components (red dots)
    red_labels = measure.label(thresholded_image, connectivity=1)
    props_red = measure.regionprops(red_labels, red_channel)

    return props_red, thresholded_image


def count_plot_reds_per_cells(
    filtered_cell_props, filtered_cell_labels, props_red, original_image, plot_path
):
    """
    Count and plot the number of red dots per cell in the image.

    Args:
        filtered_cell_props (list): Properties of the filtered cells.
        filtered_cell_labels (ndarray): Labeled image of the filtered cells.
        props_red (list): Properties of the red dots.
        original_image (ndarray): The original input image.
        plot_path (str): Path to save the plot with counted red dots.

    Returns:
        list: A list containing the number of red dots per cell.
    """
    red_dots_per_cell = {i: 0 for i in range(1, len(filtered_cell_props) + 1)}

    # Plot the blue channel image with the filtered cells
    plt.figure(figsize=(10, 10))
    filtered_cell_labels_plot = copy.deepcopy(filtered_cell_labels)
    filtered_cell_labels_plot[filtered_cell_labels_plot > 0] = 255
    plt.imshow(filtered_cell_labels_plot, cmap="gray")

    # Count and plot red dots within each filtered cell
    for red_prop in props_red:
        for i, prop in enumerate(filtered_cell_props, start=1):
            cell_mask = filtered_cell_labels == prop.label
            # Check if the red dot is within the cell
            if cell_mask[int(red_prop.centroid[0]), int(red_prop.centroid[1])] and red_prop.num_pixels > 2:
                red_dots_per_cell[i] += 1
                plt.plot(red_prop.centroid[1], red_prop.centroid[0], "ro", markersize=1)

    red_dots = list(red_dots_per_cell.values())

    # Numbering the filtered cells in the plot
    for i, prop in enumerate(filtered_cell_props, start=1):
        y, x = prop.centroid
        plt.text(x, y, str(i), color="blue", fontsize=12, ha="center", va="center")

    plt.title("Used Cells and counted dots")
    plt.axis("off")
    plt.savefig(plot_path)
    return red_dots


def analyze_image(full_file_path, output_dir_path, blue_area_threshold, red_gaussian_sigma):
    """
    Analyze an image to count red dots per cell and save the results.

    Args:
        full_file_path (str): The full path to the input image file.
        output_dir_path (str): The directory path to save output files.

    Returns:
        list: A row containing the filename and the counts of red dots per cell.
    """
    min_red_int = 0.06
    table_row = None
    file = os.path.basename(full_file_path)
    if full_file_path.endswith(".tif") and "_counted.tif" not in full_file_path:
        # Step 1: Load whole image
        image = skimage.io.imread(full_file_path)

        # Step 2: Get cell props and labels of cells which are not border cells
        filtered_cell_props, filtered_cell_labels = get_filtered_cell_props(image, blue_area_threshold)

        # Step 3: Get labels of red dots
        red_props, thresholded_image = get_props_red(image, min_red_int, red_gaussian_sigma)

        # Step 4: Count red dots per cells + add to result table + plot the image
        reds_per_cell_list = count_plot_reds_per_cells(
            filtered_cell_props,
            filtered_cell_labels,
            red_props,
            thresholded_image,
            output_dir_path / file.replace(".tif", "_counted.tif"),
        )

        table_row = [
            file,
            (np.mean(reds_per_cell_list) if reds_per_cell_list else 0),
        ] + reds_per_cell_list

    return table_row


def new_or_update_csv(output_dir_path, all_table_rows):
    """
    Create or update a CSV file with the results of red dot counts.

    Args:
        output_dir_path (str): The directory path where the CSV file will be saved.
        all_table_rows (list): A list of rows containing the results.
    """
    csv_file = os.path.join(output_dir_path, "counts.csv")
    existing_data = {}

    # Read existing CSV file if it exists
    if os.path.exists(csv_file):
        with open(csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                filename = row[0]
                mean_value = float(row[1])
                cell_counts = row[2:]
                existing_data[filename] = [mean_value] + cell_counts

    # Update or add rows based on all_table_rows
    for row in all_table_rows:
        filename = row[0]
        mean_value = row[1]
        cell_counts = row[2:]

        existing_data[filename] = [mean_value] + cell_counts

    # Determine the maximum number of cells in any row
    max_cells = max(len(data) - 1 for data in existing_data.values())

    # Create a new header based on the maximum number of cells
    header = ["filename", "mean"] + list(map(str, range(1, max_cells + 1)))

    # Write the updated data back to the CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for filename, data in existing_data.items():
            padded_data = data + [""] * (max_cells - len(data) + 1)
            writer.writerow([filename] + padded_data)


def main(args):
    """
    Main function to process images and count red dots per cell.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    min_red_int = 0.06
    if not args.output_dir:
        if os.path.isdir(args.input):
            args.output_dir = Path(args.input)
        else:
            args.output_dir = os.path.dirname(args.input)

    output_dir_path = Path(args.output_dir)

    # Create the directory if it does not exist
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    csv_file = os.path.join(output_dir_path, "counts.csv")
    all_table_rows = []
    if os.path.isdir(args.input):
        input_dir_path = Path(args.input)
        for file in os.listdir(input_dir_path):
            print(f"Image {file} is currently being processed...")
            full_file_path = os.path.join(input_dir_path, file)
            table_row = analyze_image(
                full_file_path, output_dir_path, args.blue_area_threshold, args.red_gaussian_sigma
            )
            if table_row:
                all_table_rows.append(table_row)
    else:
        table_row = analyze_image(
            args.input, output_dir_path, args.blue_area_threshold, args.red_gaussian_sigma
        )
        if table_row:
            all_table_rows.append(table_row)

    new_or_update_csv(output_dir_path, all_table_rows)


if __name__ == "__main__":
    # Argument Parsing
    description = "Counts the number of red dots per cell (DAPI/blue signal). Can be called with a directory or individual image file. If the cells contain holes try increasing -blue_area_threshold. If too many/less red dots are identified increase/lower -red_bg_threshold. "
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i",
        dest="input",
        type=str,
        help="Path to the directory which contains all .tif files, which should be analyzed (should not contain any other .tif files) or to the input file",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        help="Path to directory where the count.csv and annotated .tifs (annotated cells + marked red dots, which where counted; default: input_dir)",
        required=False,
    )

    parser.add_argument(
        "-red_bg_threshold",
        dest="red_gaussian_sigma",
        default=0.75,
        type=float,
        help="Adjustment for red background noise. Set between 0 (no adjustment) and approx. 3 (default: 0.75)",
        required=False,
    )
    parser.add_argument(
        "-blue_area_threshold",
        dest="blue_area_threshold",
        default=64,
        type=int,
        help="Maximal size of a hole to be filled (cell area). Minimal value 0 (default: 64)",
        required=False,
    )

    args = parser.parse_args()
    main(args)
