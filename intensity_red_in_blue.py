import os
import csv
import copy
import argparse
import numpy as np
import skimage
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gaussian
import warnings

# Suppress specific UserWarnings related to skimage's remove_small_holes function
warnings.filterwarnings(
    "ignore",
    message="Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?",
)


def get_filtered_cell_props(
    image: ndarray, blue_hole_threshold: int, gaussian_sigma: float, min_size: int
) -> tuple:
    """
    Filter and retrieve cell properties based on the blue channel of the image.

    Args:
        image (ndarray): The input image containing cells.
        blue_hole_threshold (int): The maximum size of a hole to be filled.
        gaussian_sigma (float): Sigma value for Gaussian smoothing to reduce noise.
        min_size (int): Minimal size for a cell to be used.

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
            and prop.num_pixels >= min_size
        ):
            filtered_cell_props.append(prop)
        else:
            # Remove edge cells from the mask
            mask[cell_labels == prop.label] = False

    # Apply the mask to remove edge cells from the labeled image
    filtered_cell_labels = cell_labels * mask
    return filtered_cell_props, filtered_cell_labels


def get_plot_red_intensity_per_cell(
    filtered_cell_props: list, filtered_cell_labels: ndarray, original_image: ndarray, plot_path: str
) -> str:
    """
    Calculate and plot the mean red intensity within each cell region.

    Args:
        filtered_cell_props (list): Properties of the filtered cells.
        filtered_cell_labels (ndarray): Labeled image of the filtered cells.
        original_image (ndarray): The original input image.
        plot_path (str): Path to save the plot.

    Returns:
        list: A list of mean red intensities for each cell.
    """
    red_channel = original_image[:, :, 0]

    # Initialize an image to visualize the used intensity regions
    intensity_mask = np.zeros_like(red_channel)
    red_intensities = []

    for prop in filtered_cell_props:
        # Create a mask for the current cell
        cell_mask = filtered_cell_labels == prop.label

        # Add the mask to the intensity mask for visualization
        intensity_mask[cell_mask] = red_channel[cell_mask]

        # Calculate the mean intensity of the red channel within the cell
        mean_intensity = red_channel[cell_mask].mean() / 255
        red_intensities.append(mean_intensity)

    # Plot the regions used for intensity calculation over the red channel
    plt.imshow(intensity_mask / 255, cmap=cm.hot.reversed(), alpha=0.5, vmin=0, vmax=1)
    plt.title("Regions Used for Red Intensity Calculation")
    plt.colorbar(label="Red Intensity")
    plt.axis("off")

    # Number the filtered cells in the plot
    for i, prop in enumerate(filtered_cell_props, start=1):
        y, x = prop.centroid
        plt.text(x, y, str(i), color="blue", fontsize=12, ha="center", va="center")

    plt.savefig(plot_path)
    plt.clf()

    return red_intensities


def analyze_image(
    full_file_path: str, output_dir_path: str, blue_area_threshold: int, min_blue_size: int
) -> list:
    """
    Analyze an image to calculate the red intensity per cell and save the results.

    Args:
        full_file_path (str): The full path to the input image file.
        output_dir_path (str): The directory path to save output files.
        blue_area_threshold (int): The maximum size of a hole to be filled.

    Returns:
        list: A row containing the filename and the calculated red intensities per cell.
    """
    table_row = None
    file = os.path.basename(full_file_path)

    if full_file_path.endswith(".tif") and "_intensity.tif" not in full_file_path:
        # Step 1: Load the entire image
        image = skimage.io.imread(full_file_path)

        # Step 2: Get cell props and labels of cells that are not border cells
        filtered_cell_props, filtered_cell_labels = get_filtered_cell_props(
            image, blue_area_threshold, min_blue_size
        )

        # Step 3: Calculate red intensities per cell and plot the results
        red_intensity_per_cell_list = get_plot_red_intensity_per_cell(
            filtered_cell_props,
            filtered_cell_labels,
            image,
            output_dir_path / file.replace(".tif", "_intensity.tif"),
        )

        table_row = [file] + red_intensity_per_cell_list

    return table_row


def new_or_update_csv(output_dir_path: str, all_table_rows: list) -> None:
    """
    Create or update a CSV file with the results of red intensity calculations.

    Args:
        output_dir_path (str): The directory path where the CSV file will be saved.
        all_table_rows (list): A list of rows containing the results.
    """
    csv_file = os.path.join(output_dir_path, "intensities.csv")
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

    # Determine the maximum number of cells in any row
    max_cells = max(len(data) for data in existing_data.values())

    # Create a new header based on the maximum number of cells
    header = ["filename"] + list(map(str, range(1, max_cells + 1)))

    # Write the updated data back to the CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for filename, data in existing_data.items():
            padded_data = data + [""] * (max_cells - len(data) + 1)
            writer.writerow([filename] + padded_data)


def main(args) -> None:
    """
    Main function to process images and calculate red intensities per cell.

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
            table_row = analyze_image(
                full_file_path, output_dir_path, args.blue_area_threshold, args.min_cell_size
            )
            if table_row:
                all_table_rows.append(table_row)
    else:
        table_row = analyze_image(args.input, output_dir_path, args.blue_area_threshold, args.min_cell_size)
        if table_row:
            all_table_rows.append(table_row)

    # Update or create the CSV file with the collected data
    new_or_update_csv(output_dir_path, all_table_rows)


if __name__ == "__main__":
    # Argument Parsing
    description = (
        "Calulates the mean intensity of red per cell (DAPI/blue signal). "
        "Can be called with a directory or individual image file. "
        "If the cells contain holes, try increasing -blue_area_threshold."
    )
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
        help="Path to the directory where the intensities.csv and annotated .tifs (annotated cells + red intensity, which were counted; default: input_dir)",
        required=False,
    )
    parser.add_argument(
        "-min_cell_size",
        dest="min_cell_size",
        default=1450,
        type=int,
        help="Minimal number of pixels a cell has to have to be counted (default: 1450)",
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
