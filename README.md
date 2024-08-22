# Image Analysis Scripts

## Overview

This repository contains two Python scripts designed to analyze biological images, focusing specifically on identifying and quantifying red dots or red intensity within cells stained with a blue dye (DAPI). These scripts are intended to process `.tif` images and generate outputs that include CSV files with the analysis results and visual plots of the processed images.

## Scripts

### 1. `count_red_dots.py`

#### Purpose
This script processes images to count the number of red dots within each cell, which is identified using the blue channel. The results are saved as annotated `.tif` images and a CSV file summarizing the counts.

#### Key Functionality
This script is used to count the number of red dots within each cell in an image. It processes images by:

- Identifying and labeling cells based on their blue channel intensity.
- Counting the number of red dots within each identified cell.
- Generating and saving annotated images that show the identified cells and the red dots within them.
- Outputting the counts in a CSV file.

#### Usage
To run the script, use the following command:

```bash
python count_red_dots.py -i <input_path> [-o <output_dir>] [options]
```

Arguments:

- -i: Path to the directory containing .tif files or an individual .tif file.
- -o: Path to the directory where the output files (CSV and annotated images) will be saved.
- -red_bg_threshold: Adjustment for red background noise.
- -blue_bg_threshold: Adjustment for blue background noise.
- -min_cell_size: Minimum size of a cell to be counted.
- -red_max_size: Maximum size of a red dot to be considered.
- -blue_hole_threshold: Maximum size of a hole in a cell area to be filled.

#### Output
Annotated Images: Images with cells and red dots marked, saved in the output directory.
CSV File: A counts.csv file containing the filename, mean number of red dots per cell, and the count of red dots for each cell.

### 2. calculate_red_intensity.py

#### Purpose
The calculate_red_intensity.py script calculates the mean red intensity within each cell in an image. This script processes individual .tif images or directories containing multiple .tif files and outputs the results in a CSV file.

#### Key Functions
This script calculates the mean red intensity within each cell. It performs the following tasks:

- Identifying and labeling cells based on their blue channel intensity.
- Calculating the mean red intensity within each labeled cell.
- Generating and saving annotated images showing the regions used for intensity calculation.
- Outputting the intensity values in a CSV file.

#### Usage
To run the script, use the following command:

```bash
python calculate_red_intensity.py -i <input_path> [-o <output_dir>] [options]
```

Arguments:

- -i: Path to the directory containing .tif files or an individual .tif file.
- -o: Path to the directory where the output files (CSV and annotated images) will be saved.
- -red_bg_threshold: Adjustment for red background noise.
- -blue_bg_threshold: Adjustment for blue background noise.
- -min_cell_size: Minimum size of a cell to be counted.
- -red_max_size: Maximum size of a red dot to be considered.
- -blue_hole_threshold: Maximum size of a hole in a cell area to be filled.

#### Output
Annotated Images: Images with the regions used for intensity calculations marked, saved in the output directory.
CSV File: An intensities.csv file containing the filename and the mean red intensities for each cell.


## Dependencies
- Python 3.x
- numpy
- skimage
- matplotlib
- scipy

You can install the required dependencies using the following command:
```bash
pip install numpy scikit-image matplotlib scipy
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
