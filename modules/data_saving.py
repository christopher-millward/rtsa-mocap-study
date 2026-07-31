"""Module to save processed data"""
import pickle


def save_data_to_pickle(data, file_path):
    """Saves the given data object to a pickle file at the specified file path.


    Args:
        data: The data object to be saved.
        file_path: The path to the pickle file where the data will be saved.

    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def save_heatmap_to_csv(all_heatmap_data, file_path):
    """Saves the heatmap data for all participants into a single CSV file.

    The CSV will hav the following columns:
        - fname: The filename of the participant's data.
        - side: The side of the body (left or right).
        - unit: The unit of measurement for the heatmap data.
        - bin_elevation_start: The starting elevation angle for the heatmap.
        - bin_elevation_end: The ending elevation angle for the heatmap.
        - bin_poe_start: The starting point of the point of interest (POE) for the heatmap.
        - bin_poe_end: The ending point of the point of interest (POE) for the heatmap.
        - sum_elevation: The elevation angle for the heatmap.
        - sum_poe: The point of interest (POE) for the heatmap.
        - sum_ir_er: The internal rotation to external rotation (IR-ER) for the heatmap.
        - cumulative_motion: The cumulative motion for the heatmap.
        - sample_count: The number of samples in the heatmap.


    Args:
        all_heatmap_data: The heatmap data to be saved (should be in a format compatible with CSV).
        file_path: The path to the CSV file where the heatmap data will be saved.

    Returns:
        None
    """
    raise NotImplementedError("This function is not yet implemented. Please implement the logic to save heatmap data to CSV.")
