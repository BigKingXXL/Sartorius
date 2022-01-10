from mmdet.apis import init_detector, inference_detector
import argparse
import glob
import os
import cupy as numpy
import pandas as pd

def remove_overlap_by_confidence(current_file, results):
    """
    This method removes overlap of multiple cells based on the confidence of the model.
    Whenever two cells overlap, the overlapping pixels are assigned to the cell with a higher confidence.
    This descision is made individually for every pixel.

    Parameters:
        current_file (int): It of the file in the results tensor for which we are computing the confidence values.
                            We can only work on single images. Therefore we need to select an image with a batch size greater than 1.
                            When using batch size 1 this should be 0.
        results           : Result tensor produced by mmdetection models. This contains the predictions.

    Returns:
        cell_masks (list) : Returns a list of 2D arrays with the annotations for each individual cell withour overlap.
    """

    # Extract the predictions for every cell type.
    class0 = results[current_file][1][0] if len(results[current_file][1][0]) > 1 else []
    class1 = results[current_file][1][1] if len(results[current_file][1][1]) > 1 else []
    class2 = results[current_file][1][2] if len(results[current_file][1][2]) > 1 else []

    # Combine confidence matrices and masks.
    # Each mask is a 2D array with ones where the cell was detected and zeros for background.
    # Additionally we get a confidence value for every pixel.
    # We are only interested in the confidence values at every pixel where a cell was detected and remove the confidence values for background.
    confidences = [numpy.sum(el, axis=1) for el in results[current_file][0]]
    class0_confident = [element.astype(float) * confidences[0][index] for index, element in enumerate(class0)]
    class1_confident = [element.astype(float) * confidences[1][index] for index, element in enumerate(class1)]
    class2_confident = [element.astype(float) * confidences[2][index] for index, element in enumerate(class2)]
    cell_confident = class0_confident + class1_confident + class2_confident

    # Keep only the mask with the highest confidence for every pixel.
    for x in range(cell_confident[0].shape[0]):
        for y in range(cell_confident[0].shape[1]):
            max_index = 0
            max_value = cell_confident[0][x][y]
            for index, element in enumerate(cell_confident[1:]):
                if element[x][y] > max_value:
                    max_index = index + 1
                    max_value = element[x][y]
            for index, element in enumerate(cell_confident):
                if max_index != index:
                    element[x][y] = 0
                else:
                    element[x][y] = min(1, element[x][y])

    # Convert the cleaned up masks back to binary
    cell_masks = [element.astype(bool) for element in cell_confident]
    return cell_masks

def run_length_encode(id: str, masks):
    """
    This method converts 2D cell masks into a dataframe of run length encoded annotations for every cell.

    Parameters:
        id (str)        : Id of the current file which the masks belong to.
        masks (list)    : A list of 2D arrays with the annotations for each individual cell.

    Returns:
        pandas.Dataframe: A dataframe with two columns: "id" contains the current file id, "predicted" contains the RLE cell masks.
    """

    rles = []
    # We computed the mean and std of the area of each predicted cell. We later use this to filter our small predictions / noise.
    sizes = numpy.array([numpy.sum(el) for el in masks])
    std_sizes = numpy.std(sizes)
    mean = numpy.mean(sizes)
    print(f"mean: {mean}")
    print(f"std: {std_sizes}")
    print(f"threshold: {(mean - 2 * std_sizes)}")
    for mask in masks:
        rle = []
        start = 0
        in_mask = False
        # 1D representation of the cell mask
        flat_mask = mask.flatten()

        # Filter out noise
        if numpy.sum(flat_mask) <= max(mean - 2 * std_sizes, 10):
            print("skipped")
            continue

        # RLE encode the 1D mask by saving the start and stop index of annotations.
        for index, el in enumerate(flat_mask):
            if el:
                if not in_mask:
                    in_mask = True
                    start = index
                if index == len(flat_mask) - 1 or not flat_mask[index + 1]:
                    rle.append(str(start + 1))
                    rle.append(str(index - start + 2))
            else:
                in_mask = False
                    
        rles.append(" ".join(rle))

    # Convert the individual RLE encodings in a dataframe matching the submission format.
    return pd.DataFrame.from_dict({
        "id": [id for _ in rles],
        "predicted": rles
    })

def convert_to_masks(predictions_table):
    """
    This method is the opposite of "run_length_encode()" and convertes a dataframe back to the individual 2D cell masks.

    Parameters:
        predictions_table (pandas.Dataframe): A dataframe with two columns: "id" contains the current file id, "predicted" contains the RLE cell masks.

    Returns:
        results (list)                      : A list of 2D arrays with the annotations for each individual cell.
    
    """
    results = []
    for _, picture_group in predictions_table.groupby('id'):
        picture_id = picture_group.iloc[0]['id']
        picture_size = 704*520
        cell_array = numpy.zeros(picture_size, dtype=int)
        for index, cell in picture_group.iterrows():
            predicted = [int(val) for val in cell['predicted'].split(' ')]
            for index in range(0, len(predicted), 2):
                cell_array[predicted[index] - 1:predicted[index]+predicted[index+1] - 1 - 1] += 1
        results.append(cell_array.reshape(520, 704))
    return results

def main():
    """
    This file performs inference with a mmdetection model.
    Some methods are needed to convert the model outputs into the desired output for the competition.
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the mmdetection model config")
    parser.add_argument("--checkpoint", type=str, help="Path to the pretrained mmdetection model weights")
    parser.add_argument("--input", type=str, help="Path to the image filder to perform inference on")
    args = parser.parse_args()

    print("Loading model")
    model = init_detector(args.config, args.checkpoint, device='cuda')
    print("Loaded model")
    files = glob.glob(os.path.join(args.input, "*"))
    files.sort()
    ids = [file.split('/')[-1].split(".")[0] for file in files]
    ids
    result = None
    for file, id in zip(files, ids):
        print(file)

        # Run the mmdetection inference script with the path to one image.
        results = inference_detector(model, [file])

        # Remove overlap and RLE the predictions.
        result_pd = run_length_encode(id, remove_overlap_by_confidence(0, results))

        # Combine the results of each file
        if not result is None:
            result = result.append(result_pd, ignore_index=True)
        else:
            result = result_pd

    # Save the submission dataframe.
    result.to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == '__main__':
    main()
