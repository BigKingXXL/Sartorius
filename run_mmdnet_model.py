from mmdet.apis import init_detector, inference_detector
import argparse
import glob
import os
import cupy as numpy
import pandas as pd

def compute_confidence(current_file, results):
    class0 = results[current_file][1][0] if len(results[current_file][1][0]) > 1 else []
    class1 = results[current_file][1][1] if len(results[current_file][1][1]) > 1 else []
    class2 = results[current_file][1][2] if len(results[current_file][1][2]) > 1 else []
    confidences = [numpy.sum(el, axis=1) for el in results[current_file][0]]
    class0_confident = [element.astype(float) * confidences[0][index] for index, element in enumerate(class0)]
    class1_confident = [element.astype(float) * confidences[1][index] for index, element in enumerate(class1)]
    class2_confident = [element.astype(float) * confidences[2][index] for index, element in enumerate(class2)]
    cell_confident = class0_confident + class1_confident + class2_confident
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
    cell_confident = [element.astype(bool) for element in cell_confident]
    return cell_confident

def run_length_encode(id: str, masks):
    rles = []
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
        flat_mask = mask.flatten()
        if numpy.sum(flat_mask) <= max(mean - 2 * std_sizes, 10):
            print("skipped")
            continue
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
    return pd.DataFrame.from_dict({
        "id": [id for _ in rles],
        "predicted": rles
    })

def convert_to_masks(predictions_table):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--input", type=str)
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
        results = inference_detector(model, [file])
        result_pd = run_length_encode(id, compute_confidence(0, results))
        if not result is None:
            result = result.append(result_pd, ignore_index=True)
        else:
            result = result_pd
    result.to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == '__main__':
    main()
