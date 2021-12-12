from mmdet.apis import init_detector, inference_detector
import argparse
import glob
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    print("loading model")
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    test = 'dataset/test/*'
    print(glob.glob(test))
    results = (test, inference_detector(model, glob.glob(test)))
    print("Writing out")
    with open('test.pickle', 'wb') as handle:
        pickle.dump(results, handle)

if __name__ == '__main__':
    main()
