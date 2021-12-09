from mmdet.apis import init_detector, inference
import argparse
import glob
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    test = 'dataset/test/*'
    train = 'dataset/train/*'
    results = (test, inference(model, glob.glob(test)))
    with open('test.pickle', 'wb') as handle:
        pickle.dump(results, handle)

if __name__ == '__main__':
    main()
