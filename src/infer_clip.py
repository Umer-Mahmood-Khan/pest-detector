import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, conf=args.conf, save=True)

if __name__ == "__main__":
    main()
