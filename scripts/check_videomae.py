from ml.behavior import VideoMAEBehaviorClassifier

if __name__ == "__main__":
    model = VideoMAEBehaviorClassifier()
    print("VIDEOMAE_MODEL_OK")
    print(f"epoch={model.epoch}")
    print(f"val_macro_f1={model.val_macro_f1}")
    print(f"num_frames={model.num_frames}")
    print(f"device={model.device}")
