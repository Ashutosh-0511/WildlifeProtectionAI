from ml.behavior import VideoMAEBehaviorClassifier

if __name__ == "__main__":
    model = VideoMAEBehaviorClassifier()
    print("VIDEOMAE_MODEL_OK")
    print("epoch=", model.epoch)
    print("val_macro_f1=", model.val_macro_f1)
    print("num_frames=", model.num_frames)
