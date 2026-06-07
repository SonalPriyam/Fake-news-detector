import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import timm
    from torch.utils.data import Subset
    from dataset import FakedditDataset
    from model import MultimodalFakeNewsClassifier
    from train import train_model

    print("Loading dataset...")
    dataset = FakedditDataset()

    # use 10k samples for fusion model (more than baseline)
    small_dataset = Subset(dataset, range(10000))

    print("Building fusion model...")
    model = MultimodalFakeNewsClassifier(num_classes=2)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    train_model(
        model=model,
        dataset=small_dataset,
        epochs=5,
        batch_size=16,
        lr=2e-5,
        save_path=os.path.join(
            os.path.dirname(__file__),
            "../backend/models/fusion_model.pt"
        )
    )