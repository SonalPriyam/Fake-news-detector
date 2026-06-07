import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    from torch.utils.data import Subset
    from dataset import FakedditDataset
    from bert_baseline import BertBaseline
    from train import train_model

    dataset = FakedditDataset()

    small_dataset = Subset(dataset, range(5000))

    model = BertBaseline(num_classes=2)

    train_model(
        model=model,
        dataset=small_dataset,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        save_path=os.path.join(os.path.dirname(__file__), "../backend/models/best_model.pt")
    )