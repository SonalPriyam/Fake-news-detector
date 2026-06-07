import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import FakedditDataset


dataset = FakedditDataset()

print(f"Total samples: {len(dataset)}")
print(f"Columns in TSV: {list(dataset.df.columns)}")
print(f"Label distribution:\n{dataset.df['2_way_label'].value_counts()}")


sample = dataset[0]
print(f"\ninput_ids shape:      {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")
print(f"pixel_values shape:   {sample['pixel_values'].shape}")
print(f"label:                {sample['label']}")
print(f"clean_title:          {dataset.df.iloc[0]['clean_title']}")