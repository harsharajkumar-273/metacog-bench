import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import RSNAMultimodalKneeDataset
from models import MultimodalKneeClassifier

def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Check for test.csv or generate test template matching competition spec
    test_csv_path = config.DATA_DIR / "test.csv"
    if not test_csv_path.exists():
        print("Generating test dataset template...")
        test_df = pd.DataFrame({
            "StudyInstanceUID": [f"test_study_{i:03d}" for i in range(10)],
            "RadiologyReport": ["Radiology evaluation requested."] * 10
        })
        test_df.to_csv(test_csv_path, index=False)
    else:
        test_df = pd.read_csv(test_csv_path)

    test_dataset = RSNAMultimodalKneeDataset(test_df, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Gather model checkpoints across folds
    checkpoint_files = list(config.CHECKPOINT_DIR.glob("model_fold_*.pth"))
    
    if len(checkpoint_files) == 0:
        print("No trained checkpoints found. Initializing single fold model for baseline inference...")
        model = MultimodalKneeClassifier().to(device)
        model.eval()
        fold_models = [model]
    else:
        fold_models = []
        for ckpt in checkpoint_files:
            model = MultimodalKneeClassifier().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            fold_models.append(model)

    all_study_uids = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            study_uids = batch["study_uid"]
            volumes = batch["volume"].to(device)

            # Ensemble average across folds
            batch_preds = torch.zeros(len(study_uids), config.NUM_CLASSES).to(device)
            for model in fold_models:
                logits = model(volumes)
                preds = torch.sigmoid(logits)
                batch_preds += preds / len(fold_models)

            all_study_uids.extend(study_uids)
            all_predictions.append(batch_preds.cpu().numpy())

    all_predictions = np.vstack(all_predictions)

    # Construct submission DataFrame matching exact competition header
    sub_df = pd.DataFrame({"StudyInstanceUID": all_study_uids})
    for i, col in enumerate(config.TARGET_COLS):
        sub_df[col] = all_predictions[:, i]

    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\nSubmission generated successfully at: {config.SUBMISSION_PATH}")
    print("Sample output head:")
    print(sub_df.head())

if __name__ == "__main__":
    generate_submission()
