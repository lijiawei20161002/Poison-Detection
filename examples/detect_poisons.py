#!/usr/bin/env python3
"""
Example script for detecting poisoned samples using influence functions.

This script demonstrates the complete pipeline:
1. Load model and data
2. Compute influence scores
3. Detect poisoned samples
4. Evaluate detection performance
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from poison_detection.config import Config
from poison_detection.data import DataLoader as PoisonDataLoader
from poison_detection.data import DataPreprocessor, InstructionDataset
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask
from poison_detection.detection import PoisonDetector, DetectionMetrics
from poison_detection.utils import load_model_and_tokenizer, save_clean_dataset


def main():
    """Main detection pipeline."""

    # 1. Setup configuration
    config = Config(
        train_data_path="/path/to/poison_train.jsonl",
        test_data_path="/path/to/test_data.jsonl",
        model_path="/path/to/checkpoint.pt",
        poisoned_indices_path="/path/to/poisoned_indices.txt",
        output_dir="./outputs",
        num_test_samples=50,
        detection_method="delta_scores",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("POISON DETECTION PIPELINE")
    print("=" * 80)

    # 2. Load model and tokenizer
    print("\n[1/7] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        checkpoint_path=config.model_path,
        device=config.device
    )
    print(f"✓ Loaded model: {config.model_name}")

    # 3. Load and preprocess data
    print("\n[2/7] Loading data...")
    train_loader = PoisonDataLoader(config.train_data_path)
    test_loader = PoisonDataLoader(config.test_data_path)

    train_samples = train_loader.load()
    test_samples = test_loader.load()

    print(f"✓ Loaded {len(train_samples)} training samples")
    print(f"✓ Loaded {len(test_samples)} test samples")

    # Select subset of test samples
    if config.num_test_samples:
        test_indices = test_loader.get_top_n_by_countnorm(config.num_test_samples)
        test_samples = [test_samples[i] for i in test_indices]
        print(f"✓ Selected top {len(test_samples)} test samples by countnorm")

    # Preprocess data
    preprocessor = DataPreprocessor(tokenizer)
    train_inputs, train_labels, train_label_spaces = preprocessor.preprocess_samples(train_samples)
    test_inputs, test_labels, test_label_spaces = preprocessor.preprocess_samples(test_samples)

    # Create PyTorch datasets
    train_dataset = InstructionDataset(
        train_inputs, train_labels, train_label_spaces,
        tokenizer, config.max_input_length, config.max_output_length
    )
    test_dataset = InstructionDataset(
        test_inputs, test_labels, test_label_spaces,
        tokenizer, config.max_input_length, config.max_output_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. Compute original influence scores
    print("\n[3/7] Computing original influence scores...")
    task = ClassificationTask(device=config.device)
    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name="original",
        output_dir=config.output_dir / "influence_results"
    )

    original_scores = analyzer.run_full_analysis(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        compute_factors=config.compute_factors,
        csv_path=config.output_dir / "original_influence_scores.csv"
    )
    print(f"✓ Computed influence scores for {len(original_scores)} training samples")

    # 5. Compute negative influence scores
    print("\n[4/7] Computing negative influence scores...")
    negative_test_samples = preprocessor.create_negative_samples(
        test_samples,
        method=config.negative_method
    )
    neg_test_inputs, neg_test_labels, neg_test_label_spaces = preprocessor.preprocess_samples(
        negative_test_samples
    )

    neg_test_dataset = InstructionDataset(
        neg_test_inputs, neg_test_labels, neg_test_label_spaces,
        tokenizer, config.max_input_length, config.max_output_length
    )
    neg_test_dataloader = DataLoader(neg_test_dataset, batch_size=1, shuffle=False)

    analyzer_neg = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name="negative",
        output_dir=config.output_dir / "influence_results"
    )

    negative_scores = analyzer_neg.run_full_analysis(
        train_loader=train_dataloader,
        test_loader=neg_test_dataloader,
        compute_factors=False,  # Reuse factors from original
        csv_path=config.output_dir / "negative_influence_scores.csv"
    )
    print(f"✓ Computed negative influence scores")

    # 6. Detect poisoned samples
    print("\n[5/7] Detecting poisoned samples...")

    # Load ground truth if available
    poisoned_indices = None
    if config.poisoned_indices_path:
        poisoned_indices = set(PoisonDataLoader.load_indices_file(config.poisoned_indices_path))
        print(f"✓ Loaded {len(poisoned_indices)} ground truth poison indices")

    # Convert tensors to list of tuples
    original_scores_list = [(i, score.item()) for i, score in enumerate(original_scores)]
    negative_scores_list = [(i, score.item()) for i, score in enumerate(negative_scores)]

    # Create detector
    detector = PoisonDetector(
        original_scores=original_scores_list,
        negative_scores=negative_scores_list,
        poisoned_indices=poisoned_indices
    )

    # Detect poisons using configured method
    if config.detection_method == "delta_scores":
        detected = detector.detect_by_delta_scores(
            positive_threshold=config.positive_threshold,
            negative_threshold=config.negative_threshold
        )
    elif config.detection_method == "threshold":
        detected = detector.detect_by_threshold(threshold=config.threshold)
    elif config.detection_method == "zscore":
        detected = detector.detect_by_zscore(z_threshold=config.z_threshold)
    elif config.detection_method == "clustering":
        detected = detector.detect_by_clustering()
    else:
        raise ValueError(f"Unknown detection method: {config.detection_method}")

    print(f"✓ Detected {len(detected)} suspicious samples using {config.detection_method}")

    # Save detected indices
    detector.save_detected_indices(
        detected,
        config.output_dir / "detected_poisons.txt"
    )

    # 7. Evaluate detection performance
    print("\n[6/7] Evaluating detection performance...")
    if poisoned_indices:
        metrics = detector.evaluate_detection(detected)

        print("\nDetection Results:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  Precision:       {metrics['precision']:.3f}")
        print(f"  Recall:          {metrics['recall']:.3f}")
        print(f"  F1 Score:        {metrics['f1_score']:.3f}")
        print(f"  Accuracy:        {metrics['accuracy']:.3f}")

        # Save metrics
        import json
        with open(config.output_dir / "detection_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    # 8. Create clean dataset
    print("\n[7/7] Creating clean dataset...")
    detected_indices = {idx for idx, _ in detected}
    save_clean_dataset(
        input_path=config.train_data_path,
        output_path=config.output_dir / "clean_train.jsonl",
        indices_to_remove=detected_indices
    )

    print("\n" + "=" * 80)
    print("DETECTION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
