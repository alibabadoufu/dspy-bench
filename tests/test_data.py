"""Tests for data loading and management."""

import json
import pytest
import tempfile
from pathlib import Path

from dspy_bench.data import JSONLDataset, DatasetExample, DatasetSplit, SplitsManager


class TestDatasetExample:
    """Test dataset example schema."""

    def test_dataset_example_creation(self):
        """Test creating dataset examples."""
        example = DatasetExample(
            input="Test input",
            label="Test label",
            id="test_1"
        )
        assert example.input == "Test input"
        assert example.label == "Test label"
        assert example.id == "test_1"

    def test_dataset_example_with_metadata(self):
        """Test dataset example with metadata."""
        example = DatasetExample(
            input="Test input",
            label="Test label",
            metadata={"source": "test", "difficulty": "easy"}
        )
        assert example.metadata["source"] == "test"
        assert example.metadata["difficulty"] == "easy"


class TestJSONLDataset:
    """Test JSONL dataset loader."""

    def test_load_valid_jsonl(self):
        """Test loading valid JSONL file."""
        # Create temporary JSONL file
        test_data = [
            {"input": "What is 2+2?", "label": "4", "id": "q1"},
            {"input": "What is 3+3?", "label": "6", "id": "q2"},
            {"input": "What is 4+4?", "label": "8", "id": "q3"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            # Load dataset
            dataset = JSONLDataset.load(temp_path, input_key="input", label_key="label", id_key="id")

            assert len(dataset) == 3
            assert dataset[0].input == "What is 2+2?"
            assert dataset[0].label == "4"
            assert dataset[0].id == "q1"

        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            JSONLDataset.load("nonexistent.jsonl")

    def test_load_invalid_jsonl(self):
        """Test loading invalid JSONL file."""
        # Create file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input": "test", "label": "answer"}\n')
            f.write('invalid json line\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                JSONLDataset.load(temp_path, input_key="input", label_key="label")
        finally:
            Path(temp_path).unlink()

    def test_load_missing_required_fields(self):
        """Test loading JSONL with missing required fields."""
        test_data = [
            {"input": "Test input"},  # Missing label
            {"label": "Test label"},  # Missing input
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required input key"):
                JSONLDataset.load(temp_path, input_key="input", label_key="label")
        finally:
            Path(temp_path).unlink()

    def test_dataset_validation(self):
        """Test dataset validation."""
        # Create test dataset
        examples = [
            DatasetExample(input="test1", label="label1", id="id1"),
            DatasetExample(input="test2", label="label2", id="id1"),  # Duplicate ID
            DatasetExample(input="", label="label3"),  # Empty input
        ]
        dataset = JSONLDataset(examples)

        result = dataset.validate()
        assert not result.is_valid  # Should have warnings/errors
        assert len(result.warnings) > 0  # Should warn about duplicate IDs and empty inputs

    def test_dataset_split(self):
        """Test dataset splitting."""
        # Create test dataset with known data
        examples = [
            DatasetExample(input=f"input_{i}", label=f"label_{i}")
            for i in range(10)
        ]
        dataset = JSONLDataset(examples)

        train_split, val_split, test_split = dataset.split(
            seed=42,
            train=0.6,
            val=0.2,
            test=0.2
        )

        assert len(train_split) == 6
        assert len(val_split) == 2
        assert len(test_split) == 2
        assert train_split.size == 6
        assert val_split.size == 2
        assert test_split.size == 2

    def test_dataset_sampling(self):
        """Test dataset sampling."""
        examples = [
            DatasetExample(input=f"input_{i}", label=f"label_{i}")
            for i in range(100)
        ]
        dataset = JSONLDataset(examples)

        # Sample 10 examples
        sampled = dataset.sample(10, seed=42)
        assert len(sampled) == 10

        # Sample more than available
        sampled_all = dataset.sample(200, seed=42)
        assert len(sampled_all) == 100  # Should return all

    def test_dataset_save(self):
        """Test dataset saving."""
        examples = [
            DatasetExample(input="test_input", label="test_label", id="test_id"),
            DatasetExample(input="test_input2", label="test_label2"),
        ]
        dataset = JSONLDataset(examples, input_key="input", label_key="label", id_key="id")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            dataset.save(temp_path)

            # Load and verify
            loaded_dataset = JSONLDataset.load(temp_path, input_key="input", label_key="label", id_key="id")
            assert len(loaded_dataset) == 2
            assert loaded_dataset[0].input == "test_input"
            assert loaded_dataset[0].id == "test_id"

        finally:
            Path(temp_path).unlink()


class TestSplitsManager:
    """Test dataset splits manager."""

    def test_create_and_save_splits(self):
        """Test creating and saving splits."""
        # Create test dataset
        examples = [
            DatasetExample(input=f"input_{i}", label=f"label_{i}")
            for i in range(10)
        ]
        dataset = JSONLDataset(examples)

        manager = SplitsManager(dataset)
        splits = manager.create_splits(seed=42, train=0.6, val=0.2, test=0.2)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 6
        assert len(splits["val"]) == 2
        assert len(splits["test"]) == 2

        # Save splits
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager.save_splits(tmp_dir)

            # Verify files exist
            assert Path(tmp_dir).exists()
            assert (Path(tmp_dir) / "dataset_train.jsonl").exists()
            assert (Path(tmp_dir) / "dataset_val.jsonl").exists()
            assert (Path(tmp_dir) / "dataset_test.jsonl").exists()

    def test_get_split_stats(self):
        """Test getting split statistics."""
        examples = [
            DatasetExample(input=f"input_{i}", label=f"label_{i}")
            for i in range(10)
        ]
        dataset = JSONLDataset(examples)

        manager = SplitsManager(dataset)
        splits = manager.create_splits(train=0.7, val=0.2, test=0.1)

        stats = manager.get_split_stats()
        assert stats["train"]["size"] == 7
        assert stats["val"]["size"] == 2
        assert stats["test"]["size"] == 1
        assert abs(stats["train"]["percentage"] - 70.0) < 0.1