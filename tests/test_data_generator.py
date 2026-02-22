"""Tests for the transformer dataset generation pipeline and tokenizer."""

import json
import os

import numpy as np
import pytest

from ac_solver.envs.utils import is_array_valid_presentation
from ac_solver.transformer.data_generator import (
    apply_random_ac_moves,
    generate_dataset_for_presentation,
    get_word_lengths,
    load_progress,
    merge_shards,
    resize_presentation_np,
    save_progress,
    save_shard,
)
from ac_solver.transformer.tokenizer import (
    TOKEN_EOS,
    TOKEN_SEP,
    TOKEN_X,
    TOKEN_X_INV,
    TOKEN_Y,
    TOKEN_Y_INV,
    VOCAB_SIZE,
    presentation_to_tokens,
    tokens_to_presentation,
)


# --- resize_presentation_np tests ---


class TestResizePresentationNp:
    def test_expand(self):
        """Expanding from max_rel=4 to max_rel=8 preserves content."""
        pres = np.array([1, 2, 0, 0, -1, -2, 0, 0], dtype=np.int8)
        resized = resize_presentation_np(pres, old_max=4, new_max=8)
        assert len(resized) == 16
        # r0 should be [1, 2, 0, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(resized[:2], [1, 2])
        assert np.all(resized[2:8] == 0)
        # r1 should be [-1, -2, 0, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(resized[8:10], [-1, -2])
        assert np.all(resized[10:16] == 0)

    def test_shrink(self):
        """Shrinking from max_rel=8 to max_rel=4 preserves content."""
        pres = np.array(
            [1, 2, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0], dtype=np.int8
        )
        resized = resize_presentation_np(pres, old_max=8, new_max=4)
        assert len(resized) == 8
        np.testing.assert_array_equal(resized, [1, 2, 0, 0, -1, -2, 0, 0])

    def test_identity(self):
        """Resizing to same size returns equal array."""
        pres = np.array([1, -2, 0, 0, 2, -1, 0, 0], dtype=np.int8)
        resized = resize_presentation_np(pres, old_max=4, new_max=4)
        np.testing.assert_array_equal(pres, resized)

    def test_round_trip(self):
        """Expand then shrink returns original."""
        pres = np.array([1, 2, -1, 0, -2, 1, 0, 0], dtype=np.int8)
        expanded = resize_presentation_np(pres, old_max=4, new_max=10)
        shrunk = resize_presentation_np(expanded, old_max=10, new_max=4)
        np.testing.assert_array_equal(pres, shrunk)

    def test_shrink_too_small_raises(self):
        """Shrinking below actual word length raises assertion."""
        pres = np.array([1, 2, -1, 0, -2, 1, 0, 0], dtype=np.int8)
        with pytest.raises(AssertionError):
            resize_presentation_np(pres, old_max=4, new_max=2)

    def test_dtype_preserved(self):
        """Output is always int8."""
        pres = np.array([1, 0, -2, 0], dtype=np.int8)
        resized = resize_presentation_np(pres, old_max=2, new_max=5)
        assert resized.dtype == np.int8


# --- get_word_lengths tests ---


class TestGetWordLengths:
    def test_basic(self):
        pres = np.array([1, 2, 0, 0, -1, -2, -1, 0], dtype=np.int8)
        assert get_word_lengths(pres, 4) == [2, 3]

    def test_full_length(self):
        pres = np.array([1, 2, -1, 2, -2, 1], dtype=np.int8)
        assert get_word_lengths(pres, 3) == [3, 3]

    def test_minimal(self):
        pres = np.array([1, 0, 0, 2, 0, 0], dtype=np.int8)
        assert get_word_lengths(pres, 3) == [1, 1]


# --- apply_random_ac_moves tests ---


class TestApplyRandomACMoves:
    def test_preserves_validity(self):
        """After applying random moves, presentation is still valid."""
        pres = np.array(
            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int8,
        )
        max_rel = 18
        lengths = get_word_lengths(pres, max_rel)
        rng = np.random.default_rng(42)

        result_pres, result_lengths = apply_random_ac_moves(
            pres, max_rel, lengths, 100, rng
        )

        assert is_array_valid_presentation(result_pres)
        assert result_lengths == get_word_lengths(result_pres, max_rel)

    def test_deterministic_with_same_seed(self):
        """Same seed produces same result."""
        pres = np.array([1, 2, 0, 0, -2, 1, 0, 0], dtype=np.int8)
        max_rel = 4
        lengths = [2, 2]

        rng1 = np.random.default_rng(123)
        result1, _ = apply_random_ac_moves(pres.copy(), max_rel, lengths.copy(), 50, rng1)

        rng2 = np.random.default_rng(123)
        result2, _ = apply_random_ac_moves(pres.copy(), max_rel, lengths.copy(), 50, rng2)

        np.testing.assert_array_equal(result1, result2)


# --- generate_dataset_for_presentation tests ---


class TestGenerateDatasetForPresentation:
    def test_output_shape(self):
        """Output has correct number of presentations."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {
            "n_phases": 3,
            "n_chains": 2,
            "n_moves": 10,
            "lmax": 32,
            "seed": 42,
        }
        presentations, metadata = generate_dataset_for_presentation((0, p0, config))

        expected_count = config["n_phases"] * config["n_chains"]
        assert presentations.shape == (expected_count, 2 * config["lmax"])
        assert metadata.shape == (expected_count, 3)

    def test_all_valid_presentations(self):
        """All generated presentations pass validity check."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {
            "n_phases": 4,
            "n_chains": 3,
            "n_moves": 50,
            "lmax": 32,
            "seed": 42,
        }
        presentations, _ = generate_dataset_for_presentation((0, p0, config))

        for i in range(len(presentations)):
            assert is_array_valid_presentation(presentations[i]), (
                f"Presentation {i} is invalid"
            )

    def test_metadata_correctness(self):
        """Metadata origin, phase, and chain indices are correct."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {
            "n_phases": 3,
            "n_chains": 2,
            "n_moves": 10,
            "lmax": 32,
            "seed": 42,
        }
        _, metadata = generate_dataset_for_presentation((7, p0, config))

        # All rows should have origin_index = 7
        assert np.all(metadata[:, 0] == 7)

        # Phase indices should cycle through 0, 0, 1, 1, 2, 2 (for n_chains=2)
        expected_phases = []
        expected_chains = []
        for phase in range(3):
            for chain in range(2):
                expected_phases.append(phase)
                expected_chains.append(chain)
        np.testing.assert_array_equal(metadata[:, 1], expected_phases)
        np.testing.assert_array_equal(metadata[:, 2], expected_chains)

    def test_deterministic(self):
        """Same input produces identical output."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {
            "n_phases": 3,
            "n_chains": 2,
            "n_moves": 50,
            "lmax": 32,
            "seed": 99,
        }
        pres1, meta1 = generate_dataset_for_presentation((0, p0, config))
        pres2, meta2 = generate_dataset_for_presentation((0, p0, config))

        np.testing.assert_array_equal(pres1, pres2)
        np.testing.assert_array_equal(meta1, meta2)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config1 = {"n_phases": 3, "n_chains": 2, "n_moves": 50, "lmax": 32, "seed": 1}
        config2 = {"n_phases": 3, "n_chains": 2, "n_moves": 50, "lmax": 32, "seed": 2}

        pres1, _ = generate_dataset_for_presentation((0, p0, config1))
        pres2, _ = generate_dataset_for_presentation((0, p0, config2))

        assert not np.array_equal(pres1, pres2)

    def test_relator_lengths_within_lmax(self):
        """No relator exceeds lmax in length."""
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {
            "n_phases": 5,
            "n_chains": 3,
            "n_moves": 100,
            "lmax": 32,
            "seed": 42,
        }
        presentations, _ = generate_dataset_for_presentation((0, p0, config))

        for i in range(len(presentations)):
            lengths = get_word_lengths(presentations[i], config["lmax"])
            assert lengths[0] <= config["lmax"], f"r0 length {lengths[0]} > lmax"
            assert lengths[1] <= config["lmax"], f"r1 length {lengths[1]} > lmax"


# --- Tokenizer tests ---


class TestTokenizer:
    def test_basic_tokenization(self):
        """Basic presentation converts to expected token sequence."""
        # [x, y, pad, pad, x^-1, y^-1, pad, pad]
        pres = np.array([1, 2, 0, 0, -1, -2, 0, 0], dtype=np.int8)
        tokens = presentation_to_tokens(pres, max_relator_length=4)
        expected = [TOKEN_X, TOKEN_Y, TOKEN_SEP, TOKEN_X_INV, TOKEN_Y_INV, TOKEN_EOS]
        assert tokens == expected

    def test_single_element_relators(self):
        """Trivial presentation tokenizes correctly."""
        pres = np.array([1, 0, 0, 2, 0, 0], dtype=np.int8)
        tokens = presentation_to_tokens(pres, max_relator_length=3)
        assert tokens == [TOKEN_X, TOKEN_SEP, TOKEN_Y, TOKEN_EOS]

    def test_round_trip(self):
        """tokens_to_presentation(presentation_to_tokens(P)) == P."""
        pres = np.array(
            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int8,
        )
        max_rel = 18
        tokens = presentation_to_tokens(pres, max_relator_length=max_rel)
        recovered = tokens_to_presentation(tokens, max_relator_length=max_rel)
        np.testing.assert_array_equal(pres, recovered)

    def test_round_trip_list_input(self):
        """Works with list input too."""
        pres_list = [1, -2, 0, 0, 2, -1, 0, 0]
        tokens = presentation_to_tokens(pres_list)
        recovered = tokens_to_presentation(tokens, max_relator_length=4)
        np.testing.assert_array_equal(recovered, pres_list)

    def test_token_range(self):
        """All tokens are in valid range [0, VOCAB_SIZE)."""
        pres = np.array([1, 2, -1, -2, -2, 2, 1, -1], dtype=np.int8)
        tokens = presentation_to_tokens(pres, max_relator_length=4)
        assert all(0 <= t < VOCAB_SIZE for t in tokens)

    def test_separator_and_eos_present(self):
        """Every token sequence has exactly one SEP and one EOS."""
        pres = np.array([1, 2, 0, 0, -1, -2, 0, 0], dtype=np.int8)
        tokens = presentation_to_tokens(pres, max_relator_length=4)
        assert tokens.count(TOKEN_SEP) == 1
        assert tokens.count(TOKEN_EOS) == 1
        # SEP comes before EOS
        assert tokens.index(TOKEN_SEP) < tokens.index(TOKEN_EOS)
        # EOS is last
        assert tokens[-1] == TOKEN_EOS

    def test_inferred_max_relator_length(self):
        """When max_relator_length is None, it's inferred from array length."""
        pres = np.array([1, 2, 0, 0, -1, -2, 0, 0], dtype=np.int8)
        tokens_explicit = presentation_to_tokens(pres, max_relator_length=4)
        tokens_inferred = presentation_to_tokens(pres)
        assert tokens_explicit == tokens_inferred

    def test_round_trip_all_1190_format(self):
        """Round-trip works for a presentation in the all_presentations.txt format."""
        # First presentation from all_presentations.txt
        pres = np.array(
            [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int8,
        )
        tokens = presentation_to_tokens(pres)
        recovered = tokens_to_presentation(tokens, max_relator_length=18)
        np.testing.assert_array_equal(pres, recovered)


# --- Incremental saving tests ---


class TestSaveAndLoadShard:
    def test_save_shard_creates_files(self, tmp_path):
        """save_shard writes .npy files in a shards/ subdirectory."""
        lmax = 32
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {"n_phases": 2, "n_chains": 2, "n_moves": 10, "lmax": lmax, "seed": 42}

        results = [generate_dataset_for_presentation((0, p0, config))]
        save_shard(0, results, str(tmp_path), lmax)

        shard_dir = tmp_path / "shards"
        assert (shard_dir / "shard_0000_presentations.npy").exists()
        assert (shard_dir / "shard_0000_metadata.npy").exists()

    def test_shard_round_trip(self, tmp_path):
        """Saved shard data matches original results."""
        lmax = 32
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {"n_phases": 3, "n_chains": 2, "n_moves": 10, "lmax": lmax, "seed": 42}

        pres, meta = generate_dataset_for_presentation((0, p0, config))
        results = [(pres, meta)]
        save_shard(0, results, str(tmp_path), lmax)

        shard_dir = tmp_path / "shards"
        loaded_pres = np.load(shard_dir / "shard_0000_presentations.npy")
        loaded_meta = np.load(shard_dir / "shard_0000_metadata.npy")

        np.testing.assert_array_equal(loaded_pres, pres)
        np.testing.assert_array_equal(loaded_meta, meta)

    def test_shard_multiple_results(self, tmp_path):
        """Shard correctly concatenates results from multiple presentations."""
        lmax = 32
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {"n_phases": 2, "n_chains": 2, "n_moves": 10, "lmax": lmax, "seed": 42}

        result0 = generate_dataset_for_presentation((0, p0, config))
        result1 = generate_dataset_for_presentation((1, p0, config))
        results = [result0, result1]

        save_shard(0, results, str(tmp_path), lmax)

        shard_dir = tmp_path / "shards"
        loaded_pres = np.load(shard_dir / "shard_0000_presentations.npy")

        expected_rows = len(result0[0]) + len(result1[0])
        assert loaded_pres.shape[0] == expected_rows
        assert loaded_pres.shape[1] == 2 * lmax


class TestProgressTracking:
    def test_load_progress_empty(self, tmp_path):
        """load_progress returns defaults when no file exists."""
        progress = load_progress(str(tmp_path))
        assert progress["completed_shards"] == []
        assert progress["total_elapsed"] == 0.0

    def test_save_and_load_progress(self, tmp_path):
        """save_progress writes data that load_progress reads back."""
        config = {"n_phases": 2, "n_chains": 2, "n_moves": 10, "lmax": 32, "seed": 42}
        save_progress(str(tmp_path), [0, 2, 5], config, 123.4)

        progress = load_progress(str(tmp_path))
        assert progress["completed_shards"] == [0, 2, 5]
        assert progress["total_elapsed"] == 123.4
        assert progress["config"] == config

    def test_progress_shards_are_sorted(self, tmp_path):
        """Completed shards are stored in sorted order."""
        config = {"n_phases": 2, "seed": 42}
        save_progress(str(tmp_path), [5, 2, 0, 3], config, 10.0)

        progress = load_progress(str(tmp_path))
        assert progress["completed_shards"] == [0, 2, 3, 5]

    def test_progress_overwrites(self, tmp_path):
        """Subsequent saves overwrite previous progress."""
        config = {"seed": 42}
        save_progress(str(tmp_path), [0], config, 10.0)
        save_progress(str(tmp_path), [0, 1], config, 20.0)

        progress = load_progress(str(tmp_path))
        assert progress["completed_shards"] == [0, 1]
        assert progress["total_elapsed"] == 20.0


class TestMergeShards:
    def test_merge_two_shards(self, tmp_path):
        """Merging two shards produces correct combined output."""
        lmax = 32
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {"n_phases": 2, "n_chains": 2, "n_moves": 10, "lmax": lmax, "seed": 42}

        # Generate and save two shards
        result0 = generate_dataset_for_presentation((0, p0, config))
        result1 = generate_dataset_for_presentation((1, p0, config))

        save_shard(0, [result0], str(tmp_path), lmax)
        save_shard(1, [result1], str(tmp_path), lmax)

        # Merge
        merged_pres, merged_meta = merge_shards(str(tmp_path), 2, lmax)

        # Verify shape
        expected_rows = len(result0[0]) + len(result1[0])
        assert merged_pres.shape == (expected_rows, 2 * lmax)
        assert merged_meta.shape == (expected_rows, 3)

        # Verify content: first half from shard 0, second half from shard 1
        np.testing.assert_array_equal(merged_pres[:len(result0[0])], result0[0])
        np.testing.assert_array_equal(merged_pres[len(result0[0]):], result1[0])

        # Verify merged files saved to disk
        assert (tmp_path / "presentations.npy").exists()
        assert (tmp_path / "metadata.npy").exists()

    def test_merge_preserves_all_data(self, tmp_path):
        """All presentations survive the shard-merge round-trip."""
        lmax = 32
        p0 = [-1, 2, 1, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        config = {"n_phases": 3, "n_chains": 2, "n_moves": 10, "lmax": lmax, "seed": 42}

        all_results = []
        for idx in range(3):
            result = generate_dataset_for_presentation((idx, p0, config))
            all_results.append(result)
            save_shard(idx, [result], str(tmp_path), lmax)

        merged_pres, merged_meta = merge_shards(str(tmp_path), 3, lmax)

        # Verify every presentation is valid
        for i in range(len(merged_pres)):
            assert is_array_valid_presentation(merged_pres[i]), (
                f"Merged presentation {i} is invalid"
            )

        # Verify total count
        expected_total = sum(len(r[0]) for r in all_results)
        assert len(merged_pres) == expected_total
