"""Tests for the transformer dataset generation pipeline and tokenizer."""

import numpy as np
import pytest

from ac_solver.envs.utils import is_array_valid_presentation
from ac_solver.transformer.data_generator import (
    apply_random_ac_moves,
    generate_dataset_for_presentation,
    get_word_lengths,
    resize_presentation_np,
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
