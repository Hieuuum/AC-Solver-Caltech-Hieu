"""Tests for the prepare_dataset module (tokenization, padding, GS labels, split)."""

import numpy as np
import pytest

from ac_solver.transformer.prepare_dataset import (
    load_gs_solved_indices,
    pad_sequences,
    tokenize_presentations,
)
from ac_solver.transformer.tokenizer import TOKEN_EOS, VOCAB_SIZE


class TestLoadGsSolvedIndices:
    def test_returns_set(self):
        """load_gs_solved_indices returns a set of ints."""
        indices = load_gs_solved_indices()
        assert isinstance(indices, set)
        assert all(isinstance(i, int) for i in indices)

    def test_correct_count(self):
        """Should find exactly 533 GS-solved presentations."""
        indices = load_gs_solved_indices()
        assert len(indices) == 533

    def test_indices_in_range(self):
        """All indices should be in [0, 1189]."""
        indices = load_gs_solved_indices()
        assert all(0 <= i < 1190 for i in indices)


class TestTokenizePresentations:
    def test_basic(self):
        """Tokenizes a small batch correctly."""
        # Two simple presentations with max_relator_length=4
        pres = np.array([
            [1, 2, 0, 0, -1, -2, 0, 0],
            [1, 0, 0, 0, 2, 0, 0, 0],
        ], dtype=np.int8)
        tokens = tokenize_presentations(pres, lmax=4)
        assert len(tokens) == 2
        # Each should end with EOS
        assert tokens[0][-1] == TOKEN_EOS
        assert tokens[1][-1] == TOKEN_EOS

    def test_all_tokens_valid(self):
        """All tokens in valid range."""
        pres = np.array([
            [1, 2, -1, 0, -2, 1, -2, 0],
        ], dtype=np.int8)
        tokens = tokenize_presentations(pres, lmax=4)
        assert all(0 <= t < VOCAB_SIZE for t in tokens[0])


class TestPadSequences:
    def test_padding(self):
        """Short sequences are padded to context_length."""
        seqs = [[0, 1, 4, 2, 5], [0, 4, 2, 5]]
        padded = pad_sequences(seqs, context_length=8)
        assert padded.shape == (2, 8)
        # Check that padding uses EOS token
        assert padded[0, 5] == TOKEN_EOS
        assert padded[1, 4] == TOKEN_EOS

    def test_truncation(self):
        """Long sequences are truncated to context_length."""
        seqs = [list(range(20))]
        padded = pad_sequences(seqs, context_length=5)
        assert padded.shape == (1, 5)
        np.testing.assert_array_equal(padded[0], [0, 1, 2, 3, 4])

    def test_exact_length(self):
        """Sequence exactly at context_length is unchanged."""
        seqs = [[0, 1, 2, 3, 4]]
        padded = pad_sequences(seqs, context_length=5)
        np.testing.assert_array_equal(padded[0], [0, 1, 2, 3, 4])

    def test_dtype(self):
        """Output dtype is int16."""
        seqs = [[0, 1, 4, 2, 5]]
        padded = pad_sequences(seqs, context_length=10)
        assert padded.dtype == np.int16
