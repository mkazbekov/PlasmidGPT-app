import unittest

import numpy as np
import torch

from plasmidgpt_app.core import (
    SimpleNN,
    fasta_bytes,
    parse_sequence_input,
    predict_attributes,
    sequence_stats,
    validate_sequence,
)


class SequenceInputTests(unittest.TestCase):
    def test_normalizes_plain_dna(self):
        self.assertEqual(validate_sequence("at cg\nN"), "ATCGN")

    def test_rejects_invalid_bases(self):
        with self.assertRaisesRegex(ValueError, "only A, T, C, G, and N"):
            validate_sequence("ATCGX")

    def test_parses_multiple_fasta_records(self):
        parsed = parse_sequence_input(">one\natcg\n>two\nGGNN\n")
        self.assertEqual(parsed, [("one", "ATCG"), ("two", "GGNN")])

    def test_fasta_export_wraps_and_round_trips(self):
        exported = fasta_bytes([("my sequence", "ATCG" * 30)], line_width=40)
        parsed = parse_sequence_input(exported)
        self.assertEqual(parsed, [("my_sequence", "ATCG" * 30)])

    def test_sequence_statistics_ignore_ambiguous_bases_for_gc(self):
        stats = sequence_stats("GGCANN")
        self.assertEqual(stats["length"], 6)
        self.assertEqual(stats["ambiguous_bases"], 2)
        self.assertAlmostEqual(stats["gc_percent"], 75.0)


class PredictionTests(unittest.TestCase):
    def test_prediction_output_is_ranked_and_bounded(self):
        model = SimpleNN(input_dim=768, num_classes=3)
        labels = np.asarray(["a", "b", "c"])
        embeddings = np.zeros((2, 768), dtype=np.float32)
        results = predict_attributes(
            embeddings, model, labels, top_n=99, device=torch.device("cpu")
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 3)
        self.assertGreaterEqual(results[0][0][1], results[0][1][1])


if __name__ == "__main__":
    unittest.main()
