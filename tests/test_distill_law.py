import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from net import QNet
from sepsis_distill_law import margins, margin_weights


class TestSampleWeight(unittest.TestCase):
    def test_equal_weights_match_unweighted(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5)); y = rng.normal(size=(40, 3))
        n1 = QNet(5, 3, width=8, seed=1)
        n2 = QNet(5, 3, width=8, seed=1)
        l1 = n1.fit(x, y, epochs=3, seed=1)
        l2 = n2.fit(x, y, epochs=3, seed=1, sample_weight=np.ones(40))
        self.assertTrue(np.allclose(n1.predict(x), n2.predict(x)))
        self.assertTrue(np.allclose(l1, l2))

    def test_weights_change_fit(self):
        rng = np.random.default_rng(0)
        x = np.eye(4); y = np.array([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
        n1 = QNet(4, 2, width=8, seed=2)
        n2 = QNet(4, 2, width=8, seed=2)
        n1.fit(x, y, epochs=5, seed=2)
        n2.fit(x, y, epochs=5, seed=2, sample_weight=np.array([1., 1., 1., 20.]))
        self.assertFalse(np.allclose(n1.predict(x), n2.predict(x)))

    def test_weight_validation(self):
        n = QNet(4, 2, width=8, seed=0)
        with self.assertRaises(AssertionError):
            n.fit(np.eye(4), np.zeros((4, 2)), sample_weight=np.ones(3))
        with self.assertRaises(AssertionError):
            n.fit(np.eye(4), np.zeros((4, 2)), sample_weight=np.array([1., 1., 0., 1.]))


class TestMargins(unittest.TestCase):
    def test_margins_basic(self):
        q = np.array([[1., 2., 0.5], [0., 0., 0.]])
        m = margins(q)
        self.assertAlmostEqual(m[0], 1.0)
        self.assertAlmostEqual(m[1], 0.0)

    def test_weight_rule_bounds(self):
        m = np.array([0.0, 0.02, 1.0, 10.0])
        w = margin_weights(m)
        self.assertEqual(w[0], 50.0)   # clipped high
        self.assertEqual(w[1], 50.0)   # 1/0.02 = 50
        self.assertEqual(w[2], 1.0)    # clipped low
        self.assertEqual(w[3], 1.0)
        self.assertTrue(np.all(w >= 1) and np.all(w <= 50))


if __name__ == '__main__':
    unittest.main()
