import importlib.util
import pathlib
import sys
import types
import unittest
import numpy as np
import torch

stub = types.ModuleType('plot_causal_heatmap')
stub.plot_heatmap = lambda *args, **kwargs: None
sys.modules['plot_causal_heatmap'] = stub

MODULE_PATH = pathlib.Path('/mnt/e/AA\u53d1\u8868\u8bba\u6587\u7684\u6570\u636e/3.CIGAN/causality_gpu/causilty.py')
spec = importlib.util.spec_from_file_location('cigan_causilty', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _synthetic_eeg(n=12, c=4, t=120, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, c, t)).astype(np.float32) * scale
    return torch.from_numpy(x)


class CausiltyRegressionTest(unittest.TestCase):
    def test_reduced_model_rss_not_smaller_than_full_model_on_noise(self):
        eeg = _synthetic_eeg()
        p = 2
        _, rss_full, _, xxt, xyt = mod.fit_var_ols_batched(eeg, p, batch_size=6)
        c = eeg.shape[1]
        ysq = float((eeg[:, 0, p:] ** 2).sum().item())
        for src in range(1, c):
            exclude = [lag * c + src for lag in range(p)]
            keep = [i for i in range(c * p) if i not in exclude]
            keep_t = torch.tensor(keep, device=mod.device)
            xxr = xxt[keep_t][:, keep_t]
            xyr = xyt[0:1, keep_t]
            stab = 1e-6 * torch.eye(xxr.shape[0], device=mod.device, dtype=xxr.dtype)
            a_red = xyr @ torch.linalg.inv(xxr + stab)
            rss_red = ysq - torch.sum(a_red * xyr).item()
            self.assertGreaterEqual(rss_red + 1e-5, rss_full[0].item())

    def test_no_rss_clamping_under_large_scale_random_data(self):
        eeg = _synthetic_eeg(n=24, c=8, t=240, seed=1, scale=1e4)
        _, rss_full, _, _, _ = mod.fit_var_ols_batched(eeg, p=3, batch_size=8)
        self.assertEqual(int((rss_full <= 1e-8).sum().item()), 0)


if __name__ == '__main__':
    unittest.main()
