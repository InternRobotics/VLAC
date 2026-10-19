"""Device selection and attention dispatch for the XPU, CUDA and CPU backends.

Everything except the last test runs without any accelerator, so CUDA-only CI
stays green. The last test compares a tiny random-weight model on CPU and XPU
and is skipped when no XPU is present.
"""

import pathlib
import sys
import unittest
from unittest import mock

import torch

from evo_vlac.utils import device_utils as du

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _hosts(*, xpu: bool, cuda: bool):
    """Make ``is_available`` report exactly the given backends."""

    def fake(device_type: str) -> bool:
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    return mock.patch.object(du, "is_available", side_effect=fake)


class DetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _hosts(xpu=True, cuda=True):
            self.assertEqual(du.default_device_type(), "xpu")
            self.assertEqual(du.default_device_map(), "xpu:0")

    def test_cuda_keeps_previous_default(self):
        with _hosts(xpu=False, cuda=True):
            self.assertEqual(du.default_device_map(), "cuda:0")

    def test_cpu_has_no_index(self):
        with _hosts(xpu=False, cuda=False):
            self.assertEqual(du.default_device_map(), "cpu")

    def test_cuda_wheel_without_xpu_attribute(self):
        with mock.patch.object(du, "torch", mock.Mock(spec=["cuda"])):
            self.assertFalse(du.is_available("xpu"))


class AttnImplTest(unittest.TestCase):
    def test_flash_attn_kept_on_cuda_when_installed(self):
        with mock.patch.object(du.importlib.util, "find_spec", return_value=object()):
            self.assertEqual(du.resolve_attn_impl("cuda:0"), "flash_attn")

    def test_eager_on_cuda_without_flash_attn_package(self):
        with mock.patch.object(du.importlib.util, "find_spec", return_value=None):
            self.assertEqual(du.resolve_attn_impl("cuda:0"), "eager")

    def test_eager_on_xpu_and_cpu_even_if_flash_attn_is_installed(self):
        with mock.patch.object(du.importlib.util, "find_spec", return_value=object()):
            self.assertEqual(du.resolve_attn_impl("xpu:0"), "eager")
            self.assertEqual(du.resolve_attn_impl("cpu"), "eager")

    def test_auto_follows_detection(self):
        with mock.patch.object(du.importlib.util, "find_spec", return_value=object()):
            with _hosts(xpu=True, cuda=True):
                self.assertEqual(du.resolve_attn_impl("auto"), "eager")
            with _hosts(xpu=False, cuda=True):
                self.assertEqual(du.resolve_attn_impl("auto"), "flash_attn")


class InitModelDispatchTest(unittest.TestCase):
    """``init_model`` hands the explicit device_map and the resolved attention to swift."""

    def _init(self, device_map, find_spec):
        from evo_vlac.utils import model_utils

        critic = model_utils.GAC_model.__new__(model_utils.GAC_model)
        loader = mock.Mock(return_value=(mock.Mock(), mock.Mock()))
        swift_llm = mock.Mock()
        modules = {"swift.llm": swift_llm, "swift.plugin": mock.Mock()}
        with mock.patch.object(model_utils, "get_model_tokenizer", loader), \
                mock.patch.object(model_utils, "get_template"), \
                mock.patch.object(model_utils, "seed_everything"), \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(du.importlib.util, "find_spec", return_value=find_spec):
            critic.init_model("path", device_map=device_map)
        return loader.call_args.kwargs

    def test_explicit_device_map_is_forwarded(self):
        kwargs = self._init("xpu:1", object())
        self.assertEqual(kwargs["device_map"], "xpu:1")
        self.assertEqual(kwargs["attn_impl"], "eager")

    def test_cuda_still_requests_flash_attn(self):
        kwargs = self._init("cuda:0", object())
        self.assertEqual(kwargs["attn_impl"], "flash_attn")


class NoHardCodedCudaTest(unittest.TestCase):
    def test_examples_and_readme_use_the_device_module(self):
        files = list((ROOT / "evo_vlac" / "examples").glob("*.py")) + [ROOT / "README.md"]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("device_map=f'cuda:0'", text, path.name)
            if "init_model(" in text:
                self.assertIn("default_device_map()", text, path.name)


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU")
class XpuParityTest(unittest.TestCase):
    def test_tiny_random_internlm_style_decoder_matches_cpu(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, attn_implementation="eager",
        )
        model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
        ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            expected = model(ids).logits.float()
            actual = model.to(du.default_device_map())(ids.to(du.default_device_map())).logits.float().cpu()
        self.assertEqual(expected.argmax(-1).tolist(), actual.argmax(-1).tolist())
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
