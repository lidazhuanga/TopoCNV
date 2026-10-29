"""CPU 兼容门禁：以 git HEAD 的原入口为参考，不读取真实数据、不启动训练。"""
import argparse
import ast
import copy
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTRY_NAMES = (
    "npc_vesselFM3D_cascade_MIA_v3_vit_memory", "npc_sota_train",
    "cnv_protocol_s2", "npc_baseline",
)
REFERENCE_REVISION = "85a6f2d"
ENTRIES = {name: importlib.import_module("trainer." + name) for name in ENTRY_NAMES}
MAIN = ENTRIES[ENTRY_NAMES[0]]
PROTOCOL = ENTRIES["cnv_protocol_s2"]
BASELINE = ENTRIES["npc_baseline"]
SOTA = ENTRIES["npc_sota_train"]
torch.set_num_threads(1)


def reference_source(name):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", REFERENCE_REVISION + ":trainer/" + name + ".py"],
        text=True,
    )


def reference_symbols(name, names):
    tree = ast.parse(reference_source(name))
    namespace = vars(ENTRIES[name]).copy()
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "reference", "exec"), namespace)
    return namespace


def parser_for(name, old=False):
    source = reference_source(name) if old else Path(ENTRIES[name].__file__).read_text()
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "args" for target in node.targets
        ):
            break
        body.append(node)
    namespace = vars(ENTRIES[name]).copy()
    if old:
        # 从旧源码恢复字面量默认值，不用新入口常量冒充参考值。
        for node in tree.body:
            value = getattr(node, "value", None)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = literal
    exec(compile(ast.Module(body=body, type_ignores=[]), "parser", "exec"), namespace)
    return namespace["parser"]


class TinyCascade(nn.Module):
    def __init__(self, channels=1, classes=6, binary=False):
        super().__init__()
        self.net = nn.Conv3d(channels, classes, 1)
        self.in_channels = channels
        self.binary = binary
        self.coords = []

    def forward(self, image, labels=None, coord_info=None):
        if coord_info is not None:
            self.coords.append(tuple(tensor.detach().clone() for tensor in coord_info))
        logits = self.net(image)
        if self.binary:
            return logits.sigmoid(), logits, torch.zeros_like(logits), None
        return logits


class TinyTree(MAIN.HierarchicalStage2Model):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Conv3d(3, 14, 1)
        self.coords = []

    def forward(self, image, labels=None, coord_info=None):
        self.coords.append(tuple(tensor.detach().clone() for tensor in coord_info))
        logits = self.net(image)
        return logits[:, :6].softmax(1), logits[:, 6:9], logits[:, 9:13], logits[:, 13:14], None


class TinyFeatures(nn.Module):
    provides_fullres_features = True

    def __init__(self):
        super().__init__()
        self.memory_bank = nn.Identity()
        self.evidence_gate = None
        self.conv = nn.Conv3d(3, 2, 1)
        self.coords = None

    def forward_features(self, image, labels=None, coord_info=None):
        self.coords = coord_info
        features = self.conv(image)
        self._mem_tok0 = (features.flatten(2).transpose(1, 2), features.shape[2:])
        self._patch_padded = features.shape[2:]
        return features, None, features.shape[2:]


class BackwardCompatibility(unittest.TestCase):
    def test_cli_actions_defaults_and_help(self):
        for name in (ENTRY_NAMES[0], "npc_sota_train", "npc_baseline"):
            old, new = parser_for(name, True), parser_for(name)
            def manifest(parser):
                return [(type(action).__name__, action.option_strings, action.dest,
                         action.default, action.required, action.choices, action.nargs,
                         action.const, action.type, action.help) for action in parser._actions]
            self.assertEqual(manifest(old), manifest(new), name)
            self.assertEqual(old.format_help(), new.format_help(), name)
            print("CLI defaults/actions:", name, len(new._actions) - 1, "OK")
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "main"
                             for node in ast.parse(reference_source("cnv_protocol_s2")).body))
        for name in ENTRY_NAMES:
            result = subprocess.run([sys.executable, str(ROOT / "trainer" / (name + ".py")), "--help"],
                                    cwd="/", env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                    text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            print("direct --help:", name, "OK")

    def test_public_signatures_and_retained_numeric_contracts(self):
        for name, module in ENTRIES.items():
            old = ast.parse(reference_source(name))
            for node in old.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    self.assertTrue(hasattr(module, node.name), (name, node.name))
            current = ast.parse(Path(module.__file__).read_text())
            current_defs = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
            for node in old.body:
                if isinstance(node, ast.FunctionDef) and node.name in current_defs:
                    self.assertEqual(ast.dump(node.args), ast.dump(current_defs[node.name].args), node.name)
            # 权重格式/输出路径的 torch.save、torch.load 调用必须保持。
            def checkpoint_calls(tree):
                return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch" and node.func.attr in ("load", "save")]
            self.assertEqual(checkpoint_calls(old), checkpoint_calls(current), name)
        old = ast.parse(reference_source(ENTRY_NAMES[0]))
        new = ast.parse(Path(MAIN.__file__).read_text())
        retained = {"Stage1Dataset", "Stage2Dataset", "train_one_epoch", "train_stage2_one_epoch",
                    "train_binary_y_one_epoch", "SoftTreeLoss", "_load_cached_fg_indices",
                    "validate", "validate_stage2", "validate_binary_y_patch",
                    "validate_binary_y_sliding", "_load_label_as_int3d", "_save_label_nrrd_with_space"}
        for node in old.body:
            if getattr(node, "name", None) in retained:
                peer = next(item for item in new.body if getattr(item, "name", None) == node.name)
                self.assertEqual(ast.dump(node), ast.dump(peer), node.name)
        self.assertTrue(MAIN._HAS_PROTOCOL_S2)

    def test_loss_value_and_gradient_reference(self):
        old = reference_symbols("cnv_protocol_s2", {"BinaryYLoss", "EvidenceGatedYLoss"})
        torch.manual_seed(91)
        target = torch.randint(0, 2, (1, 3, 4, 5)).float()
        ignore = torch.rand_like(target) > 0.8
        gate = torch.rand(1, 1, 3, 4, 5)
        for alignment in ("low", "global", "off"):
            for all_ignored in (False, True):
                mask = torch.ones_like(ignore) if all_ignored else ignore
                logits = torch.randn(1, 1, 3, 4, 5) * 25
                outputs, gradients = [], []
                for loss_class in (old["EvidenceGatedYLoss"], PROTOCOL.EvidenceGatedYLoss):
                    local = logits.clone().requires_grad_()
                    memory = (-logits).clone().requires_grad_()
                    output = loss_class(mem_align=alignment)(local, target, mask, gate, memory)
                    gradients.append(torch.autograd.grad(output["total"], (local, memory)))
                    outputs.append(output)
                self.assertEqual(outputs[0].keys(), outputs[1].keys())
                for key in outputs[0]:
                    torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
                for left, right in zip(*gradients):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_protocol_helpers_and_gate_state_dict(self):
        old = reference_symbols("cnv_protocol_s2", {"evidence_map_g1", "BinaryYStage2Model",
                                "dilate_mm", "stage1_softmax_to_cascade_prior", "scramble_cascade_prior"})
        soft = np.random.RandomState(3).rand(6, 3, 4, 5).astype(np.float32)
        for channels in (6, 3, 2):
            np.testing.assert_array_equal(old["stage1_softmax_to_cascade_prior"](soft[:channels]),
                                          PROTOCOL.stage1_softmax_to_cascade_prior(soft[:channels]))
        for mode in ("0", "drop", "spatial_roll", "channel_permute"):
            np.testing.assert_array_equal(old["scramble_cascade_prior"](soft, mode),
                                          PROTOCOL.scramble_cascade_prior(soft, mode))
        image = torch.rand(1, 3, 3, 4, 5)
        coords = (torch.zeros(1, 3), torch.tensor([[3., 4., 5.]]), torch.zeros(1, 3))
        for use_gate, alpha in ((True, None), (True, 0.5), (False, None)):
            models = []
            for cls in (old["BinaryYStage2Model"], PROTOCOL.BinaryYStage2Model):
                torch.manual_seed(81)
                models.append(cls(TinyFeatures(), nn.Identity(), 2, use_gate, 3, alpha, mem_dim=2))
            self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())
            for key in models[0].state_dict():
                torch.testing.assert_close(models[0].state_dict()[key], models[1].state_dict()[key], rtol=0, atol=0)
            for left, right in zip(models[0](image, None, coords), models[1](image, None, coords)):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertIs(models[1].net.coords, coords)

    def test_sliding_window_coordinates_padding_no_tta_and_fold_mean(self):
        old = reference_symbols(ENTRY_NAMES[0], {"sliding_window_inference"})
        from topocnv.engine.evaluator import legacy_fold_mean
        for shape in ((2, 3, 4), (6, 7, 8)):
            image = np.random.RandomState(8).rand(3, *shape).astype(np.float32)
            for binary in (True, False):
                first = TinyCascade(3, 1 if binary else 6, binary)
                second = copy.deepcopy(first)
                kwargs = dict(patch_size=(3, 4, 5), num_classes=1 if binary else 6,
                              device="cpu", step_size=0.5, coord_aware=True)
                for scipy_enabled in (False, True):
                    old["HAS_SCIPY"] = scipy_enabled
                    first.coords.clear()
                    second.coords.clear()
                    with patch.object(MAIN, "HAS_SCIPY", scipy_enabled):
                        ref = old["sliding_window_inference"](image, first, **kwargs)
                        actual = MAIN.sliding_window_inference(image, second, **kwargs)
                    np.testing.assert_array_equal(ref, actual)
                    self.assertEqual(len(first.coords), len(second.coords))
                    for a, b in zip(first.coords, second.coords):
                        for left, right in zip(a, b):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        self.assertEqual(int(torch.count_nonzero(b[2])), 0)
                np.testing.assert_array_equal(legacy_fold_mean([ref, actual]), np.mean([ref, actual], axis=0))

    def test_scheduler_lr_scale_and_resume(self):
        old = reference_symbols(ENTRY_NAMES[0], {"PolyLRWithWarmup"})
        optimizers = [torch.optim.SGD([{"params": [nn.Parameter(torch.ones(1))], "lr_scale": 0.25},
                                       {"params": [nn.Parameter(torch.ones(1))]}], lr=1e-3) for _ in range(2)]
        schedulers = [old["PolyLRWithWarmup"](optimizers[0], 20, 3), MAIN.PolyLRWithWarmup(optimizers[1], 20, 3)]
        for _ in range(20):
            for scheduler in schedulers:
                scheduler.step()
            self.assertEqual(schedulers[0].get_last_lr(), schedulers[1].get_last_lr())
            self.assertEqual([group["lr"] for group in optimizers[0].param_groups],
                             [group["lr"] for group in optimizers[1].param_groups])
        for scheduler in schedulers:
            scheduler._step = 8
        self.assertEqual(schedulers[0].get_last_lr(), schedulers[1].get_last_lr())

    def test_sota_factory_constructor_configuration(self):
        old = reference_symbols("npc_sota_train", {"build_model"})
        package = importlib.import_module("topocnv.models.sota")
        for name, constructor in zip(SOTA.SUPPORTED_MODELS,
                                     ("DynUNet", "SegResNet", "AttentionUnet", "UNETR", "SwinUNETR", "NNFormer3D")):
            calls = []
            def fake(*args, **kwargs):
                calls.append((args, kwargs))
                return nn.Identity()
            old[constructor] = fake
            old["build_model"](name)
            with patch.object(package, constructor, fake):
                SOTA.build_model(name)
            self.assertEqual(calls[0], calls[1], name)
        old_nn = reference_symbols("npc_sota_train", {"NNFormer3D", "_ConvTransformerBlock",
                                  "_nnFormerEncoderBlock", "_nnFormerDecoderBlock"})
        torch.manual_seed(12)
        first = old_nn["NNFormer3D"](1, 2, 4).eval()
        torch.manual_seed(12)
        second = SOTA.NNFormer3D(1, 2, 4).eval()
        self.assertEqual(first.state_dict().keys(), second.state_dict().keys())
        image = torch.rand(1, 1, 16, 16, 16)
        with torch.no_grad():
            torch.testing.assert_close(first(image), second(image), rtol=0, atol=0)

    def test_cascade_factory_legacy_defaults_and_fallback(self):
        old = reference_symbols(ENTRY_NAMES[0], {"_build_vit", "_build_memory_bank"})
        builder = importlib.import_module("topocnv.models.builder")
        for name, constructor in (("vit", "ViTUNet3D"), ("nnunet_mla", "DynUNetMLA3D"),
                                  ("unknown-legacy-fallback", "ViTUNet3D")):
            calls = []
            def fake(**kwargs):
                calls.append(kwargs)
                return nn.Identity()
            old[constructor] = fake
            old["_build_vit"](1, 6, backbone=name)
            with patch.object(builder, constructor, fake):
                MAIN._build_vit(1, 6, backbone=name)
            self.assertEqual(calls[0], calls[1])
        for memory_type, constructor in (("mla", "MLAMemoryBank"), ("anatomical_v4", "AnatomySlotMemoryBankV4"),
                                          ("context_anchor", "ContextAnchorMemoryBank"), ("unknown", "MLAMemoryBank")):
            calls = []
            def fake(**kwargs):
                calls.append(kwargs)
                return nn.Identity()
            old[constructor] = fake
            old["_build_memory_bank"](6, 32, 0.99, 0.5, 8, memory_type=memory_type)
            with patch.object(builder, constructor, fake):
                MAIN._build_memory_bank(6, 32, 0.99, 0.5, 8, memory_type=memory_type)
            self.assertEqual(calls[0], calls[1])

    def test_dataset_batch_metadata_and_mmap(self):
        old = reference_symbols(ENTRY_NAMES[0], {"Stage1Dataset", "Stage2Dataset", "_load_cached_case"})
        image = np.random.RandomState(7).rand(6, 7, 8).astype(np.float32)
        label = np.arange(image.size).reshape(image.shape).astype(np.uint8) % 6
        prior = np.random.RandomState(9).rand(6, *image.shape).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            np.save(path / "s_image.npy", image)
            np.save(path / "s_label.npy", label)
            # 两个实现的 mmap 行为都要校验，且仅使用临时合成数组。
            for mmap in (False, True):
                ref = old["_load_cached_case"]("s", path, mmap)
                actual = MAIN._load_cached_case("s", path, mmap)
                for left, right in zip(ref, actual):
                    self.assertEqual(type(left), type(right))
                    np.testing.assert_array_equal(left, right)
            case = [("s", "unused.nii.gz", "unused.nrrd")]
            np.save(path / "s_stage1_softmax.npy", prior)
            for training in (False, True):
                for cls_name in ("Stage1Dataset", "Stage2Dataset"):
                    kwargs = dict(cases=case, patch_size=(3, 4, 5), cache_dir=path, training=training)
                    if cls_name == "Stage2Dataset":
                        kwargs["stage1_pred_dir"] = path
                    first = old[cls_name](**kwargs)
                    second = getattr(MAIN, cls_name)(**kwargs)
                    np.random.seed(14)
                    ref = first[0]
                    np.random.seed(14)
                    actual = second[0]
                    self.assertEqual(len(ref), len(actual))
                    for left, right in zip(ref, actual):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    if cls_name == "Stage2Dataset":
                        self.assertEqual(actual[-1].shape, (9,))

    def test_minimal_synthetic_batch_epoch_dry_run(self):
        torch.manual_seed(123)
        image = torch.rand(1, 1, 2, 3, 4)
        prior = torch.rand(1, 2, 2, 3, 4)
        labels = torch.randint(0, 6, (1, 2, 3, 4))
        meta = torch.tensor([[2., 3., 4., 10., 20., 30., 1., 0., 1.]])
        stages = [
            (MAIN.train_one_epoch, TinyCascade(), [(image, labels)] * 3,
             MAIN.WeightedBCEDiceLoss(), {"grad_accum_steps": 2}),
            (MAIN.train_stage2_one_epoch, TinyCascade(3), [(image, prior, labels, meta)] * 3,
             MAIN.WeightedBCEDiceLoss(), {"grad_accum_steps": 2}),
            (MAIN.train_stage2_one_epoch, TinyTree(), [(image, prior, labels, meta)] * 3,
             MAIN.SoftTreeLoss([1] * 6, [1] * 3, [1] * 4), {"grad_accum_steps": 2}),
            (MAIN.train_binary_y_one_epoch, TinyCascade(3, 1, True),
             [(image, prior, torch.where(labels == 5, -1, (labels > 0).long()), meta)] * 3,
             PROTOCOL.EvidenceGatedYLoss(), {"grad_accum_steps": 2}),
            (BASELINE.train_one_epoch, nn.Conv3d(1, 6, 1), [(image, labels)],
             BASELINE.WeightedBCEDiceLoss(), {}),
            (SOTA.train_one_epoch, nn.Conv3d(1, 2, 1), [(image, ((labels >= 1) & (labels <= 4)).long())],
             SOTA.WeightedBCEDiceLoss(class_weights=torch.tensor([0.1, 1.])), {}),
        ]
        for function, model, loader, loss, kwargs in stages:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                result = function(model, loader, optimizer, loss, "cpu", **kwargs)
                self.assertTrue(np.isfinite(result).all(), function.__name__)
                expected_steps = 1 if function is MAIN.train_binary_y_one_epoch or len(loader) == 1 else 2
                self.assertEqual(step.call_count, expected_steps)
            if getattr(model, "coords", None):
                for left, right in zip(model.coords[0], (meta[:, :3], meta[:, 3:6], meta[:, 6:9])):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            print("synthetic epoch:", function.__module__, function.__name__, "OK")

    def test_no_data_main_dispatch_and_fold_validation(self):
        for module, argv in ((MAIN, ["--stage", "inference", "--input_dir", "/nonexistent"]),
                             (BASELINE, ["--fold", "5"]),
                             (SOTA, ["--model", "nnunet", "--fold", "5"])):
            with patch.object(sys, "argv", [module.__file__] + argv):
                if module is MAIN:
                    with patch.object(module, "run_inference_external") as inference, \
                         patch.object(module, "_get_all_cases", side_effect=AssertionError("真实数据访问")):
                        module.main()
                        self.assertEqual(inference.call_count, 1)
                else:
                    with self.assertRaisesRegex(ValueError, "fold must"):
                        module.main()
        for stage in ("0", "1", "2", "generate_cv"):
            with patch.object(sys, "argv", [MAIN.__file__, "--stage", stage]), \
                 patch.object(MAIN, "_get_all_cases", side_effect=RuntimeError("smoke data boundary")):
                with self.assertRaisesRegex(RuntimeError, "smoke data boundary"):
                    MAIN.main()
        parser = parser_for(ENTRY_NAMES[0])
        self.assertEqual(parser.get_default("stage"), None)
        self.assertEqual(next(a for a in parser._actions if a.dest == "stage").choices,
                         ["0", "1", "2", "generate_cv", "inference"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
