"""V5 冻结协议和只读预检；不依赖 Torch，不创建数据或输出目录。"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

PathLike = Union[str, Path]
VERSION = "V5-Utility-v1"
CHECKPOINT_SCHEMA = "V5-Utility-epoch-v1"
ROOT = Path("/cfs/zzkj/home/dkq/cnv/nnunet_v5_utility")
OLD_ROOT = Path("/cfs/zzkj/home/dkq/cnv/nnunet_cnv")
BASE_PLANS = OLD_ROOT / "nnUNet_preprocessed/Dataset540_CNVv42binY/nnUNetPlans.json"
BASE_PLANS_SHA256 = "e5cc8065ca5c50c67320ec5a14bad67289f317ca9ad2dbb429b021afe1e0fde6"
DATASETS = {"C": "Dataset901_CNVv5C", "MM": "Dataset902_CNVv5MM", "F": "Dataset903_CNVv5F"}
ROOT_NAMES = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
LEGACY_ALGORITHM = {
    "source": "trainer/v4/nnunet/nnUNetTrainerV43.py",
    "sha256": "6f4ee65b902cbca747aef60da4e89b7e6d6d677e96eb374daf6f33a398abccf3",
    "symbols": ["draw_drop", "keep_mask"],
    "runtime_import": False,
}
RECIPE = {
    "training_seed": 42, "split_seed": 20261022,
    "configuration": "3d_fullres", "epochs": 300, "train_iterations": 250,
    "validation_iterations": 50, "iteration_budget": 75000,
    "batch_size": 2, "patch_size": [56, 192, 192],
    "spacing": [2.5, 1.1328120231628418, 1.1328120231628418],
    "optimizer": "SGD", "initial_lr": 0.01, "momentum": 0.99,
    "weight_decay": 3e-5, "nesterov": True, "lr_schedule": "PolyLR_by_epoch",
    "lr_exponent": 0.9, "loss": "DC+CE", "deep_supervision": True,
    "gradient_clip": 12, "foreground_oversample": 0.33,
    "train_workers": 4, "validation_workers": 1, "queue_depth": 2,
    "start_method": "spawn", "cpu_threads": 1, "warmup_batches_per_queue": 1,
    "deterministic_algorithms": True, "cudnn_benchmark": False,
    "blur_benchmark": False, "compile": False, "checkpoint_selection": "final",
    "resume_boundary": "completed_epoch_only", "extend_for_amp_overflow": False,
    "cross_hardware_bitwise_guarantee": False,
    "dropout": {"candidate": "MMd1", "wholecase": 0.2, "truncation": 0.3,
                "slab": 0.2, "tilt_degrees": [0, 20], "cranial_probability": 0.5,
                "slab_count": [1, 2], "slab_thickness": [4, 10]},
}
STREAM_SCHEME = {
    "version": 1, "algorithm": "numpy.SeedSequence",
    "entropy": ["training_seed", "fold", "epoch", "worker", "purpose", "sample", "transform"],
    "purposes": {"main": 1, "train_sampling": 2, "val_sampling": 3,
                 "train_spatial": 4, "train_intensity": 5, "val_transform": 6,
                 "F_dropout": 7},
    "arm_in_common_entropy": False, "cpu_only_workers": True,
    "common_image_layout": "CT,ownMRI_or_zero,valid_or_zero",
    "intensity_layout": "first_two_channels_even_for_C",
    "epoch_queue_policy": "finish_discard_recreate_then_one_warmup_each",
}
VERSIONS = {"nnunetv2": "2.5.2", "torch": "2.6.0+cu124", "numpy": "1.26.4",
            "batchgenerators": "0.25", "batchgeneratorsv2": "0.3.6"}
PINNED_SOURCES = {
    "nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py": "b7cd24e0dcb02f5e02ecaadba386479ee0771b61cfbbf9be5c132b8b7aa0db0f",
    "nnunetv2/training/dataloading/data_loader_3d.py": "a6d57f86896ef4353a92432f00ce3c4db52005d1e4f518df8a9abcfe6e1217b9",
    "nnunetv2/training/dataloading/base_data_loader.py": "d57be0edae30715d9fa32afe5b450f6c3b9683dccfcb82af33c599f375243718",
    "nnunetv2/training/dataloading/nnunet_dataset.py": "1f7d8857d094e7ed809ccd26491d42b496014df9aa07efb07b9ce212a1006ba7",
    "nnunetv2/training/lr_scheduler/polylr.py": "b58eb2d83c0f3c6c72503d7d65870e888937321e3b2fc86266e92236d232e923",
    "batchgenerators/dataloading/multi_threaded_augmenter.py": "9ad15e4c828b9848f9489244c79748e1f562c0685ec417af60adf24e3c8e6266",
    "batchgenerators/dataloading/data_loader.py": "fea94f29f25d68a936363934592f7d06bc10c081bc18d58c38309391fa8b5a23",
    "batchgeneratorsv2/transforms/spatial/spatial.py": "134d75a69fb464ef599d76563b778206f6ec8fa1508be068d2da7c62ab8957da",
    "batchgeneratorsv2/transforms/noise/gaussian_blur.py": "9531a89bd80ef8ddc5aedab31cb3bedc52f4b287c23a018055c3dc8b12a0321d",
}
LOCAL_MODULES = ("__init__.py", "protocol.py", "randomness.py", "augmentation.py", "trainer.py", "launch.py")


class ProtocolError(ValueError):
    """错误文本仅描述契约，不包含患者标识。"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: PathLike) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: PathLike) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def guard_root(root: PathLike, test_root: bool = False) -> Path:
    """仅生产根或显式命名的独立合成临时根；拒绝符号链接逃逸。"""
    raw = Path(root).expanduser().absolute()
    resolved = raw.resolve()
    if raw != resolved or resolved == OLD_ROOT or OLD_ROOT in resolved.parents:
        raise ProtocolError("根目录包含符号链接或属于旧工程")
    if resolved == ROOT:
        if test_root:
            raise ProtocolError("生产根不能标记为合成测试根")
    elif not (test_root and resolved.name.startswith("v5-utility-test-")
              and len(resolved.name) > len("v5-utility-test-")
              and (resolved.parent in (Path("/tmp"), Path("/var/tmp"), Path(__file__).resolve().parent)
                   or Path(__file__).resolve().parent in resolved.parents)):
        raise ProtocolError("必须使用专属生产根或显式 v5-utility-test-* 临时根")
    for name in ROOT_NAMES:
        guard_child(resolved, resolved / name)
    return resolved


def guard_child(root: PathLike, path: PathLike) -> Path:
    base = Path(root).absolute()
    candidate = Path(path).absolute()
    if candidate != candidate.resolve() or base not in candidate.parents:
        raise ProtocolError("路径不在允许根内或包含符号链接")
    return candidate


def assert_dataset_slots_free(root: PathLike = ROOT, test_root: bool = False) -> None:
    """数据生产任务使用；只检查，不覆盖、不创建。"""
    base = guard_root(root, test_root)
    occupied = [f"{kind}/{name}" for kind in ROOT_NAMES for name in DATASETS.values()
                if (base / kind / name).exists()]
    if occupied:
        raise ProtocolError("V5 数据集名称已占用，禁止覆盖: " + ", ".join(occupied))


def _identifiers(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ProtocolError("病例列表必须非空")
    if any(not isinstance(v, str) or re.fullmatch(r"[A-Za-z0-9_-]+", v) is None for v in values):
        raise ProtocolError("病例标识必须是安全文件名")
    if len(set(values)) != len(values):
        raise ProtocolError("病例列表存在重复")
    return list(values)


def build_cohort_split(cohort: Sequence[str], report: Sequence[Mapping[str, Sequence[str]]],
                       dev: Sequence[Mapping[str, Sequence[str]]]) -> List[Dict[str, List[str]]]:
    """仅封装原划分：report K / dev 5+K；不生成、不重排、不随机划分。"""
    splits = [{"train": list(s["train"]), "val": list(s["val"])} for s in [*report, *dev]]
    validate_cohort_split(cohort, splits)
    return splits


def validate_cohort_split(cohort: Sequence[str], splits: Any, fold: int = 0,
                          mode: str = "report") -> Tuple[List[str], List[str]]:
    universe = set(_identifiers(cohort))
    if type(fold) is not int or fold not in range(5) or mode not in ("report", "dev"):
        raise ProtocolError("fold 必须为 0..4，mode 必须为 report/dev")
    if not isinstance(splits, list) or len(splits) != 10:
        raise ProtocolError("必须提供原有十项 split：report K 和 dev 5+K")
    normalized = []
    for entry in splits:
        if not isinstance(entry, dict) or set(entry) != {"train", "val"}:
            raise ProtocolError("split 必须仅含 train/val")
        tr, va = _identifiers(entry["train"]), _identifiers(entry["val"])
        if set(tr) & set(va):
            raise ProtocolError("split 训练验证集合有交集")
        if not (set(tr) | set(va)) <= universe:
            raise ProtocolError("split 引用了不存在的病例")
        normalized.append((tr, va))
    held_out = []
    for k in range(5):
        tr, va = normalized[k]
        dt, dv = normalized[5 + k]
        if set(tr) | set(va) != universe:
            raise ProtocolError("report split 覆盖缺项")
        if set(dt) | set(dv) != set(tr):
            raise ProtocolError("dev split 必须恰好覆盖对应 report 训练集")
        held_out.extend(va)
    if set(held_out) != universe or len(held_out) != len(universe):
        raise ProtocolError("五个 report 验证集必须无重复地覆盖 cohort")
    tr, va = normalized[fold + (5 if mode == "dev" else 0)]
    return list(tr), list(va)


def upstream_fingerprint() -> Dict[str, Any]:
    """仅通过 distribution 元数据读取源码；不会 import Torch/旧 trainer。"""
    if platform.python_version() != "3.9.25":
        raise ProtocolError("Python 版本不匹配冻结环境")
    versions = {name: metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ProtocolError("核心依赖版本不匹配冻结环境")
    sources: Dict[str, str] = {}
    packages = ("nnunetv2", "batchgenerators", "batchgeneratorsv2", "dynamic-network-architectures")
    for name in packages:
        dist = metadata.distribution(name)
        versions[name] = dist.version
        for entry in dist.files or ():
            rel = str(entry)
            if rel.endswith(".py") and ".." not in entry.parts:
                # 旧自定义安装入口不属于 V5 运行依赖。
                if "/variants/" in rel:
                    continue
                sources[rel] = file_sha256(dist.locate_file(entry))
    for rel, expected in PINNED_SOURCES.items():
        if sources.get(rel) != expected:
            raise ProtocolError("上游关键源码哈希变化: " + rel)
    for name in ("scipy", "scikit-image", "scikit-learn", "threadpoolctl"):
        versions[name] = metadata.version(name)
    return {"python": platform.python_version(), "versions": versions, "sources": sources}


def local_fingerprint() -> Dict[str, str]:
    base = Path(__file__).parent
    return {name: file_sha256(base / name) for name in LOCAL_MODULES}


def _base_plans(path: PathLike) -> Dict[str, Any]:
    if file_sha256(path) != BASE_PLANS_SHA256:
        raise ProtocolError("冻结基线 plans 来源哈希不匹配")
    return read_json(path)


def adapted_plans(base: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    if arm not in DATASETS:
        raise ProtocolError("不支持的训练臂")
    plans = copy.deepcopy(dict(base))
    plans["dataset_name"] = DATASETS[arm]
    config = plans["configurations"]["3d_fullres"]
    channels = 1 if arm == "C" else 3
    config["normalization_schemes"] = ["NoNormalization"] * channels
    config["use_mask_for_norm"] = [False] * channels
    return plans


def dataset_description(arm: str, count: int) -> Dict[str, Any]:
    return {"channel_names": {"0": "CT"} if arm == "C" else {"0": "CT", "1": "ownMRI", "2": "valid"},
            "labels": {"background": 0, "Y": 1}, "numTraining": count, "file_ending": ".nii.gz"}


@dataclass(frozen=True)
class V5Protocol:
    """不可变 canonical JSON；to_dict 返回副本，不暴露可变内部状态。"""
    _json: str

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self._json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._json.encode("utf-8")).hexdigest()

    @property
    def split_index(self) -> int:
        p = self.to_dict()
        return p["fold"] + (5 if p["mode"] == "dev" else 0)

    def plans(self) -> Dict[str, Any]:
        return self.to_dict()["plans"]

    def envelope(self) -> Dict[str, Any]:
        return {"protocol": self.to_dict(), "sha256": self.sha256}

    def identity(self) -> Dict[str, Any]:
        p = self.to_dict()
        return {"version": VERSION, "arm": p["arm"], "fold": p["fold"], "mode": p["mode"],
                "training_seed": 42, "protocol_sha256": self.sha256,
                "plans_sha256": digest(p["plans"]), "split_sha256": digest(p["splits"]),
                "gt_sha256": p["gt_sha256"], "source_sha256": digest(p["sources"]),
                "upstream_sha256": digest(p["upstream"]), "checkpoint_selection": "final"}


def freeze_protocol(*, arm: str, fold: int, mode: str, cohort: Sequence[str],
                    splits_file: PathLike, gt_sha256: str, root: PathLike = ROOT,
                    test_root: bool = False, plans_source: PathLike = BASE_PLANS) -> V5Protocol:
    """只读加载既有划分并返回冻结协议；序列化和真实数据生成由调用方负责。"""
    base = guard_root(root, test_root)
    splits = read_json(splits_file)
    validate_cohort_split(cohort, splits, fold, mode)
    if not _sha(gt_sha256):
        raise ProtocolError("必须提供三臂共享 GT 内容哈希")
    plans = adapted_plans(_base_plans(plans_source), arm)
    payload = {"version": VERSION, "arm": arm, "fold": fold, "mode": mode,
               "root": str(base), "test_root": test_root, "recipe": copy.deepcopy(RECIPE),
               "stream_scheme": copy.deepcopy(STREAM_SCHEME), "legacy_algorithm": copy.deepcopy(LEGACY_ALGORITHM),
               "cohort": sorted(_identifiers(cohort)), "splits": splits,
               "split_source_sha256": file_sha256(splits_file), "gt_sha256": gt_sha256,
               "plans_source_sha256": BASE_PLANS_SHA256, "plans_source": str(Path(plans_source).resolve()),
               "plans": plans, "dataset_json": dataset_description(arm, len(cohort)),
               "upstream": upstream_fingerprint(), "sources": local_fingerprint()}
    return V5Protocol(canonical_json(payload))


def load_protocol(source: Union[PathLike, Mapping[str, Any]]) -> V5Protocol:
    envelope = read_json(source) if isinstance(source, (str, Path)) else copy.deepcopy(dict(source))
    if not isinstance(envelope, dict) or set(envelope) != {"protocol", "sha256"}:
        raise ProtocolError("未知协议封装 schema")
    p = envelope["protocol"]
    required = {"version", "arm", "fold", "mode", "root", "test_root", "recipe", "stream_scheme",
                "legacy_algorithm", "cohort", "splits", "split_source_sha256", "gt_sha256",
                "plans_source_sha256", "plans_source", "plans", "dataset_json", "upstream", "sources"}
    if not isinstance(p, dict) or set(p) != required or p["version"] != VERSION:
        raise ProtocolError("未知 V5 协议 schema")
    protocol = V5Protocol(canonical_json(p))
    if protocol.sha256 != envelope["sha256"]:
        raise ProtocolError("协议哈希不匹配")
    if p["recipe"] != RECIPE or p["stream_scheme"] != STREAM_SCHEME or p["legacy_algorithm"] != LEGACY_ALGORITHM:
        raise ProtocolError("协议偏离固定配方或子流方案")
    if p["arm"] not in DATASETS or type(p["test_root"]) is not bool:
        raise ProtocolError("协议臂或根类型无效")
    guard_root(p["root"], p["test_root"])
    validate_cohort_split(p["cohort"], p["splits"], p["fold"], p["mode"])
    if not all(_sha(p[key]) for key in ("gt_sha256", "split_source_sha256", "plans_source_sha256")):
        raise ProtocolError("协议来源哈希无效")
    if p["plans_source_sha256"] != BASE_PLANS_SHA256:
        raise ProtocolError("基线 plans 来源变化")
    if p["dataset_json"] != dataset_description(p["arm"], len(p["cohort"])):
        raise ProtocolError("通道/标签/病例数不符合固定协议")
    return protocol


def verify_sources(protocol: V5Protocol) -> None:
    p = load_protocol(protocol.envelope()).to_dict()
    if adapted_plans(_base_plans(p["plans_source"]), p["arm"]) != p["plans"]:
        raise ProtocolError("plans 非冻结基线的通道适配副本")
    if upstream_fingerprint() != p["upstream"] or local_fingerprint() != p["sources"]:
        raise ProtocolError("上游或 V5 源码与冻结协议不一致，拒绝训练/恢复")


def run_directory(protocol: V5Protocol) -> Path:
    p = protocol.to_dict()
    base = guard_root(p["root"], p["test_root"])
    path = base / "nnUNet_results" / DATASETS[p["arm"]] / "V5Trainer__nnUNetPlans__3d_fullres" / f"fold_{protocol.split_index}"
    return guard_child(base, path)


def validate_training_inputs(protocol: V5Protocol) -> None:
    """无副作用预检；实际几何/GT 内容核对仍是数据生产任务的责任。"""
    verify_sources(protocol)
    p = protocol.to_dict()
    root = guard_root(p["root"], p["test_root"])
    folder = guard_child(root, root / "nnUNet_preprocessed" / DATASETS[p["arm"]])
    for name in ("nnUNetPlans.json", "dataset.json", "splits_final.json", "v5_protocol.json"):
        path = guard_child(root, folder / name)
        if not path.is_file():
            raise ProtocolError("缺少 V5 预处理输入元数据: " + name)
    if read_json(folder / "nnUNetPlans.json") != p["plans"]:
        raise ProtocolError("实际 plans 与协议不匹配")
    if read_json(folder / "dataset.json") != p["dataset_json"]:
        raise ProtocolError("实际 dataset.json 与协议不匹配")
    if read_json(folder / "splits_final.json") != p["splits"]:
        raise ProtocolError("实际 split 与原冻结划分不同")
    registered = read_json(folder / "v5_protocol.json")
    if registered != data_registration(protocol):
        raise ProtocolError("数据生产来源登记不匹配")
    data = guard_child(root, folder / p["plans"]["configurations"]["3d_fullres"]["data_identifier"])
    if not data.is_dir():
        raise ProtocolError("预处理数据目录不存在")
    actual = sorted(f.stem for f in data.glob("*.npz"))
    if actual != sorted(p["cohort"]):
        raise ProtocolError("真实病例列表与冻结 cohort 存在缺项或额外项")
    for case in actual:
        for suffix in (".npz", ".pkl", ".npy", "_seg.npy"):
            path = guard_child(root, data / (case + suffix))
            if suffix in (".npz", ".pkl") and not path.is_file():
                raise ProtocolError("病例数据或采样属性文件缺失")


def data_registration(protocol: V5Protocol) -> Dict[str, Any]:
    """供任务19写入预处理目录 v5_protocol.json 的最小来源登记。"""
    p = protocol.to_dict()
    return {"version": VERSION, "arm": p["arm"], "gt_sha256": p["gt_sha256"],
            "cohort_sha256": digest(p["cohort"]), "split_sha256": digest(p["splits"]),
            "plans_sha256": digest(p["plans"]), "split_source_sha256": p["split_source_sha256"]}
