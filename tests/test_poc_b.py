from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POC_B = ROOT / "poc_b"
SCRIPTS = POC_B / "scripts"


def run_b_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SCRIPTS) if not existing else f"{SCRIPTS}{os.pathsep}{existing}"
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class PocBDataTest(unittest.TestCase):
    def test_builder_is_deterministic_and_alias_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            outputs = {
                "train-output": destination / "train.jsonl",
                "canonical-output": destination / "eval_canonical.jsonl",
                "alias-output": destination / "eval_alias.jsonl",
                "class-map-output": destination / "class_map.json",
                "manifest-output": destination / "data_manifest.json",
            }
            command = [sys.executable, str(SCRIPTS / "build_training_data.py")]
            for argument, path in outputs.items():
                command.extend([f"--{argument}", str(path)])
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            committed = {
                "train-output": POC_B / "data/train.jsonl",
                "canonical-output": POC_B / "data/eval_canonical.jsonl",
                "alias-output": POC_B / "data/eval_alias.jsonl",
                "class-map-output": POC_B / "data/class_map.json",
                "manifest-output": POC_B / "data/data_manifest.json",
            }
            for key, generated in outputs.items():
                self.assertEqual(generated.read_bytes(), committed[key].read_bytes())

            class_map = json.loads(outputs["class-map-output"].read_text())
            classes = class_map["classes"]
            self.assertEqual(len(classes), 1000)
            self.assertEqual(
                [row["appid"] for row in classes],
                sorted(row["appid"] for row in classes),
            )
            train = [json.loads(line) for line in outputs["train-output"].read_text().splitlines()]
            canonical = [json.loads(line) for line in outputs["canonical-output"].read_text().splitlines()]
            alias = [json.loads(line) for line in outputs["alias-output"].read_text().splitlines()]
            self.assertEqual((len(train), len(canonical), len(alias)), (6000, 1000, 184))
            self.assertTrue(all(row["surface_form"] == row["canonical_name"] for row in train))
            self.assertTrue(all(row["model_input"].endswith(row["surface_form"]) for row in train + canonical + alias))
            train_surfaces = {row["surface_form"].casefold() for row in train}
            self.assertFalse(
                train_surfaces.intersection(row["surface_form"].casefold() for row in alias)
            )

    def test_repository_data_and_poc_a_snapshot_pass_contracts(self) -> None:
        result = run_b_python(
            """
from pathlib import Path
from training_common import load_config, load_poc_a_reference, validate_data
config = load_config(Path('poc_b/configs/qwen3_8b_frozen_prototype.yaml').resolve())
bundle = validate_data(config)
reference = load_poc_a_reference(config)
assert (bundle.class_count, len(bundle.train_rows), len(bundle.canonical_rows), len(bundle.alias_rows)) == (1000, 6000, 1000, 184)
assert reference['source']['revision'] == '013d72f9bd85f2eb314eb1d178c117049125dc14'
assert reference['canonical']['top1_accuracy'] == 0.979
assert reference['alias']['top1_accuracy'] == 15 / 184
print('ok')
"""
        )
        self.assertEqual(result.stdout.strip(), "ok")


class PocBClassifierTest(unittest.TestCase):
    def test_fake_backbone_tiny_end_to_end_smoke(self) -> None:
        result = run_b_python(
            r"""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch import nn
from evaluation_core import checkpoint_metrics
from feature_cache import cache_identity, stable_sha256, tokenizer_sha256
from steam_entity_classifier import (
    FrozenPrototypeHead, SteamEntityLinker, classifier_config, decode_logits, extract_features,
    freeze_backbone, initialize_prototypes, pool_last_non_padding,
    validate_classifier_config,
)
from training_common import TrainingToolError, select_best_checkpoint
from train import validate_resume_identity, validate_resume_progress

torch.manual_seed(42)

class FakeTokenizer:
    padding_side = 'right'
    truncation_side = 'right'
    model_max_length = 32
    special_tokens_map = {'pad_token': '<pad>'}
    def get_vocab(self):
        return {'<pad>': 0, 'A': 1, 'B': 2, 'x': 3}
    def __call__(self, texts, **kwargs):
        del kwargs
        encoded = []
        for text in texts:
            last = text[-1]
            token = 1 if last in {'A', 'a'} else 2 if last in {'B', 'b'} else 3
            encoded.append([3, token] if len(text) > 1 else [token])
        width = max(map(len, encoded))
        ids = [row + [0] * (width - len(row)) for row in encoded]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.config = SimpleNamespace(hidden_size=4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([
                [0., 0., 0., 0.], [1., 0., 0., 0.],
                [0., 1., 0., 0.], [0., 0., 1., 0.],
            ]))
    def forward(self, input_ids, attention_mask, return_dict=True):
        del attention_mask, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
pooled = pool_last_non_padding(hidden, mask)
assert torch.equal(pooled, torch.stack((hidden[0, 2], hidden[1, 1])))

tokenizer = FakeTokenizer()
backbone = freeze_backbone(FakeBackbone())
assert not backbone.training
assert not any(parameter.requires_grad for parameter in backbone.parameters())
train_texts = ['A', 'prefixA', 'B', 'prefixB']
train_features = extract_features(backbone, tokenizer, train_texts, batch_size=2, max_length=16, device=torch.device('cpu'))
labels = torch.tensor([0, 0, 1, 1])
prototypes = initialize_prototypes(train_features, labels, 2)
head = FrozenPrototypeHead(prototypes, bottleneck_dim=2, temperature=0.05)
assert torch.count_nonzero(head.up.weight) == 0
assert torch.count_nonzero(head.up.bias) == 0
expected_logits = F.normalize(train_features, dim=-1) @ F.normalize(prototypes, dim=-1).T / 0.05
assert torch.allclose(head(train_features), expected_logits)

optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
for _ in range(3):
    optimizer.zero_grad()
    loss, parts = head.training_loss(train_features, labels, anchor_weight=0.01)
    assert set(parts) == {'loss', 'cross_entropy', 'prototype_anchor'}
    loss.backward()
    optimizer.step()

classes = [
    {'class_index': 0, 'appid': 10, 'canonical_name': 'A', 'cohort': 'popular'},
    {'class_index': 1, 'appid': 20, 'canonical_name': 'B', 'cohort': 'latest'},
]
canonical_rows = [
    {'surface_form': 'A', 'model_input': 'A', 'class_index': 0, 'appid': 10, 'canonical_name': 'A', 'cohort': 'popular', 'prompt_style': 'raw', 'type': 'canonical'},
    {'surface_form': 'B', 'model_input': 'B', 'class_index': 1, 'appid': 20, 'canonical_name': 'B', 'cohort': 'latest', 'prompt_style': 'raw', 'type': 'canonical'},
]
alias_rows = [
    {'surface_form': 'a', 'model_input': 'a', 'class_index': 0, 'appid': 10, 'canonical_name': 'A', 'cohort': 'popular', 'prompt_style': 'raw', 'type': 'alias'},
    {'surface_form': 'b', 'model_input': 'b', 'class_index': 1, 'appid': 20, 'canonical_name': 'B', 'cohort': 'latest', 'prompt_style': 'raw', 'type': 'abbreviation'},
]
canonical_features = extract_features(backbone, tokenizer, ['A', 'B'], batch_size=2, max_length=16, device=torch.device('cpu'))
alias_features = extract_features(backbone, tokenizer, ['a', 'b'], batch_size=2, max_length=16, device=torch.device('cpu'))
metric, records = checkpoint_metrics(epoch=1, head=head, canonical_features=canonical_features, alias_features=alias_features, canonical_rows=canonical_rows, alias_rows=alias_rows, classes=classes, batch_size=2, diagnostic_top_k=2, device=torch.device('cpu'))
assert metric['canonical']['top1_accuracy'] == 1.0
assert metric['alias']['top5_accuracy'] == 1.0
assert metric['alias']['mrr'] == 1.0
assert len(metric['prediction_sha256']) == 64 and len(records) == 4

decoded = decode_logits(torch.tensor([[3.0, 1.0]]), classes, top_k=2)[0]
assert decoded['prediction']['appid'] == 10
assert len(decoded['top_k']) == 2
linker = SteamEntityLinker(backbone=backbone, tokenizer=tokenizer, head=head, classifier_settings={'prompt_template': '{surface_form}', 'max_length': 16}, classes=classes, device=torch.device('cpu'))
prediction = linker.predict(['a'], top_k=2)[0]
assert prediction['appid'] == 10
assert {'appid', 'canonical_name', 'class_index', 'confidence', 'top_k'} <= set(prediction)

tokenizer_hash = tokenizer_sha256(tokenizer)
identity = cache_identity(mode='smoke', model_id='fake/model', model_revision='abc', tokenizer_hash=tokenizer_hash, data_sha256={'train': 'hash'}, max_length=16, pooling='last_non_padding', class_count=2, row_counts={'train': 4, 'canonical': 2, 'alias': 2})
assert stable_sha256(identity) == stable_sha256(dict(identity))
contract_head = FrozenPrototypeHead(torch.randn(32, 4), bottleneck_dim=256, temperature=0.05)
settings = classifier_config(contract_head, base_model_id='Qwen/Qwen3-8B-Base', base_model_revision='a' * 40, max_length=256, prototype_anchor_weight=0.01, feature_cache_sha256='b' * 64, tokenizer_sha256=tokenizer_hash, class_map_sha256='c' * 64, training_config_sha256='d' * 64, mode='smoke', epoch=1)
validate_classifier_config(settings)
assert settings['closed_set'] is True and settings['num_classes'] == 32

selected = select_best_checkpoint([
    {'epoch': 1, 'canonical': {'top1_accuracy': .96}, 'alias': {'top1_accuracy': .2}},
    {'epoch': 3, 'canonical': {'top1_accuracy': .97}, 'alias': {'top1_accuracy': .2}},
    {'epoch': 5, 'canonical': {'top1_accuracy': .94}, 'alias': {'top1_accuracy': .9}},
], .95)
assert selected['epoch'] == 3
resume_manifest = {
 'model': {'id': 'Qwen/Qwen3-8B-Base', 'revision': 'a' * 40},
 'mode': 'smoke', 'data_sha256': {'train': 'x'},
 'feature_cache_sha256': 'b' * 64, 'tokenizer_sha256': 'c' * 64,
 'class_map_sha256': 'd' * 64, 'training_config_sha256': 'e' * 64,
}
resume_metadata = {**resume_manifest, 'epoch': 3, 'global_step': 9}
validate_resume_identity(resume_manifest, resume_metadata)
assert validate_resume_progress({'schema_version': 1, 'epoch': 3, 'global_step': 9}, resume_metadata, {'epoch': 3}, steps_per_epoch=3) == (3, 9)
try:
 validate_resume_progress({'schema_version': 1, 'epoch': 3, 'global_step': 8}, resume_metadata, {'epoch': 3}, steps_per_epoch=3)
except TrainingToolError:
 pass
else:
 raise AssertionError('inconsistent resume progress was accepted')
print('ok')
"""
        )
        self.assertEqual(result.stdout.strip(), "ok")

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") is not None,
        "local runtime does not include safetensors",
    )
    def test_feature_cache_hash_rejects_tampering(self) -> None:
        result = run_b_python(
            r"""
import json, tempfile
from pathlib import Path
import torch
from feature_cache import cache_identity, load_feature_cache, save_feature_cache
from steam_entity_classifier import ClassifierArtifactError, FrozenPrototypeHead, classifier_config, load_classifier_artifact, save_classifier_artifact
from training_common import TrainingToolError
identity = cache_identity(mode='smoke', model_id='fake', model_revision='abc', tokenizer_hash='tok', data_sha256={'train': 'x'}, max_length=8, pooling='last_non_padding', class_count=2, row_counts={'train': 2, 'canonical': 2, 'alias': 1})
tensors = {
 'train_features': torch.eye(2), 'train_labels': torch.tensor([0, 1]),
 'canonical_features': torch.eye(2), 'canonical_labels': torch.tensor([0, 1]),
 'alias_features': torch.tensor([[1., 0.]]), 'alias_labels': torch.tensor([0]),
}
with tempfile.TemporaryDirectory() as temp:
 path = Path(temp)
 metadata = save_feature_cache(path, tensors, identity)
 loaded, _ = load_feature_cache(path, expected_identity=identity, expected_cache_sha256=metadata['cache_sha256'])
 assert torch.equal(loaded['train_features'], tensors['train_features'])
 raw = json.loads((path / 'feature_cache.json').read_text())
 raw['identity']['model_revision'] = 'tampered'
 (path / 'feature_cache.json').write_text(json.dumps(raw))
 try:
  load_feature_cache(path, expected_identity=identity)
 except TrainingToolError:
  pass
 else:
  raise AssertionError('tampered cache was accepted')
 head_dir = path / 'head'
 head = FrozenPrototypeHead(torch.randn(32, 4), bottleneck_dim=256, temperature=0.05)
 settings = classifier_config(
  head,
  base_model_id='Qwen/Qwen3-8B-Base',
  base_model_revision='a' * 40,
  max_length=256,
  prototype_anchor_weight=0.01,
  feature_cache_sha256=metadata['cache_sha256'],
  tokenizer_sha256='b' * 64,
  class_map_sha256='c' * 64,
  training_config_sha256='d' * 64,
  mode='smoke',
  epoch=1,
 )
 save_classifier_artifact(head, head_dir, config=settings)
 restored, restored_settings = load_classifier_artifact(head_dir)
 sample = torch.randn(3, 4)
 assert torch.equal(head(sample), restored(sample))
 assert restored_settings == settings
 (head_dir / 'model.safetensors').write_bytes(b'forbidden')
 try:
  load_classifier_artifact(head_dir)
 except ClassifierArtifactError as error:
  assert 'backbone weights' in str(error)
 else:
  raise AssertionError('embedded backbone weight was accepted')
print('ok')
"""
        )
        self.assertEqual(result.stdout.strip(), "ok")


class PocBPublicationAndNotebookTest(unittest.TestCase):
    def test_publisher_defaults_to_ready_repo_and_rejects_base_weights(self) -> None:
        result = run_b_python(
            r"""
import tempfile
from pathlib import Path
from publish_hf import (_assert_remote_file_set, _assert_staging_safe, _ensure_private, _make_public, configured_repository, parse_args, resolve_publish_repo_id)
from training_common import TrainingToolError
assert configured_repository()['status'] == 'ready'
assert resolve_publish_repo_id(None) == 'hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b'
assert parse_args(['--run-dir', 'run', '--dry-run']).dry_run
_assert_remote_file_set(['.gitattributes', 'a'], ['a'], require_complete=True)
try:
 _assert_remote_file_set(['stale'], ['a'], require_complete=False)
except TrainingToolError:
 pass
else:
 raise AssertionError('stale remote file was accepted')
class FakeApi:
 def __init__(self):
  self.private = False
  self.updates = []
 def model_info(self, *, repo_id):
  assert repo_id == 'org/model'
  return type('Info', (), {'private': self.private})()
 def update_repo_settings(self, *, repo_id, private):
  assert repo_id == 'org/model'
  self.private = private
  self.updates.append(private)
api = FakeApi()
_ensure_private(api, 'org/model')
_make_public(api, 'org/model')
assert api.updates == [True, False]
with tempfile.TemporaryDirectory() as temp:
 path = Path(temp)
 (path / 'model.safetensors').write_bytes(b'base')
 try:
  _assert_staging_safe(path, [])
 except TrainingToolError as error:
  assert 'Forbidden' in str(error)
 else:
  raise AssertionError('base model weight was accepted')
print('ok')
"""
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_model_card_contains_independent_loader_example(self) -> None:
        result = run_b_python(
            r"""
from pathlib import Path
from publish_hf import _model_card
from training_common import load_config
config = load_config(Path('poc_b/configs/qwen3_8b_frozen_prototype.yaml').resolve())
summary = {'count': 1, 'top1_correct': 1, 'top1_accuracy': 1.0, 'top5_correct': 1, 'top5_accuracy': 1.0, 'mrr': 1.0}
row = {'epoch': 20, 'canonical': summary, 'alias': summary}
metrics = {'zero_training_prototype': {'canonical': summary, 'alias': summary}, 'checkpoints': [row], 'selection': {'epoch': 20}, 'canonical_threshold': .95, 'alias_improved_over_poc_a': True}
manifest = {'model': {'id': 'Qwen/Qwen3-8B-Base', 'revision': 'abc'}, 'git': {'commit': 'deadbeef', 'remote': 'origin'}}
card = _model_card(repo_id='org/poc-b', manifest=manifest, metrics=metrics, config=config)
assert 'repo_dir = snapshot_download("org/poc-b")' in card
assert 'sys.path.insert(0, repo_dir)' in card
assert 'SteamEntityLinker.from_pretrained(repo_dir)' in card
assert 'does not support `UNKNOWN`' in card
assert 'feature cache' in card
print('ok')
"""
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_notebooks_compile_use_fixed_repo_and_contain_no_token(self) -> None:
        expected = "HF_REPO_ID = 'hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b'"
        for notebook in sorted((POC_B / "notebooks").glob("*.ipynb")):
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in payload["cells"]
                if cell["cell_type"] == "code"
            )
            for index, cell in enumerate(payload["cells"]):
                if cell["cell_type"] == "code":
                    compile(
                        "".join(cell.get("source", [])),
                        f"{notebook}:cell-{index}",
                        "exec",
                    )
            self.assertIn(expected, code)
            self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", code))
        training_code = (POC_B / "notebooks/runpod_training.ipynb").read_text()
        for entrypoint in (
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/publish_hf.py",
        ):
            self.assertIn(entrypoint, training_code)
        self.assertIn("PUBLISH_PUBLIC = False", training_code)


if __name__ == "__main__":
    unittest.main()
