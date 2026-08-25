from __future__ import annotations

import unittest

from common.huggingface_repositories import (
    RepositoryConfigError,
    load_repository_registry,
    repository_for,
    validate_repository_registry,
)


class HuggingFaceRepositoriesTest(unittest.TestCase):
    def test_pocs_use_two_distinct_model_repositories(self) -> None:
        registry = load_repository_registry()
        poc_a = repository_for("poc_a", registry)
        poc_b = repository_for("poc_b", registry)

        self.assertEqual(
            poc_a["repo_id"], "hxgdzyuyi/qwen3-8b-steam-entity-linking"
        )
        self.assertEqual(
            poc_b["repo_id"], "hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b"
        )
        self.assertEqual(poc_a["repo_type"], "model")
        self.assertEqual(poc_b["repo_type"], "model")
        self.assertNotEqual(poc_a["repo_id"], poc_b["repo_id"])
        self.assertEqual(poc_a["status"], "active")
        self.assertEqual(poc_b["status"], "planned")

    def test_registry_rejects_a_space_destination(self) -> None:
        invalid = {
            "schema_version": 1,
            "repositories": {
                "poc_a": {
                    "repo_id": "org/poc-a",
                    "repo_type": "model",
                    "status": "active",
                    "artifact_kind": "adapter",
                },
                "poc_b": {
                    "repo_id": "org/poc-b",
                    "repo_type": "space",
                    "status": "planned",
                    "artifact_kind": "classifier",
                },
            },
        }
        with self.assertRaises(RepositoryConfigError):
            validate_repository_registry(invalid)

    def test_registry_rejects_a_shared_destination(self) -> None:
        invalid = {
            "schema_version": 1,
            "repositories": {
                experiment: {
                    "repo_id": "org/shared",
                    "repo_type": "model",
                    "status": status,
                    "artifact_kind": artifact_kind,
                }
                for experiment, status, artifact_kind in (
                    ("poc_a", "active", "adapter"),
                    ("poc_b", "planned", "classifier"),
                )
            },
        }
        with self.assertRaises(RepositoryConfigError):
            validate_repository_registry(invalid)


if __name__ == "__main__":
    unittest.main()
