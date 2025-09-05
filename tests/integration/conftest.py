import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import levanter.serve.min_server as server_mod
from levanter.inference.service import GenerationService
from levanter.trainer import TrainerConfig
from tests.integration.create_tiny_llama import (
    create_tiny_llama_model,
    save_as_hf_model,
    create_test_tokenizer,
)


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory):
    """Build a tiny HF Llama model in a temporary directory and return its path.

    The directory is removed after the test session.
    """
    tmp_dir = tmp_path_factory.mktemp("tiny_llama")
    # Generate tiny model and tokenizer into tmp_dir
    model, config, vocab = create_tiny_llama_model()
    save_as_hf_model(model, config, vocab.size, str(tmp_dir))
    create_test_tokenizer(str(tmp_dir))

    yield tmp_dir

    # Cleanup
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def client(tiny_llama_dir: Path) -> TestClient:
    """In-process FastAPI TestClient backed by a real GenerationService using the tiny model."""
    trainer = TrainerConfig(require_accelerator=False, log_jaxprs=False, log_xla_hlo=False)

    # Initialize service using local HF checkpoint directory via hf_checkpoint
    server_mod._service = GenerationService(
        hf_checkpoint=str(tiny_llama_dir),
        tokenizer=str(tiny_llama_dir),
        trainer=trainer,
    )

    if not server_mod._service.ready():
        pytest.skip(f"Service failed to initialize: {getattr(server_mod._service, 'last_error', None)}")

    with TestClient(server_mod.app) as c:
        yield c

    server_mod._service = None
