# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.llama import LlamaConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

try:
    from fastapi.testclient import TestClient

    from levanter.main.sample_lm import (
        SampleLmConfig,
        app,
        initialize_service,
    )

except ImportError:
    pytest.skip("Serving imports not installed, use --extra=serve", allow_module_level=True)


def test_endpoints_exist():
    """Test that the endpoints are properly defined"""
    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/v1/completions" in routes
    assert "/v1/chat/completions" in routes


@pytest.mark.slow
def test_baby_llama_integration():
    """
    Integration test that replicates running:
    uv run python src/levanter/main/sample_lm.py --hf_checkpoint timinar/baby-llama-58m
    --tokenizer timinar/baby-llama-58m --model.type llama --trainer.wandb.mode=disabled

    Sends a short prompt with max_tokens=32 and stop="."
    """

    # Configure the model similar to CLI arguments
    config = SampleLmConfig(
        hf_checkpoint=RepoRef("timinar/baby-llama-58m"),
        tokenizer="timinar/baby-llama-58m",
        model=LlamaConfig(),
        trainer=TrainerConfig(wandb=WandbConfig(mode="disabled")),
        max_new_tokens=32,
        temperature=0.7,
        seed=42,
    )

    # Initialize the service
    initialize_service(config)

    # Create test client with the configured app using context manager to handle lifespan events
    with TestClient(app) as client:
        # Test completion request with short prompt, max_tokens=32, and stop="."
        response = client.post(
            "/v1/completions",
            json={
                "model": "timinar/baby-llama-58m",
                "prompt": "The quick brown fox",
                "max_tokens": 32,
                "temperature": 0.7,
                "stop": ".",
                "seed": 42,
            },
        )

        # Verify successful response
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI API structure
        assert "id" in data
        assert data["object"] == "text_completion"
        assert "created" in data
        assert data["model"] == "timinar/baby-llama-58m"
        assert "choices" in data
        assert "usage" in data

        # Verify choices structure
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert "text" in choice
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"

        # Verify usage tracking
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

        # Verify the generated text exists and is reasonable
        generated_text = choice["text"]
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0

        # Test that completion tokens doesn't exceed max_tokens
        assert usage["completion_tokens"] <= 32

        # If response finished with stop token, check that it doesn't contain "." at the end
        # (unless the generation naturally ended with ".")
        print(f"Generated text: '{generated_text}'")
        print(f"Usage: {usage}")


@pytest.mark.slow
def test_weight_reloading_during_requests():
    """
    Test that weight reloading works correctly while requests are being processed.

    This test queues multiple requests on a background thread and triggers a reload
    while they are being processed, ensuring all requests complete successfully.
    """
    # Configure the model
    config = SampleLmConfig(
        hf_checkpoint=RepoRef("timinar/baby-llama-58m"),
        tokenizer="timinar/baby-llama-58m",
        model=LlamaConfig(),
        trainer=TrainerConfig(wandb=WandbConfig(mode="disabled")),
        max_new_tokens=8,
        temperature=0.7,
        seed=42,
    )

    # Initialize the service
    initialize_service(config)

    # Create test client with the configured app using context manager to handle lifespan events
    with TestClient(app) as client:
        from levanter.main.sample_lm import inference_context

        # Wait for the inference service to fully initialize
        time.sleep(1.0)

        if not inference_context:
            pytest.skip("Inference context not initialized")

        # Dummy weight callback that just returns the same model
        def dummy_weight_callback(model):
            time.sleep(0.1)  # Simulate some work
            return model

        # Submit several requests concurrently using ThreadPoolExecutor
        def make_request(request_id):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "timinar/baby-llama-58m",
                    "prompt": f"Request {request_id}: The quick brown fox",
                    "max_tokens": 8,
                    "temperature": 0.7,
                    "seed": request_id,
                },
            )
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text,
            }

        # Start multiple concurrent requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit requests that will be processed before, during, and after reload
            futures = [executor.submit(make_request, i) for i in range(6)]

            # Wait a moment for some requests to start processing
            time.sleep(0.2)

            # Trigger model reload while requests are in flight
            print("Triggering model reload...")
            reload_start = time.time()
            inference_context.reload(dummy_weight_callback)
            reload_duration = time.time() - reload_start
            print(f"Model reload completed in {reload_duration:.2f}s")

            # Collect all results
            results = [future.result() for future in futures]

        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        failed_requests = [r for r in results if r["status_code"] != 200]

        print(f"Total successful requests: {len(successful_requests)}")
        print(f"Total failed requests: {len(failed_requests)}")

        # Verify all requests completed successfully
        assert len(successful_requests) > 0, "Expected at least some successful requests"
        assert len(failed_requests) == 0, f"No requests should fail, but got: {failed_requests}"

        # Verify response structure for successful requests
        for result in successful_requests:
            response_data = result["response"]
            assert "choices" in response_data
            assert "usage" in response_data
            assert len(response_data["choices"]) > 0
            assert "text" in response_data["choices"][0]

        print("Weight reloading test passed successfully!")
