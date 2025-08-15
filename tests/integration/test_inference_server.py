"""
Integration tests for the Levanter inference server (in-process).
"""

from fastapi.testclient import TestClient
import levanter.serve.min_server as server_mod


class TestInferenceServerIntegration:
    """Integration tests for the inference server."""

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert "detail" in data

    def test_completion_basic(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "Hello", "max_tokens": 3, "temperature": 0.7}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data and len(data["choices"]) == 1
        choice = data["choices"][0]
        assert "text" in choice and "finish_reason" in choice and "index" in choice
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] >= 1
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert choice["text"].startswith("Hello")
        assert len(choice["text"]) > len("Hello")

    def test_completion_longer_prompt(self, client: TestClient):
        prompt = "The quick brown fox jumps over the lazy dog"
        payload = {"model": "local-test", "prompt": prompt, "max_tokens": 4, "temperature": 0.7}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        choice = data["choices"][0]
        usage = data["usage"]
        assert choice["text"].startswith(prompt)
        assert usage["prompt_tokens"] > 5
        assert usage["completion_tokens"] >= 1

    def test_completion_with_stop(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "Hello there", "max_tokens": 10, "temperature": 0.7, "stop": ["."]}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"

    def test_completion_with_seed(self, client: TestClient):
        prompt = "The weather is"
        payload = {"model": "local-test", "prompt": prompt, "max_tokens": 3, "temperature": 0.1, "seed": 42}
        r1 = client.post("/v1/completions", json=payload)
        r2 = client.post("/v1/completions", json=payload)
        assert r1.status_code == 200 and r2.status_code == 200
        c1 = r1.json()["choices"][0]
        c2 = r2.json()["choices"][0]
        assert c1["text"] == c2["text"]

    def test_error_handling_empty_prompt(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "", "max_tokens": 5}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data and "prompt must be non-empty" in data["detail"]

    def test_error_handling_invalid_max_tokens(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "Hello", "max_tokens": -1}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code in [200, 422]
        if response.status_code == 422:
            assert "detail" in response.json()
        else:
            assert "choices" in response.json()

    def test_server_under_load(self, client: TestClient):
        import concurrent.futures

        def make_request():
            payload = {"model": "local-test", "prompt": "Test", "max_tokens": 2, "temperature": 0.7}
            r = client.post("/v1/completions", json=payload)
            return r.status_code == 200

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda _: make_request(), range(4)))
        assert all(results)

    def test_multiple_generations(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "Hello", "max_tokens": 3, "temperature": 0.7, "n": 3}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data and len(data["choices"]) == 3

        # Check each choice has correct index and structure
        for i, choice in enumerate(data["choices"]):
            assert choice["index"] == i
            assert "text" in choice and "finish_reason" in choice
            assert choice["text"].startswith("Hello")

        # Check usage reflects multiple generations
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] >= 3  # At least 1 token per generation
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_models_endpoint(self, client: TestClient):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data and len(data["data"]) >= 1
        model = data["data"][0]
        assert model["object"] == "model"
        assert server_mod._service is not None
        assert model["id"] == server_mod._service.model_id
        assert "created" in model
        assert model["owned_by"] == "levanter"

    def test_model_metadata(self, client: TestClient):
        payload = {"model": "local-test", "prompt": "Hello", "max_tokens": 1, "temperature": 0.7}
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert server_mod._service is not None
        assert "model" in data and data["model"] == server_mod._service.model_id
        assert data["object"] == "text_completion"
