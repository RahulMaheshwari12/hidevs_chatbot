"""API tests. They use a fake agent, so no Gemini or Qdrant calls are made."""

from fastapi.testclient import TestClient

from app.main import app


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = []

    def process_message(self, message):
        self.received.append(message)
        if self.error:
            raise self.error
        return self.result


def make_client(agent):
    """Build a client whose lifespan is skipped and whose agent is the fake."""
    app.state.agent = agent
    return TestClient(app, raise_server_exceptions=False)


def test_health():
    client = make_client(FakeAgent())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources():
    agent = FakeAgent(
        result={
            "answer": "LeetPrompt is a prompt-engineering platform.",
            "intent": "knowledge_query",
            "sources": [
                {
                    "content": "LeetPrompt passage",
                    "metadata": {
                        "source_title": "Platform",
                        "section_number": "4",
                        "section_title": "LeetPrompt",
                        "score": 0.9,
                        "point_id": "abc",
                    },
                }
            ],
        }
    )
    client = make_client(agent)

    response = client.post("/api/chat", json={"message": "  What is LeetPrompt?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "knowledge_query"
    assert body["answer"].startswith("LeetPrompt")
    assert body["sources"] == [
        {
            "source_title": "Platform",
            "section_number": "4",
            "section_title": "LeetPrompt",
            "content": "LeetPrompt passage",
        }
    ]
    # The message is stripped before it reaches the agent.
    assert agent.received == ["What is LeetPrompt?"]


def test_chat_small_talk_has_no_sources():
    agent = FakeAgent(
        result={"answer": "Hello!", "intent": "small_talk", "sources": []}
    )
    client = make_client(agent)

    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Hello!", "intent": "small_talk", "sources": []}


def test_chat_rejects_empty_message():
    client = make_client(FakeAgent(result={}))

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_rejects_missing_message():
    client = make_client(FakeAgent(result={}))

    response = client.post("/api/chat", json={})

    assert response.status_code == 422


def test_chat_hides_internal_errors():
    client = make_client(FakeAgent(error=RuntimeError("secret qdrant detail")))

    response = client.post("/api/chat", json={"message": "What is LeetPrompt?"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_cors_allows_configured_origin():
    client = make_client(FakeAgent())

    response = client.get(
        "/api/health", headers={"Origin": "http://localhost:3000"}
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
