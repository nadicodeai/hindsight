"""Inside and outside: with ``inside_platforms`` set, memory belongs to people and to the company.
Each case drives the provider through the Hermes interface and asserts which banks it reads and
writes on the recording fake client."""

import json
import sys
import types

import hindsight_hermes as plugin
import pytest
from conftest import FakeClient

INSIDE = {"inside_platforms": "buzz"}


class _ScoredClient(FakeClient):
    """Answers each bank's recall with its own facts, each with a final score."""

    def __init__(self, facts: dict[str, list[tuple[str, float]]], reflect_text: str = ""):
        super().__init__(reflect_text=reflect_text)
        self._facts = facts
        self.banks_named: list[tuple[str, str]] = []

    async def arecall(self, **kwargs):
        self.recalls.append(kwargs)
        results = [
            types.SimpleNamespace(text=text, scores=types.SimpleNamespace(final=score))
            for text, score in self._facts.get(kwargs["bank_id"], [])
        ]
        return types.SimpleNamespace(results=results)

    async def acreate_bank(self, bank_id, name=None, **kwargs):
        self.banks_named.append((bank_id, name))


def _searched(fake: FakeClient) -> list[str]:
    return [call["bank_id"] for call in fake.recalls]


def test_an_inside_sender_reads_their_own_memory_and_the_company_knowledge(provider):
    instance, fake = provider(INSIDE, platform="buzz", user_id="ab12", user_name="Mario")
    instance.handle_tool_call("long_term_memory_search", {"query": "opening hours"})
    instance.shutdown()
    assert _searched(fake) == ["person-buzz-ab12", "company", "public"]


def test_an_outside_sender_reads_their_own_memory_and_public_knowledge_only(provider):
    instance, fake = provider(INSIDE, platform="whatsapp", user_id="393331234567", user_name="Rossi")
    instance.handle_tool_call("long_term_memory_search", {"query": "margin"})
    instance.shutdown()
    assert _searched(fake) == ["person-whatsapp-393331234567", "public"]


def test_no_sender_is_the_owner_and_inside(provider):
    instance, fake = provider(INSIDE, platform="cli")
    instance.handle_tool_call("long_term_memory_search", {"query": "anything"})
    instance.shutdown()
    assert _searched(fake) == ["person-owner", "company", "public"]


def test_the_turn_is_saved_in_the_senders_own_memory(provider):
    instance, fake = provider(INSIDE, platform="telegram", user_id="777", user_name="Bianchi")
    instance.sync_turn("I'm Bianchi", "Hello.")
    instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["person-telegram-777"]


def test_an_inside_sender_saves_for_the_team_or_for_everyone(provider):
    instance, fake = provider(INSIDE, platform="buzz", user_id="ab12")
    team = json.loads(instance.handle_tool_call("long_term_memory_save", {"content": "Margin 32%", "for": "team"}))
    everyone = json.loads(
        instance.handle_tool_call("long_term_memory_save", {"content": "Opens at 7:30", "for": "everyone"})
    )
    mine = json.loads(instance.handle_tool_call("long_term_memory_save", {"content": "On holiday"}))
    instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["company", "public", "person-buzz-ab12"]
    assert team["result"] == "Saved in the team's knowledge."
    assert everyone["result"] == "Saved in public knowledge."
    assert mine["result"] == "Saved in this person's own memory."


def test_an_outside_save_for_everyone_stays_in_their_own_memory_and_says_so(provider):
    instance, fake = provider(INSIDE, platform="telegram", user_id="777")
    result = json.loads(
        instance.handle_tool_call("long_term_memory_save", {"content": "Closed forever", "for": "everyone"})
    )
    instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["person-telegram-777"]
    assert result["result"].startswith("Saved only in this person's own memory")


def test_results_from_several_memories_are_ranked_by_score(provider):
    client = _ScoredClient(
        {"person-buzz-ab12": [("own", 0.2)], "company": [("team", 0.9)], "public": [("public", 0.5)]}
    )
    instance, _ = provider(INSIDE, client=client, platform="buzz", user_id="ab12")
    result = json.loads(instance.handle_tool_call("long_term_memory_search", {"query": "q"}))
    instance.shutdown()
    assert result["result"] == "1. team\n2. public\n3. own"


def test_an_inside_sender_finds_an_outside_person_by_name(provider, monkeypatch):
    instance, fake = provider(INSIDE, platform="buzz", user_id="ab12")
    monkeypatch.setattr(instance, "_outside_banks_named", lambda name: ["person-whatsapp-393331234567"])
    instance.handle_tool_call("long_term_memory_search", {"query": "order", "client": "Rossi"})
    instance.shutdown()
    assert _searched(fake) == ["person-whatsapp-393331234567"]


def test_an_outside_sender_cannot_search_another_person(provider, monkeypatch):
    instance, fake = provider(INSIDE, platform="telegram", user_id="777")
    monkeypatch.setattr(instance, "_outside_banks_named", lambda name: ["person-whatsapp-393331234567"])
    instance.handle_tool_call("long_term_memory_search", {"query": "order", "client": "Rossi"})
    instance.shutdown()
    assert _searched(fake) == ["person-telegram-777", "public"]


def test_the_name_lookup_skips_inside_people_and_the_company(provider, monkeypatch):
    instance, _ = provider(INSIDE, platform="buzz", user_id="ab12")
    listing = {
        "banks": [
            {"bank_id": "person-whatsapp-1", "name": "Giulia Rossi"},
            {"bank_id": "person-buzz-ab99", "name": "Rossi the colleague"},
            {"bank_id": "person-owner", "name": "Rossi"},
            {"bank_id": "company", "name": "Rossi"},
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(listing).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    assert instance._outside_banks_named("rossi") == ["person-whatsapp-1"]
    instance.shutdown()


def test_reflect_names_the_memory_of_each_answer(provider):
    instance, fake = provider(INSIDE, client=FakeClient(reflect_text="Opens at 7:30."), platform="buzz", user_id="ab12")
    result = json.loads(instance.handle_tool_call("long_term_memory_reflect", {"query": "hours?"}))["result"]
    instance.shutdown()
    assert [call["bank_id"] for call in fake.reflects] == ["person-buzz-ab12", "company", "public"]
    assert "From this person's own memory:" in result
    assert "From the team's knowledge:" in result and "From public knowledge:" in result


def test_an_outside_persons_memory_is_named_after_them(provider):
    client = _ScoredClient({})
    instance, _ = provider(INSIDE, client=client, platform="whatsapp", user_id="39333", user_name="Giulia Rossi")
    instance.handle_tool_call("long_term_memory_save", {"content": "Order 9921 to Padova"})
    instance.shutdown()
    assert client.banks_named == [("person-whatsapp-39333", "Giulia Rossi")]


def test_the_settings_come_from_the_environment_without_a_config_file(hermes_env):
    for name, value in {
        "HINDSIGHT_MODE": "local_external",
        "HINDSIGHT_INSIDE_PLATFORMS": "buzz",
        "HINDSIGHT_RETAIN_CONTEXT": "conversation with Nadia",
        "HINDSIGHT_RETAIN_INDICATOR": "false",
        "HINDSIGHT_RECALL_INDICATOR": "false",
        "HINDSIGHT_RECALL_SYNC": "true",
    }.items():
        plugin_secrets = __import__("conftest").SECRETS
        plugin_secrets[name] = value
    cfg = plugin._load_config()
    assert cfg["mode"] == "local_external"
    assert cfg["inside_platforms"] == "buzz"
    assert cfg["retain_context"] == "conversation with Nadia"
    assert cfg["retain_indicator"] is False and cfg["recall_indicator"] is False
    assert cfg["recall_sync"] is True


def test_without_inside_platforms_the_provider_keeps_one_bank(provider):
    instance, fake = provider({"bank_id": "team"}, platform="whatsapp", user_id="393331234567")
    instance.handle_tool_call("long_term_memory_save", {"content": "x", "for": "everyone"})
    instance.handle_tool_call("long_term_memory_search", {"query": "x", "client": "Rossi"})
    instance.shutdown()
    assert [call["bank_id"] for call in fake.retains] == ["team"]
    assert _searched(fake) == ["team"]
    assert all("for" not in s["parameters"]["properties"] for s in instance.get_tool_schemas())


def test_the_embedded_servers_llm_and_login_come_from_the_environment(hermes_env):
    for name, value in {
        "HINDSIGHT_LLM_PROVIDER": "openai",
        "HINDSIGHT_LLM_MODEL": "deepseek/deepseek-v4.1-flash",
        "HINDSIGHT_LLM_CREDENTIAL_POOL": "nadicode",
    }.items():
        __import__("conftest").SECRETS[name] = value
    cfg = plugin._load_config()
    assert cfg["llm_provider"] == "openai"
    assert cfg["llm_model"] == "deepseek/deepseek-v4.1-flash"
    assert cfg["llm_credential_pool"] == "nadicode"


NADIA_MEMORY_RULES = {
    "HINDSIGHT_MODE": "local_embedded",
    "HINDSIGHT_INSIDE_PLATFORMS": "buzz",
    "HINDSIGHT_RECALL_INDICATOR": "false",
    "HINDSIGHT_RECALL_SYNC": "true",
}


@pytest.mark.parametrize(("pinned", "mode", "inside", "recall_sync"), [(True, "local_embedded", "buzz", True), (False, "cloud", "", False)])
def test_a_profile_s_config_file_keeps_its_own_settings_but_not_those_the_managed_env_pins(
    hermes_env, monkeypatch, pinned, mode, inside, recall_sync
):
    __import__("conftest").SECRETS.update(NADIA_MEMORY_RULES)
    managed = frozenset(NADIA_MEMORY_RULES) if pinned else frozenset()
    monkeypatch.setitem(sys.modules, "hermes_cli.env_loader", types.SimpleNamespace(managed_dotenv_keys=lambda: managed))
    (hermes_env / "hindsight").mkdir()
    (hermes_env / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "cloud", "recall_sync": False, "bank_id": "shared"}), encoding="utf-8"
    )

    cfg = plugin._load_config()

    assert (cfg["mode"], cfg.get("inside_platforms", ""), cfg["recall_sync"]) == (mode, inside, recall_sync)
    assert cfg["bank_id"] == "shared"
