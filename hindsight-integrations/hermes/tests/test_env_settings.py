"""Settings a profile can give through its environment (``.env``) when it has no
``hindsight/config.json``: the retain label, both status-line indicators and synchronous recall."""

import hindsight_hermes as plugin
from conftest import SECRETS, FakeClient


def _provider_from_env(monkeypatch, **env) -> plugin.HindsightMemoryProvider:
    SECRETS.update({"HINDSIGHT_MODE": "cloud", "HINDSIGHT_API_KEY": "test-key", **env})
    instance = plugin.HindsightMemoryProvider()
    monkeypatch.setattr(instance, "_new_cloud_client", lambda: FakeClient())
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)
    instance.initialize("session-1")
    return instance


def test_the_environment_sets_the_retain_label_the_indicators_and_sync_recall(hermes_env, monkeypatch):
    instance = _provider_from_env(
        monkeypatch,
        HINDSIGHT_RETAIN_CONTEXT="conversation between the support bot and a customer",
        HINDSIGHT_RETAIN_INDICATOR="false",
        HINDSIGHT_RECALL_INDICATOR="0",
        HINDSIGHT_RECALL_SYNC="true",
    )
    assert instance._retain_context == "conversation between the support bot and a customer"
    assert instance._retain_indicator is False
    assert instance._recall_indicator is False
    assert instance._recall_sync is True
    instance.shutdown()


def test_unset_variables_keep_the_defaults(hermes_env, monkeypatch):
    instance = _provider_from_env(monkeypatch)
    assert instance._retain_context == plugin._RETAIN_CONTEXT_DEFAULT
    assert instance._retain_indicator is True
    assert instance._recall_indicator is True
    assert instance._recall_sync is False
    instance.shutdown()


def test_flags_read_the_usual_spellings():
    for value, expected in {
        "true": True,
        "TRUE": True,
        "1": True,
        "yes": True,
        "on": True,
        "false": False,
        "0": False,
        "no": False,
        "off": False,
    }.items():
        SECRETS["HINDSIGHT_RECALL_SYNC"] = value
        assert plugin._scoped_flag("HINDSIGHT_RECALL_SYNC", not expected) is expected
    SECRETS.pop("HINDSIGHT_RECALL_SYNC")
    assert plugin._scoped_flag("HINDSIGHT_RECALL_SYNC", True) is True


def test_a_config_file_still_takes_precedence(provider):
    SECRETS.update({"HINDSIGHT_RECALL_SYNC": "true", "HINDSIGHT_RETAIN_INDICATOR": "false"})
    instance, _ = provider({"recall_sync": False})
    assert instance._recall_sync is False
    assert instance._retain_indicator is True
    instance.shutdown()


def test_the_names_and_scheduled_runs_set_in_the_environment_reach_a_profile_with_its_own_config_file(hermes_env):
    config_path = hermes_env / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"mode": "cloud", "apiKey": "test-key", "agent_name": "Old"}')
    SECRETS.update({"HINDSIGHT_COMPANY_NAME": "FIS Impianti", "HINDSIGHT_AGENT_NAME": "Sofia", "HINDSIGHT_RETAIN_CRON": "true"})
    loaded = plugin._load_config()
    for name in ("HINDSIGHT_COMPANY_NAME", "HINDSIGHT_AGENT_NAME", "HINDSIGHT_RETAIN_CRON"):
        SECRETS.pop(name)
    assert (loaded["company_name"], loaded["agent_name"]) == ("FIS Impianti", "Sofia")
