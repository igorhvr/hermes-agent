"""Config v53: the seeded subagent iteration cap moves 250 -> 8192."""

import hermes_yaml as yaml


def _run(tmp_path, monkeypatch, raw):
    from hermes_cli.config_migrations import run_migrations

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    run_migrations(52, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


class TestDelegationMaxIterationsDefault:
    def test_v53_raises_the_seeded_default(self, tmp_path, monkeypatch):
        """250 was the copied template value, so it follows the new default."""
        raw = _run(tmp_path, monkeypatch, {"_config_version": 52, "delegation": {"max_iterations": 250}})
        assert raw["delegation"]["max_iterations"] == 8192

    def test_v53_keeps_an_explicit_cap(self, tmp_path, monkeypatch):
        """Any other value is a real budget choice and stays put."""
        raw = _run(tmp_path, monkeypatch, {"_config_version": 52, "delegation": {"max_iterations": 400}})
        assert raw["delegation"]["max_iterations"] == 400

    def test_shipped_defaults_match_the_tool_constant(self):
        """DEFAULT_CONFIG is merged under the user file, so it — not the tool constant — is the
        runtime default for an install that sets nothing."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS

        assert DEFAULT_CONFIG["delegation"]["max_iterations"] == DEFAULT_MAX_ITERATIONS
