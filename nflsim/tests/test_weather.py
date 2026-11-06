from nflsim.features.weather import WEATHER_COLUMNS, weather_features


def test_weather_features_scale_outdoor_inputs_and_gate_indoors():
    outdoor = weather_features({"roof": "outdoors", "temp": 40, "wind": 15})
    assert outdoor == {
        "weather_outdoor": 1.0,
        "weather_temp_delta": -1.0,
        "weather_wind_speed": 1.0,
        "weather_missing": 0.0,
    }
    assert weather_features({"roof": "dome", "temp": 10, "wind": 60}) == dict.fromkeys(WEATHER_COLUMNS, 0.0)


def test_missing_or_unknown_weather_is_not_silently_treated_as_calm():
    assert weather_features({"roof": "open", "temp": None, "wind": None}) == {
        "weather_outdoor": 1.0,
        "weather_temp_delta": 0.0,
        "weather_wind_speed": 0.0,
        "weather_missing": 1.0,
    }
    assert weather_features({"roof": None, "temp": 55, "wind": 12})["weather_missing"] == 1.0


def test_every_active_model_requires_weather_context():
    from nflsim.configuration import context_columns
    from nflsim.engine.simulation_runtime import SimulationRuntime

    for model in SimulationRuntime.models:
        assert set(WEATHER_COLUMNS) <= set(context_columns(model))