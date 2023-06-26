from pytest_check import check


def assert_all_scenarios_0(test_scenarios: dict) -> None:
    for scenario_name, result_dict in test_scenarios.items():
        for name, result in result_dict.items():
            with check:
                assert result == (
                    0,
                    0,
                ), f"non-zero result for {scenario_name}:{name} :: {result}"


def assert_all_scenarios_above_0(test_scenarios: dict) -> None:
    for scenario_name, result_dict in test_scenarios.items():
        for name, result in result_dict.items():
            with check:
                assert (
                    result[0] > 0 and result[1] > 0
                ), f"result <= 0 for {scenario_name}:{name} :: {result}"
