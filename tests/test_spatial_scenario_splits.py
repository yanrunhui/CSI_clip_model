from __future__ import annotations

from scripts.make_spatial_scenario_splits import (
    parse_group_id,
    spatial_block_keys,
    split_by_groups,
    split_random_stratified,
)


def test_parse_group_id() -> None:
    assert parse_group_id("D2Los_Data-map_12-source_7-rx_345") == (12, 7, 345)


def test_random_split_is_complete_disjoint_and_stratified() -> None:
    statuses = ["los"] * 50 + ["nlos"] * 50
    train, test = split_random_stratified(statuses, 0.2, 23)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(100))
    assert len(test) == 20
    assert sum(statuses[index] == "los" for index in test) == 10


def test_spatial_group_split_never_crosses_blocks() -> None:
    parsed = [(map_id, source, rx) for map_id in range(2) for source in range(2) for rx in range(16)]
    coordinates = [(float(rx % 4) * 10.0, float(rx // 4) * 10.0, 1.5) for _, _, rx in parsed]
    keys = spatial_block_keys(parsed, coordinates, block_size_m=20.0)
    train, test = split_by_groups(keys, 0.25, 19, per_map=True)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(len(parsed)))
    assert {keys[index] for index in train}.isdisjoint(
        {keys[index] for index in test}
    )
    assert {parsed[index][0] for index in train} == {0, 1}
    assert {parsed[index][0] for index in test} == {0, 1}


def test_scenario_split_never_crosses_maps() -> None:
    keys = [(map_id,) for map_id in range(10) for _ in range(10)]
    strata = ["los" if map_id % 2 == 0 else "nlos" for map_id in range(10) for _ in range(10)]
    train, test = split_by_groups(keys, 0.2, 11, per_map=False, strata=strata)
    assert {keys[index] for index in train}.isdisjoint(
        {keys[index] for index in test}
    )
    assert len(test) == 20
    assert sum(strata[index] == "los" for index in test) == 10
