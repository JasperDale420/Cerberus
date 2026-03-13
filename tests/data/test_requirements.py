from src.data.requirements import DataRequirements, aggregate_requirements


def test_default_requirements():
    r = DataRequirements()
    assert r.streams == ["bars"]
    assert r.on_scan == []
    assert r.indicators == []


def test_aggregate_unions_streams():
    r1 = DataRequirements(streams=["bars"], on_scan=["flow"])
    r2 = DataRequirements(streams=["bars", "quotes"], on_scan=["gex"])
    agg = aggregate_requirements([r1, r2])
    assert set(agg.streams) == {"bars", "quotes"}
    assert set(agg.on_scan) == {"flow", "gex"}


def test_aggregate_empty():
    agg = aggregate_requirements([])
    assert agg.streams == []
    assert agg.on_scan == []
