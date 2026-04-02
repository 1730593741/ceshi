"""用于测试 simplified hypervolume 感知 接口."""

from sensing.hypervolume import SimplifiedHypervolumeCalculator, compute_hypervolume


def test_simplified_hypervolume_2d() -> None:
    points = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    reference = (5.0, 5.0)

    hv = compute_hypervolume(points, reference)

    assert hv == 11.0


def test_simplified_hypervolume_empty_points() -> None:
    calculator = SimplifiedHypervolumeCalculator()
    assert calculator.compute([], (5.0, 5.0)) == 0.0
