from gtb.presets import get_preset, preset_plan


def test_common_5_targets_are_five_piece_two_vs_one():
    preset = get_preset("common-5")
    for spec in preset.target_specs():
        assert spec.piece_count == 5
        extras = (len(spec.white) - 1, len(spec.black) - 1)
        assert sorted(extras) == [1, 2]


def test_common_5_dependency_closure_includes_core_and_five_piece():
    plan = preset_plan("common-5")
    assert any(spec.piece_count <= 4 for spec in plan)
    assert any(spec.piece_count == 5 for spec in plan)
    signatures = {spec.signature for spec in plan}
    assert "KRPvKR" in signatures
    assert "KPPvKP" in signatures
