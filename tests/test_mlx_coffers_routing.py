import mlx_coffers
from mlx_coffers import CofferDomain, CofferPool, LayerRouter, build_coffer_pool


def test_domain_metadata_maps_expected_brain_regions_and_streams():
    assert CofferDomain.PREFRONTAL.brain_region == "Prefrontal Cortex"
    assert CofferDomain.PREFRONTAL.apple_silicon_stream == "Metal GPU"
    assert CofferDomain.LEFT_HEMI.cognitive_function == "Language / Logic / Sequential"
    assert CofferDomain.LEFT_HEMI.apple_silicon_stream == "CPU"
    assert CofferDomain.TEMPORAL.brain_region == "Temporal Lobe"
    assert CofferDomain.TEMPORAL.apple_silicon_stream == "CPU"


def test_layer_router_keyword_rules_take_precedence_over_layer_index():
    router = LayerRouter(total_layers=12)

    assert router.classify("model.layers.11.self_attn.q_proj", layer_index=11) == CofferDomain.LEFT_HEMI
    assert router.classify("model.layers.0.mlp.gate_proj", layer_index=0) == CofferDomain.RIGHT_HEMI
    assert router.classify("model.embed_tokens", layer_index=4) == CofferDomain.PREFRONTAL
    assert router.classify("past_key_values.0.key", layer_index=1) == CofferDomain.TEMPORAL


def test_layer_router_index_fallbacks_cover_early_mid_late_and_missing_index():
    router = LayerRouter(total_layers=9)

    assert router.classify("custom.layer.early", layer_index=0) == CofferDomain.LEFT_HEMI
    assert router.classify("custom.layer.mid", layer_index=4) == CofferDomain.RIGHT_HEMI
    assert router.classify("custom.layer.late", layer_index=8) == CofferDomain.PREFRONTAL
    assert router.classify("custom.layer.default") == CofferDomain.LEFT_HEMI


def test_layer_router_caches_classification_for_repeated_layer_names():
    router = LayerRouter(total_layers=10)

    first = router.classify("custom.layer.memoized", layer_index=1)
    second = router.classify("custom.layer.memoized", layer_index=9)

    assert first == CofferDomain.LEFT_HEMI
    assert second == first


def test_routing_table_assigns_unknown_layers_by_position():
    router = LayerRouter(total_layers=3)

    table = router.routing_table(["custom.0", "custom.1", "custom.2"])

    assert table == {
        "custom.0": CofferDomain.LEFT_HEMI,
        "custom.1": CofferDomain.RIGHT_HEMI,
        "custom.2": CofferDomain.PREFRONTAL,
    }


def test_coffer_pool_stub_initializes_all_domains_and_tracks_route_hits(monkeypatch):
    monkeypatch.setattr(mlx_coffers, "HAS_MLX", False)
    pool = CofferPool(verbose=False)

    left = pool.route("model.layers.0.self_attn.q_proj")
    right = pool.route("model.layers.0.mlp.down_proj")
    temporal = pool.get(CofferDomain.TEMPORAL)

    assert left.domain == CofferDomain.LEFT_HEMI
    assert right.domain == CofferDomain.RIGHT_HEMI
    assert temporal.domain == CofferDomain.TEMPORAL
    assert all(coffer.device is None for coffer in pool._coffers.values())
    assert left.hits == 1
    assert right.hits == 1


def test_build_coffer_pool_uses_stub_mode_when_mlx_is_unavailable(monkeypatch):
    monkeypatch.setattr(mlx_coffers, "HAS_MLX", False)

    pool = build_coffer_pool(verbose=False)

    assert set(pool._coffers) == set(CofferDomain)
    assert pool.get(CofferDomain.PREFRONTAL).device is None
    stats = pool.stats()
    assert "PREFRONTAL" in stats
    assert "TOTAL" in stats
