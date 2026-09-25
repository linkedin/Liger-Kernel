import re

from pathlib import Path

MOE_DIR = Path(__file__).resolve().parents[1] / "csrc" / "core" / "src" / "moe"

DISJOINT_HEADERS = (
    "cta_barrier.cuh",
    "math.cuh",
    "mlp.cuh",
    "mlp1_fused.cuh",
    "mlp1_fused_act.cuh",
    "mlp2_fused.cuh",
    "mlp2_t.cuh",
    "mlp2_t_fused.cuh",
    "mlp3.cuh",
    "mlp4.cuh",
    "mlp5.cuh",
    "mlp5_fused.cuh",
    "mlp_bwd.cuh",
    "mlp_comms.cuh",
    "mlp_comms_bwd.cuh",
    "models.cuh",
    "moe.cuh",
    "moe_bwd.cuh",
    "moe_comm_config.cuh",
    "moe_symm_config.cuh",
    "silu_bwd_fused.cuh",
    "tile_iterator.cuh",
    "tile_iterator_bwd.cuh",
    "tma_copy_atom.cuh",
    "tmem_load_op.cuh",
)


def _architecture_name(name: str, architecture: str) -> str:
    path = Path(name)
    return f"{path.stem}_{architecture}{path.suffix}"


def test_moe_gemm_include_graph_is_architecture_local():
    split_names = set(DISJOINT_HEADERS)
    split_names.update(("moe.cu", "moe_bwd.cu"))

    for architecture, opposite in (("sm90", "sm100"), ("sm100", "sm90")):
        local_files = [_architecture_name(name, architecture) for name in DISJOINT_HEADERS]
        local_files.extend(
            (
                f"moe_{architecture}.inc",
                f"moe_bwd_{architecture}.inc",
                f"moe_fwd_bwd_tune_configs_{architecture}.hpp",
                f"moe_fwd_bwd_tuning_configs_{architecture}.cuh",
            )
        )
        for local_name in local_files:
            text = (MOE_DIR / local_name).read_text()
            includes = re.findall(r'#include\s+"([^"]+)"', text)
            assert not any(f"_{opposite}." in include for include in includes), local_name
            assert split_names.isdisjoint(includes), local_name


def test_moe_entrypoints_select_one_architecture():
    for name in DISJOINT_HEADERS:
        text = (MOE_DIR / name).read_text()
        assert f'#include "{_architecture_name(name, "sm90")}"' in text
        assert f'#include "{_architecture_name(name, "sm100")}"' in text

    for name in ("moe.cu", "moe_bwd.cu"):
        stem = Path(name).stem
        text = (MOE_DIR / name).read_text()
        assert f'#include "{stem}_sm90.inc"' in text
        assert f'#include "{stem}_sm100.inc"' in text


def test_moe_communication_geometry_matches_mainline():
    for architecture in ("sm90", "sm100"):
        text = (MOE_DIR / f"moe_comm_config_{architecture}.cuh").read_text()
        for compute in (90, 100):
            geometry = re.search(
                rf"struct MoeCommGeometry<{compute}> \{{(?P<body>.*?)\n\}};",
                text,
                re.DOTALL,
            )
            assert geometry is not None
            assert "static constexpr int TileM = 128;" in geometry["body"]
            assert "static constexpr int NC = 4;" in geometry["body"]
