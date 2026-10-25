"""Stub for basicsr.ops.dcn.

Modulated deformable conv is a training-time op never reached when constructing
CodeFormer/RRDBNet for export; arch_util.py imports the names at module scope,
so provide inert placeholders that raise if actually used.
"""


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "ModulatedDeformConvPack is not available in the export-only vendor "
        "basicsr package")


# arch_util.py imports both names; nothing we export instantiates them.
class ModulatedDeformConvPack:  # noqa: D101 - import-time stub
    def __init__(self, *_args, **_kwargs):
        _unavailable()


def modulated_deform_conv(*_args, **_kwargs):
    _unavailable()
