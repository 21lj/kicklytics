from .renderer import PitchRenderer
from .mplsoccer_renderer import MplSoccerRenderer


def create_pitch_renderer(backend="opencv"):
    backend = str(backend).strip().lower()

    if backend == "opencv":
        return PitchRenderer()

    if backend == "mplsoccer":
        return MplSoccerRenderer()

    raise ValueError(
        f"Unknown pitch renderer backend {backend!r}. "
        "Expected one of: 'opencv', 'mplsoccer'."
    )