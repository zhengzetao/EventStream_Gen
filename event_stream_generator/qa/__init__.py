"""Template QA generation from simulator ground truth."""

from .causal_qa import generate_causal_qa
from .propagation_qa import generate_propagation_qa
from .temporal_qa import generate_temporal_qa


def generate_all_qa(stream):
    return [
        *generate_causal_qa(stream),
        *generate_temporal_qa(stream),
        *generate_propagation_qa(stream),
    ]
