from typing import TYPE_CHECKING
from dataclasses import dataclass
from TurretSRC.RuntimeConfigs.DepthVision.oak_d_pipeline_component import OakDPipelineComponent


# if TYPE_CHECKING:
#     from TurretSRC.RuntimeConfigs.DepthVision.oak_d_pipeline_component import OakDPipelineComponent


@dataclass(frozen=True)
class OakDNode:
    """
    Args:
        pipeline_component (OakDPipelineComponent): The pipeline component that this node contains
        name (str): The name of the node for future reference across dependencies.
    """
    pipeline_component: OakDPipelineComponent
    name: str
