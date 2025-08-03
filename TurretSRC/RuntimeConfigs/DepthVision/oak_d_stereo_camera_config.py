from __future__ import annotations
from src.RuntimeConfigs.depth_vision_configurable import DepthVisionConfigurable
from TurretSRC.StereoCameras.oak_d_stereo_camera import OakD
from TurretSRC.RuntimeConfigs.DepthVision.oak_d_node import OakDNode
from oak_d_graph import OakDGraph


class OakDBuilder(DepthVisionConfigurable):

    def __init__(self, graph: dict[OakDNode, tuple[OakDNode]]) -> None:
        """
        Your dependency graph here should be a graph of OakDNodes
        You can use whatever naming convention you want, but you will have to edit the modules that you are using
        to reflect the modules that you named.
        For example, you might have: (The tuples represent a OakDNode)
        {
            ("depth_vision", depth_vision_ref): {("oakd_base", oak_d_ref)},
            ("tracker", tracker_ref): {("oakd_base,oak_d_ref), ("depth_vision", depth_vision_ref)}
        }
        """
        self.oak_d_graph: OakDGraph = OakDGraph(graph)

    def get_depthvision(self) -> OakD:
        """
        This function will return an instance of the OakD with all of its dependencies also initialized.
        Those dependencies references should be managed by the caller.
        Returns:
            A reference to the base OakD.
        """
        return self.oak_d_graph.get_oakd_with_dependents()

