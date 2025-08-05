from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING
from graphlib import TopologicalSorter

from TurretSRC.StereoCameras.oak_d_stereo_camera import OakD
from TurretSRC.RuntimeConfigs.DepthVision.oak_d_node import OakDNode
from TurretSRC.RuntimeConfigs.DepthVision.oak_d_pipeline_component import OakDPipelineComponent


class OakDGraph:
    registered_names: ClassVar[set[str]] = set()

    def __init__(self, graph: dict[OakDNode, set[OakDNode]]):
        """
        Args:
            graph (OakDNode): A DAG of OakDNodes
        """
        self.dependencies: TopologicalSorter[OakDNode] = TopologicalSorter(graph)
        self.dependencies.prepare()

    def _register(self, name: str) -> None:
        """
        This function should register the node name.
        You should still maintain your name locally yourself.
        This is the default implementation in the interface. You probably should not change this.
        """
        if not (name and name.strip()):
            raise ValueError("Name is empty or None")

        if name in self.registered_names:
            raise ValueError("Name already registered!")
        self.registered_names.add(name)

    def get_oakd_with_dependents(self) -> OakD:
        """
        This function will handle the startup of all dependent nodes in order via DAG.
        The way that it will do this is by having all dependents declare ahead of time what they will need
        and this function will pass all parents to the dependents for them to do as they wish before
        passing along to their children which will do the same.
        This implementation is a bit of a nasty way to solve this problem, but still gives decent flexibility
        for not a crazy amount of complexity.
        Of course, most of the time, there will be maybe 3 dependents at most so this is still overkill.

        Returns:
            This function returns a reference to the OakD base object for returning to the builder.
        """
        top_level_deps: tuple[OakDNode, ...] = self.dependencies.get_ready()

        oak_d: OakD = self._resolve_top_level_dependencies(top_level_deps)

        first_iteration: bool = True
        # We will pass a shared pointer to a dict, but this is ok since we will only add to it.
        initialized_vars: dict[str, OakDPipelineComponent] = {}

        while self.dependencies.is_active():

            if first_iteration:
                node_group = top_level_deps
                first_iteration = False
            else:
                node_group: tuple[OakDNode, ...] = self.dependencies.get_ready()

            buffer: list[tuple[str, OakDPipelineComponent]] = []
            for node in node_group:
                node: OakDNode
                # Asserts non-empty name and unique name.
                self._register(node.name)
                component: OakDPipelineComponent = node.pipeline_component
                buffer.append((node.name, component))
                # initialized_vars[node.name] = component
                component.initialize_pipelines(initialized_vars)
                self.dependencies.done(node)
            initialized_vars.update(dict(buffer))

        return oak_d

    def _resolve_top_level_dependencies(self, top_level_deps: tuple[OakDNode, ...]) -> OakD:
        """
        This function will check the top level dependencies for both non-emptiness and also for the existence
        of the base OakD component.
        """
        if len(top_level_deps) == 0:
            raise ValueError("You have no nodes. You need at least one node, with that being the root being the OakD"
                             " base camera itself.")
        for node in top_level_deps:
            node: OakDNode
            if isinstance(node.pipeline_component, OakD):
                return node.pipeline_component
        raise ValueError("There is no OakD node in the top level of dependencies!")
