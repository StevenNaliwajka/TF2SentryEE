from src.CV.analyzer import Analyzer
from src.RuntimeConfigs.analyzer_configurable import AnalyzerConfigurable
from TurretSRC.CVImplementations.default_analyzer import DefaultAnalyzer


class DoubleDetector(AnalyzerConfigurable):
    def get_analyzer(self) -> Analyzer:
        return DefaultAnalyzer()
