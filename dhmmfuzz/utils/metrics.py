"""
Fuzzing Metrics:   Coverage, Diversity, and Performance
"""
import numpy as np
from typing import List, Set, Dict
from collections import defaultdict
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


class CoverageMetrics:
    """
    Track various coverage metrics
    """

    def __init__(self):
        self.edge_coverage: Set[tuple] = set()
        self.state_coverage: Set[int] = set()
        self.function_coverage: Set[int] = set()
        self.path_coverage: Set[str] = set()

        self.coverage_over_time: List[int] = []
        self.timestamps: List[float] = []

    def update_edge_coverage(self, edge: tuple) -> bool:
        """
        Update edge coverage (state transitions)
        Returns True if new edge discovered
        """
        if edge not in self.edge_coverage:
            self.edge_coverage.add(edge)
            return True
        return False

    def update_state_coverage(self, state: int) -> bool:
        """
        Update state coverage
        Returns True if new state discovered
        """
        if state not in self.state_coverage:
            self.state_coverage.add(state)
            return True
        return False

    def update_function_coverage(self, function_code: int) -> bool:
        """
        Update function code coverage
        """
        if function_code not in self.function_coverage:
            self.function_coverage.add(function_code)
            return True
        return False

    def update_path_coverage(self, path: List[int]) -> bool:
        """
        Update path coverage (sequence of states)
        """
        path_hash = hashlib.md5(str(path).encode()).hexdigest()
        if path_hash not in self.path_coverage:
            self.path_coverage.add(path_hash)
            return True
        return False

    def get_edge_coverage_rate(self, total_edges: int) -> float:
        """
        Calculate edge coverage percentage
        """
        if total_edges == 0:
            return 0.0
        return len(self.edge_coverage) / total_edges

    def get_state_coverage_rate(self, total_states: int) -> float:
        """
        Calculate state coverage percentage
        """
        if total_states == 0:
            return 0.0
        return len(self.state_coverage) / total_states

    def snapshot(self, timestamp: float) -> None:
        """
        Take a coverage snapshot for time-series analysis
        """
        self.timestamps.append(timestamp)
        self.coverage_over_time.append(len(self.state_coverage))

    def get_coverage_growth_rate(self) -> float:
        """
        Calculate average coverage growth rate
        """
        if len(self.coverage_over_time) < 2:
            return 0.0

        diffs = np.diff(self.coverage_over_time)
        return np.mean(diffs)

    def export_metrics(self) -> Dict:
        """
        Export all metrics as dictionary
        """
        return {
            'edge_coverage': len(self.edge_coverage),
            'state_coverage': len(self.state_coverage),
            'function_coverage': len(self.function_coverage),
            'path_coverage': len(self.path_coverage),
            'coverage_over_time': self.coverage_over_time,
            'timestamps': self.timestamps
        }


class DiversityMetrics:
    """
    Measure test case diversity
    """

    def __init__(self):
        self.test_cases: List[bytes] = []
        self.test_case_features: List[np.ndarray] = []

    def extract_features(self, message: bytes) -> np.ndarray:
        """
        Extract features from message for diversity calculation
        """
        features = []

        # Length
        features.append(len(message))

        # Byte statistics
        if len(message) > 0:
            byte_array = np.array(list(message))
            features.extend([
                np.mean(byte_array),
                np.std(byte_array),
                np.min(byte_array),
                np.max(byte_array),
                np.median(byte_array)
            ])

            # Entropy
            _, counts = np.unique(byte_array, return_counts=True)
            probs = counts / len(byte_array)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features.append(entropy)

            # Byte distribution (histogram)
            hist, _ = np.histogram(byte_array, bins=16, range=(0, 256))
            features.extend(hist / len(byte_array))
        else:
            features.extend([0] * 6)
            features.extend([0] * 16)

        return np.array(features)

    def add_test_case(self, message: bytes) -> None:
        """
        Add test case to diversity tracking
        """
        self.test_cases.append(message)
        features = self.extract_features(message)
        self.test_case_features.append(features)

    def calculate_pairwise_diversity(self) -> float:
        """
        Calculate average pairwise distance (diversity)
        """
        if len(self.test_case_features) < 2:
            return 0.0

        feature_matrix = np.array(self.test_case_features)
        distances = pdist(feature_matrix, metric='euclidean')

        return np.mean(distances)

    def calculate_coverage_diversity(self) -> float:
        """
        Calculate diversity based on feature space coverage
        """
        if len(self.test_case_features) < 2:
            return 0.0

        feature_matrix = np.array(self.test_case_features)

        # Calculate covariance determinant (volume in feature space)
        try:
            cov = np.cov(feature_matrix.T)
            diversity = np.log(np.linalg.det(cov) + 1e-10)
            return diversity
        except:
            return 0.0

    def calculate_entropy_diversity(self) -> float:
        """
        Calculate Shannon entropy of test case distribution
        """
        if len(self.test_cases) == 0:
            return 0.0

        # Hash-based entropy
        hashes = [hashlib.md5(tc).hexdigest()[:8] for tc in self.test_cases]
        unique_hashes = set(hashes)

        # Simple uniqueness ratio
        return len(unique_hashes) / len(self.test_cases)

    def calculate_novelty_score(self, message: bytes, k: int = 5) -> float:
        """
        Calculate novelty score of a new message
        (average distance to k-nearest neighbors)
        """
        if len(self.test_case_features) == 0:
            return 1.0

        features = self.extract_features(message)
        feature_matrix = np.array(self.test_case_features)

        # Calculate distances to all existing test cases
        distances = np.linalg.norm(feature_matrix - features, axis=1)

        # Average distance to k-nearest neighbors
        k = min(k, len(distances))
        k_nearest = np.partition(distances, k)[:k]

        return np.mean(k_nearest)

    def get_diversity_report(self) -> Dict:
        """
        Generate comprehensive diversity report
        """
        return {
            'total_test_cases': len(self.test_cases),
            'pairwise_diversity': self.calculate_pairwise_diversity(),
            'coverage_diversity': self.calculate_coverage_diversity(),
            'entropy_diversity': self.calculate_entropy_diversity(),
            'unique_test_cases': len(set(self.test_cases))
        }


class PerformanceMetrics:
    """
    Track fuzzer performance metrics
    """

    def __init__(self):
        self.executions = 0
        self.execution_times: List[float] = []
        self.generation_times: List[float] = []
        self.crashes = 0
        self.unique_crashes = set()
        self.start_time = None

    def record_execution(self, execution_time: float) -> None:
        """Record single execution time"""
        self.executions += 1
        self.execution_times.append(execution_time)

    def record_generation(self, generation_time: float) -> None:
        """Record test case generation time"""
        self.generation_times.append(generation_time)

    def record_crash(self, crash_signature: str) -> None:
        """Record crash"""
        self.crashes += 1
        self.unique_crashes.add(crash_signature)

    def get_executions_per_second(self) -> float:
        """Calculate average executions per second"""
        if len(self.execution_times) == 0:
            return 0.0

        total_time = sum(self.execution_times)
        if total_time == 0:
            return 0.0

        return self.executions / total_time

    def get_average_execution_time(self) -> float:
        """Get average execution time"""
        if len(self.execution_times) == 0:
            return 0.0
        return np.mean(self.execution_times)

    def get_average_generation_time(self) -> float:
        """Get average generation time"""
        if len(self.generation_times) == 0:
            return 0.0
        return np.mean(self.generation_times)

    def get_crash_rate(self) -> float:
        """Calculate crash rate"""
        if self.executions == 0:
            return 0.0
        return self.crashes / self.executions

    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            'total_executions': self.executions,
            'executions_per_second': self.get_executions_per_second(),
            'avg_execution_time': self.get_average_execution_time(),
            'avg_generation_time': self.get_average_generation_time(),
            'total_crashes': self.crashes,
            'unique_crashes': len(self.unique_crashes),
            'crash_rate': self.get_crash_rate()
        }


class RecognitionMetrics:
    """
    Track test case recognition rate (validity)
    """

    def __init__(self):
        self.total_generated = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.validation_results: List[bool] = []

    def record_validation(self, is_valid: bool) -> None:
        """Record validation result"""
        self.total_generated += 1
        self.validation_results.append(is_valid)

        if is_valid:
            self.valid_count += 1
        else:
            self.invalid_count += 1

    def get_recognition_rate(self) -> float:
        """Calculate recognition rate (percentage valid)"""
        if self.total_generated == 0:
            return 0.0
        return self.valid_count / self.total_generated

    def get_windowed_recognition_rate(self, window_size: int = 100) -> float:
        """Calculate recognition rate over recent window"""
        if len(self.validation_results) < window_size:
            window_size = len(self.validation_results)

        if window_size == 0:
            return 0.0

        recent = self.validation_results[-window_size:]
        return sum(recent) / len(recent)

    def get_recognition_report(self) -> Dict:
        """Generate recognition report"""
        return {
            'total_generated': self.total_generated,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'recognition_rate': self.get_recognition_rate(),
            'windowed_rate': self.get_windowed_recognition_rate()
        }