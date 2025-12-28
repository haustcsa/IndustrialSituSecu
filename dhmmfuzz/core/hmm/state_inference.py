"""
Hidden Markov Model for Protocol State Inference
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from hmmlearn import hmm
from collections import defaultdict
import pickle


class ProtocolStateHMM:
    """
    HMM for unsupervised inference of protocol state transitions
    """

    def __init__(
            self,
            n_states: int = 10,
            n_observations: int = 256,
            algorithm: str = "viterbi",
            n_iter: int = 100
    ):
        self.n_states = n_states
        self.n_observations = n_observations
        self.algorithm = algorithm

        # Initialize Gaussian HMM for continuous observations
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            algorithm=algorithm
        )

        # State tracking
        self.state_sequences = []
        self.observation_sequences = []
        self.state_coverage = defaultdict(int)
        self.transition_counts = defaultdict(int)

    def extract_features(self, message: bytes) -> np.ndarray:
        """
        Extract features from protocol message
        """
        features = []

        # Basic byte statistics
        if len(message) > 0:
            features.extend([
                len(message),  # Length
                np.mean(list(message)),  # Mean byte value
                np.std(list(message)),  # Std of byte values
                np.min(list(message)),  # Min byte
                np.max(list(message)),  # Max byte
                sum(1 for b in message if b == 0),  # Null byte count
                len(set(message)) / max(len(message), 1)  # Unique byte ratio
            ])
        else:
            features.extend([0] * 7)

        # Protocol-specific features (Modbus-TCP example)
        if len(message) >= 8:
            features.extend([
                int.from_bytes(message[0:2], 'big'),  # Transaction ID
                int.from_bytes(message[2:4], 'big'),  # Protocol ID
                int.from_bytes(message[4:6], 'big'),  # Length
                message[6],  # Unit ID
                message[7] if len(message) > 7 else 0  # Function code
            ])
        else:
            features.extend([0] * 5)

        return np.array(features, dtype=np.float64)

    def train(self, message_sequences: List[List[bytes]]) -> None:
        """
        Train HMM on observed message sequences
        """
        print(f"Training HMM with {len(message_sequences)} sequences...")

        # Extract features from all sequences
        feature_sequences = []
        for seq in message_sequences:
            features = [self.extract_features(msg) for msg in seq]
            feature_sequences.append(np.array(features))

        # Concatenate sequences for training
        lengths = [len(seq) for seq in feature_sequences]
        X = np.concatenate(feature_sequences)

        # Train HMM
        self.model.fit(X, lengths=lengths)

        print(f"HMM training complete.  Converged: {self.model.monitor_.converged}")

    def infer_state(self, message: bytes) -> int:
        """
        Infer hidden state from a single message
        """
        features = self.extract_features(message).reshape(1, -1)
        state = self.model.predict(features)[0]
        self.state_coverage[state] += 1
        return state

    def infer_state_sequence(self, messages: List[bytes]) -> List[int]:
        """
        Infer state sequence from message sequence
        """
        features = np.array([self.extract_features(msg) for msg in messages])
        states = self.model.predict(features)

        # Update transition counts
        for i in range(len(states) - 1):
            self.transition_counts[(states[i], states[i + 1])] += 1

        # Update coverage
        for state in states:
            self.state_coverage[state] += 1

        self.state_sequences.append(states)
        return states.tolist()

    def get_next_state_distribution(self, current_state: int) -> np.ndarray:
        """
        Get probability distribution over next states
        """
        # Use learned transition matrix
        return self.model.transmat_[current_state]

    def sample_next_state(self, current_state: int) -> int:
        """
        Sample next state given current state
        """
        probs = self.get_next_state_distribution(current_state)
        return np.random.choice(self.n_states, p=probs)

    def get_unexplored_states(self) -> List[int]:
        """
        Return states with low coverage
        """
        all_states = set(range(self.n_states))
        explored = set(self.state_coverage.keys())
        return list(all_states - explored)

    def get_state_coverage_rate(self) -> float:
        """
        Calculate state coverage percentage
        """
        explored = len(self.state_coverage)
        return explored / self.n_states

    def generate_state_sequence(
            self,
            start_state: Optional[int] = None,
            length: int = 10
    ) -> List[int]:
        """
        Generate a state sequence for guided fuzzing
        """
        if start_state is None:
            # Sample from initial state distribution
            start_state = np.random.choice(self.n_states, p=self.model.startprob_)

        sequence = [start_state]
        current = start_state

        for _ in range(length - 1):
            next_state = self.sample_next_state(current)
            sequence.append(next_state)
            current = next_state

        return sequence

    def save(self, filepath: str) -> None:
        """Save trained HMM"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'state_coverage': dict(self.state_coverage),
                'transition_counts': dict(self.transition_counts),
                'n_states': self.n_states
            }, f)

    def load(self, filepath: str) -> None:
        """Load trained HMM"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.state_coverage = defaultdict(int, data['state_coverage'])
            self.transition_counts = defaultdict(int, data['transition_counts'])
            self.n_states = data['n_states']