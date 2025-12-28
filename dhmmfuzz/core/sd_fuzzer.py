"""
Main SD-Fuzz Framework:  Integrating DDPM and HMM
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

from ..ddpm.discrete_ddpm import DiscreteDDPM, ProtocolMessageTokenizer
from ..hmm.state_inference import ProtocolStateHMM
from ..monitor.target_monitor import TargetMonitor
from ...protocols.base import ProtocolInterface


@dataclass
class FuzzConfig:
    """Fuzzing configuration"""
    protocol: str = "modbus-tcp"
    target_host: str = "127.0.0.1"
    target_port: int = 502

    # DDPM parameters
    vocab_size: int = 256
    max_seq_length: int = 260
    ddpm_timesteps: int = 1000
    embedding_dim: int = 256

    # HMM parameters
    n_states: int = 15
    hmm_n_iter: int = 100

    # Fuzzing parameters
    max_iterations: int = 10000
    batch_size: int = 32
    seed_corpus_path: Optional[str] = None

    # Training parameters
    ddpm_train_epochs: int = 50
    ddpm_lr: float = 1e-4

    # Coverage parameters
    state_exploration_weight: float = 0.3
    diversity_weight: float = 0.3

    # Output
    output_dir: str = "./output"
    save_interval: int = 1000


class SDFuzzer:
    """
    State-Aware Diffusion Fuzzer
    """

    def __init__(self, config: FuzzConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SD-Fuzz")

        # Initialize components
        self.tokenizer = ProtocolMessageTokenizer(vocab_size=config.vocab_size)

        self.ddpm = DiscreteDDPM(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            embedding_dim=config.embedding_dim,
            timesteps=config.ddpm_timesteps
        ).to(self.device)

        self.hmm = ProtocolStateHMM(
            n_states=config.n_states,
            n_observations=config.vocab_size,
            n_iter=config.hmm_n_iter
        )

        self.monitor = TargetMonitor(
            host=config.target_host,
            port=config.target_port
        )

        # Statistics
        self.stats = {
            'iterations': 0,
            'crashes': 0,
            'unique_crashes': set(),
            'test_cases_sent': 0,
            'valid_responses': 0,
            'state_coverage': set(),
            'generation_diversity': []
        }

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_seed_corpus(self) -> List[bytes]:
        """Load initial seed corpus"""
        seeds = []
        if self.config.seed_corpus_path:
            corpus_path = Path(self.config.seed_corpus_path)
            if corpus_path.exists():
                for seed_file in corpus_path.glob("*. bin"):
                    with open(seed_file, 'rb') as f:
                        seeds.append(f.read())
                self.logger.info(f"Loaded {len(seeds)} seeds from corpus")

        # Add default Modbus seeds if none found
        if len(seeds) == 0:
            seeds = self._generate_default_modbus_seeds()
            self.logger.info(f"Using {len(seeds)} default Modbus seeds")

        return seeds

    def _generate_default_modbus_seeds(self) -> List[bytes]:
        """Generate default Modbus-TCP seed messages"""
        seeds = []

        # Modbus function codes
        function_codes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0F, 0x10]

        for fc in function_codes:
            # MBAP Header (7 bytes) + Function code + data
            transaction_id = b'\x00\x01'
            protocol_id = b'\x00\x00'
            length = b'\x00\x06'
            unit_id = b'\x01'

            if fc in [0x01, 0x02, 0x03, 0x04]:  # Read functions
                data = bytes([fc]) + b'\x00\x00\x00\x0A'  # Start addr + quantity
            elif fc in [0x05, 0x06]:  # Write single
                data = bytes([fc]) + b'\x00\x00\xFF\x00'  # Addr + value
            else:  # Write multiple
                data = bytes([fc]) + b'\x00\x00\x00\x02\x04\x00\x0A\x00\x0B'

            message = transaction_id + protocol_id + length + unit_id + data
            seeds.append(message)

        return seeds

    def train_ddpm(self, seed_messages: List[bytes]) -> None:
        """Train DDPM on seed corpus"""
        self.logger.info("Training DDPM...")

        # Tokenize seeds
        X = self.tokenizer.batch_encode(seed_messages, self.config.max_seq_length)
        X = X.to(self.device)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.ddpm.parameters(),
            lr=self.config.ddpm_lr
        )

        # Training loop
        self.ddpm.train()
        for epoch in range(self.config.ddpm_train_epochs):
            epoch_loss = 0
            for batch in dataloader:
                x = batch[0]
                loss = self.ddpm.training_step(x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{self.config.ddpm_train_epochs}, Loss: {avg_loss:.4f}")

        self.logger.info("DDPM training complete")

    def train_hmm(self, seed_sequences: List[List[bytes]]) -> None:
        """Train HMM on seed message sequences"""
        self.logger.info("Training HMM...")
        self.hmm.train(seed_sequences)
        self.logger.info("HMM training complete")

    def generate_test_cases(
            self,
            batch_size: int,
            target_state_sequence: Optional[List[int]] = None
    ) -> List[bytes]:
        """
        Generate test cases using DDPM guided by HMM states
        """
        self.ddpm.eval()
        with torch.no_grad():
            # Generate using DDPM
            token_tensor = self.ddpm.sample(batch_size, device=self.device)
            messages = self.tokenizer.batch_decode(token_tensor.cpu())

        # If target state sequence provided, refine generations
        if target_state_sequence:
            # Filter messages that match target states
            refined_messages = []
            for msg in messages:
                if len(refined_messages) >= batch_size:
                    break
                state = self.hmm.infer_state(msg)
                # Accept if state matches any in target sequence
                if state in target_state_sequence:
                    refined_messages.append(msg)

            # Fill remaining with original if not enough matches
            refined_messages.extend(messages[:(batch_size - len(refined_messages))])
            messages = refined_messages[: batch_size]

        return messages

    def execute_test_case(self, message: bytes) -> Dict:
        """
        Execute single test case and collect feedback
        """
        result = self.monitor.send_and_monitor(message)

        # Update statistics
        self.stats['test_cases_sent'] += 1
        if result['response_received']:
            self.stats['valid_responses'] += 1

        if result['crash_detected']:
            self.stats['crashes'] += 1
            crash_sig = result.get('crash_signature', '')
            self.stats['unique_crashes'].add(crash_sig)

            # Save crashing input
            crash_file = Path(self.config.output_dir) / f"crash_{len(self.stats['unique_crashes'])}.bin"
            with open(crash_file, 'wb') as f:
                f.write(message)
            self.logger.warning(f"Crash detected!  Saved to {crash_file}")

        # Infer state
        state = self.hmm.infer_state(message)
        self.stats['state_coverage'].add(state)
        result['inferred_state'] = state

        return result

    def run(self, iterations: Optional[int] = None) -> Dict:
        """
        Main fuzzing loop
        """
        if iterations is None:
            iterations = self.config.max_iterations

        self.logger.info("Starting SD-Fuzz...")
        self.logger.info(f"Target:  {self.config.target_host}:{self.config.target_port}")

        # Load and prepare seeds
        seeds = self.load_seed_corpus()

        # Create seed sequences for HMM training
        seed_sequences = [seeds[i:i + 5] for i in range(0, len(seeds), 5)]
        if len(seed_sequences[-1]) < 2:
            seed_sequences = seed_sequences[:-1]

        # Train models
        self.train_ddpm(seeds)
        self.train_hmm(seed_sequences)

        # Fuzzing loop
        self.logger.info(f"Starting fuzzing for {iterations} iterations...")

        with tqdm(total=iterations, desc="Fuzzing") as pbar:
            while self.stats['iterations'] < iterations:
                # Decide fuzzing strategy
                rand = np.random.random()

                if rand < self.config.state_exploration_weight:
                    # State-guided generation
                    unexplored = self.hmm.get_unexplored_states()
                    if unexplored:
                        target_states = np.random.choice(unexplored, size=min(5, len(unexplored)),
                                                         replace=False).tolist()
                    else:
                        # Generate sequence targeting deep states
                        target_states = self.hmm.generate_state_sequence(length=5)

                    messages = self.generate_test_cases(
                        self.config.batch_size,
                        target_state_sequence=target_states
                    )
                else:
                    # Pure DDPM generation for diversity
                    messages = self.generate_test_cases(self.config.batch_size)

                # Execute test cases
                for msg in messages:
                    result = self.execute_test_case(msg)
                    self.stats['iterations'] += 1
                    pbar.update(1)

                    if self.stats['iterations'] >= iterations:
                        break

                # Periodic reporting
                if self.stats['iterations'] % self.config.save_interval == 0:
                    self._report_statistics()
                    self._save_checkpoint()

        self.logger.info("Fuzzing complete!")
        self._report_statistics()

        return self.stats

    def _report_statistics(self) -> None:
        """Report fuzzing statistics"""
        state_cov = len(self.stats['state_coverage']) / self.config.n_states * 100

        self.logger.info("=" * 60)
        self.logger.info(f"Iterations: {self.stats['iterations']}")
        self.logger.info(f"Test cases sent: {self.stats['test_cases_sent']}")
        self.logger.info(f"Valid responses: {self.stats['valid_responses']}")
        self.logger.info(f"Crashes detected: {self.stats['crashes']}")
        self.logger.info(f"Unique crashes: {len(self.stats['unique_crashes'])}")
        self.logger.info(f"State coverage: {state_cov:.2f}%")
        self.logger.info("=" * 60)

    def _save_checkpoint(self) -> None:
        """Save fuzzer checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save DDPM
        torch.save(
            self.ddpm.state_dict(),
            checkpoint_dir / f"ddpm_{self.stats['iterations']}.pt"
        )

        # Save HMM
        self.hmm.save(str(checkpoint_dir / f"hmm_{self.stats['iterations']}.pkl"))

        self.logger.info(f"Checkpoint saved at iteration {self.stats['iterations']}")