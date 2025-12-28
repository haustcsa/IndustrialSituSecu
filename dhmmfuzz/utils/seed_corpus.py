"""
Seed Corpus Management
"""
import os
from pathlib import Path
from typing import List, Dict, Set
import hashlib
import json
import logging


class SeedCorpusManager:
    """
    Manage seed corpus for fuzzing
    """

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.seeds: List[bytes] = []
        self.seed_hashes: Set[str] = set()
        self.seed_metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger("SeedCorpus")

    def add_seed(self, data: bytes, metadata: Dict = None) -> bool:
        """
        Add a seed to corpus (avoid duplicates)
        """
        seed_hash = hashlib.sha256(data).hexdigest()

        if seed_hash in self.seed_hashes:
            return False

        self.seeds.append(data)
        self.seed_hashes.add(seed_hash)

        if metadata:
            self.seed_metadata[seed_hash] = metadata

        # Save to disk
        seed_file = self.corpus_dir / f"{seed_hash[: 16]}.bin"
        with open(seed_file, 'wb') as f:
            f.write(data)

        return True

    def load_corpus(self) -> List[bytes]:
        """
        Load all seeds from corpus directory
        """
        self.seeds = []
        self.seed_hashes = set()

        if not self.corpus_dir.exists():
            self.logger.warning(f"Corpus directory {self.corpus_dir} does not exist")
            return []

        for seed_file in self.corpus_dir.glob("*.bin"):
            try:
                with open(seed_file, 'rb') as f:
                    data = f.read()
                    seed_hash = hashlib.sha256(data).hexdigest()

                    if seed_hash not in self.seed_hashes:
                        self.seeds.append(data)
                        self.seed_hashes.add(seed_hash)
            except Exception as e:
                self.logger.error(f"Error loading seed {seed_file}: {e}")

        self.logger.info(f"Loaded {len(self.seeds)} seeds from corpus")
        return self.seeds

    def get_interesting_seeds(self, top_k: int = 10) -> List[bytes]:
        """
        Get most interesting seeds based on metadata
        """
        # Sort by coverage or other metrics if available
        scored_seeds = []
        for seed in self.seeds:
            seed_hash = hashlib.sha256(seed).hexdigest()
            metadata = self.seed_metadata.get(seed_hash, {})
            score = metadata.get('coverage', 0) + metadata.get('unique_paths', 0)
            scored_seeds.append((score, seed))

        scored_seeds.sort(reverse=True)
        return [seed for _, seed in scored_seeds[:top_k]]

    def minimize_corpus(self) -> None:
        """
        Minimize corpus by removing redundant seeds
        """
        # Simple deduplication by hash
        unique_seeds = []
        seen_hashes = set()

        for seed in self.seeds:
            seed_hash = hashlib.sha256(seed).hexdigest()
            if seed_hash not in seen_hashes:
                unique_seeds.append(seed)
                seen_hashes.add(seed_hash)

        removed = len(self.seeds) - len(unique_seeds)
        self.seeds = unique_seeds
        self.seed_hashes = seen_hashes

        self.logger.info(f"Corpus minimized:  removed {removed} duplicate seeds")

    def export_metadata(self, output_file: str) -> None:
        """
        Export seed metadata to JSON
        """
        with open(output_file, 'w') as f:
            json.dump(self.seed_metadata, f, indent=2)

    def import_from_pcap(self, pcap_file: str, protocol: str = "modbus") -> int:
        """
        Import seeds from network capture (PCAP)
        """
        try:
            from scapy.all import rdpcap, TCP

            packets = rdpcap(pcap_file)
            imported = 0

            for pkt in packets:
                if TCP in pkt and pkt[TCP].payload:
                    payload = bytes(pkt[TCP].payload)
                    if len(payload) > 0:
                        if self.add_seed(payload, {'source': 'pcap', 'protocol': protocol}):
                            imported += 1

            self.logger.info(f"Imported {imported} seeds from {pcap_file}")
            return imported

        except ImportError:
            self.logger.error("scapy not installed.  Cannot import from PCAP")
            return 0
        except Exception as e:
            self.logger.error(f"Error importing from PCAP: {e}")
            return 0


class SeedScheduler:
    """
    Schedule seed selection during fuzzing
    """

    def __init__(self, seeds: List[bytes], strategy: str = "weighted"):
        self.seeds = seeds
        self.strategy = strategy
        self.seed_energy = {i: 1.0 for i in range(len(seeds))}
        self.seed_executions = {i: 0 for i in range(len(seeds))}

    def select_seed(self) -> bytes:
        """
        Select next seed based on strategy
        """
        import random

        if self.strategy == "uniform":
            return random.choice(self.seeds)

        elif self.strategy == "weighted":
            # Select based on energy
            weights = [self.seed_energy[i] for i in range(len(self.seeds))]
            total = sum(weights)
            if total == 0:
                return random.choice(self.seeds)

            weights = [w / total for w in weights]
            idx = random.choices(range(len(self.seeds)), weights=weights)[0]
            self.seed_executions[idx] += 1
            return self.seeds[idx]

        elif self.strategy == "round_robin":
            # Simple round-robin
            min_exec = min(self.seed_executions.values())
            candidates = [i for i, count in self.seed_executions.items() if count == min_exec]
            idx = random.choice(candidates)
            self.seed_executions[idx] += 1
            return self.seeds[idx]

        else:
            return random.choice(self.seeds)

    def update_energy(self, seed_idx: int, coverage_gain: float) -> None:
        """
        Update seed energy based on coverage gain
        """
        self.seed_energy[seed_idx] = self.seed_energy[seed_idx] * 0.9 + coverage_gain