"""
SD-Fuzz Main Entry Point
"""
import argparse
from pathlib import Path
from core.fuzzer.sd_fuzzer import SDFuzzer, FuzzConfig


def main():
    parser = argparse.ArgumentParser(description="SD-Fuzz:  State-Aware Fuzzing for ICS Protocols")

    # Target configuration
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=502, help="Target port")
    parser.add_argument("--protocol", type=str, default="modbus-tcp", help="Protocol to fuzz")

    # Fuzzing parameters
    parser.add_argument("--iterations", type=int, default=10000, help="Number of fuzzing iterations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--seed-corpus", type=str, default=None, help="Path to seed corpus directory")

    # Model parameters
    parser.add_argument("--n-states", type=int, default=15, help="Number of HMM states")
    parser.add_argument("--ddpm-timesteps", type=int, default=1000, help="DDPM diffusion timesteps")
    parser.add_argument("--ddpm-epochs", type=int, default=50, help="DDPM training epochs")

    # Output
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")

    args = parser.parse_args()

    # Create configuration
    config = FuzzConfig(
        protocol=args.protocol,
        target_host=args.host,
        target_port=args.port,
        max_iterations=args.iterations,
        batch_size=args.batch_size,
        seed_corpus_path=args.seed_corpus,
        n_states=args.n_states,
        ddpm_timesteps=args.ddpm_timesteps,
        ddpm_train_epochs=args.ddpm_epochs,
        output_dir=args.output_dir
    )

    # Initialize and run fuzzer
    fuzzer = SDFuzzer(config)
    stats = fuzzer.run()

    # Print final summary
    print("\n" + "=" * 60)
    print("FUZZING SUMMARY")
    print("=" * 60)
    print(f"Total iterations: {stats['iterations']}")
    print(f"Total crashes: {stats['crashes']}")
    print(f"Unique crashes: {len(stats['unique_crashes'])}")
    print(f"State coverage: {len(stats['state_coverage'])}/{config.n_states}")
    print(f"Output saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()