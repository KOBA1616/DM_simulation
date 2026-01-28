import argparse
import sys
import os

from dm_toolkit.gui.console_repl import run_console
from dm_toolkit.domain.headless_simulation import run_simulation
from dm_toolkit.validator.card_validator import CardValidator

def run_validate(args):
    filepath = args.filepath
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1

    validator = CardValidator()
    # Check if directory or file
    if os.path.isdir(filepath):
        # Scan dir? For now assume file as per CardValidator implementation
        print(f"Error: {filepath} is a directory. Please point to a JSON file.")
        return 1

    all_errors = validator.validate_file(filepath)

    if not all_errors:
        print(f"Validation Successful: {filepath}")
        return 0
    else:
        print(f"Validation Failed: {len(all_errors)} cards with errors.")
        for card_id, errors in all_errors.items():
            print(f"Card ID {card_id}:")
            for err in errors:
                print(f"  - {err}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="DM Toolkit CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Console Subcommand
    parser_console = subparsers.add_parser("console", help="Run interactive console (REPL)")
    parser_console.add_argument('--p0-human', action='store_true', help="Player 0 is human")
    parser_console.add_argument('--p1-human', action='store_true', help="Player 1 is human")
    parser_console.add_argument('--log-level', type=str, default='INFO', help='logging level')
    parser_console.add_argument('--dump-json', type=str, default=None, help='dump state to JSON on exit')
    parser_console.add_argument('--auto', type=int, default=None, help='run auto loop N iterations')

    # Sim Subcommand
    parser_sim = subparsers.add_parser("sim", help="Run headless simulation")
    parser_sim.add_argument("--cards", default=os.path.join("data", "cards.json"), help="Path to cards.json")
    parser_sim.add_argument("--meta", default=os.path.join("data", "meta_decks.json"), help="Path to meta_decks.json")
    parser_sim.add_argument("--games", type=int, default=100, help="Number of games")
    parser_sim.add_argument("--seed", type=int, default=None, help="Random seed")
    parser_sim.add_argument("--quiet", action="store_true", help="Suppress output")
    parser_sim.add_argument("--model", default=None, help="Path to trained model file")

    # Validate Subcommand
    parser_val = subparsers.add_parser("validate", help="Validate card data")
    parser_val.add_argument("filepath", help="Path to JSON file to validate")

    args = parser.parse_args()

    if args.command == "console":
        run_console(args)
    elif args.command == "sim":
        run_simulation(args)
    elif args.command == "validate":
        sys.exit(run_validate(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
