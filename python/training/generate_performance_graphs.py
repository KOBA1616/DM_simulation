import os
import sys
import time
import json
import matplotlib.pyplot as plt

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import dm_ai_module
    # Import Verify Performance from the same directory structure
    from training.verify_performance import PerformanceVerifier
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for running script directly inside python/training
    sys.path.append(os.path.dirname(__file__))
    try:
        from verify_performance import PerformanceVerifier
    except ImportError:
        print("Crucial imports failed.")
        sys.exit(1)

def run_benchmark():
    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print("Cards JSON not found.")
        return

    print("Loading Card Database...")
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    # Configuration for Benchmark
    sim_steps = [10, 25, 50, 100, 200, 400]
    episodes = 10 # Keep it low for interactive speed, increase if needed
    scenario = "lethal_puzzle_easy"
    model_type = "resnet" # using CPU random/baseline

    verifier = PerformanceVerifier(card_db, None, model_type=model_type)

    results = {
        "sims": [],
        "throughput": [],
        "win_rate": [],
        "avg_time": []
    }

    print(f"--- Starting Benchmark: {model_type.upper()} on {scenario} ---")
    
    for sims in sim_steps:
        print(f"\nRunning with {sims} simulations...")
        
        # Measure time
        t0 = time.time()
        # Note: verifier.verify has its own print but returns win_rate
        # We assume verifier setup is reused (model loaded once)
        # Episodes=loop count
        win_rate = verifier.verify(
            scenario_name=scenario,
            episodes=episodes,
            mcts_sims=sims,
            batch_size=16, # Smaller batch for CPU
            num_threads=4,
            pimc=False
        )
        t1 = time.time()
        
        total_time = t1 - t0
        throughput = episodes / total_time
        
        results["sims"].append(sims)
        results["throughput"].append(throughput)
        results["win_rate"].append(win_rate)
        results["avg_time"].append(total_time / episodes)
        
        print(f"Finished {sims} sims: {throughput:.2f} games/sec, WinRate={win_rate}%")

    # Cleanups
    if hasattr(dm_ai_module, "clear_flat_batch_callback"):
        dm_ai_module.clear_flat_batch_callback()

    # Generate Plots
    generate_plots(results)

def generate_plots(results):
    output_dir = os.path.join(project_root, 'docs', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Throughput vs Sims (Efficiency)
    plt.figure(figsize=(10, 6))
    plt.plot(results["sims"], results["throughput"], marker='o', linewidth=2, color='#1f77b4')
    plt.title('CPU Simulation Efficiency: Throughput vs Search Depth')
    plt.xlabel('MCTS Simulations per Move')
    plt.ylabel('Throughput (Games / Second)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points
    for i, txt in enumerate(results["throughput"]):
        plt.annotate(f"{txt:.1f}", (results["sims"][i], results["throughput"][i]), 
                     xytext=(0, 5), textcoords='offset points', ha='center')
        
    plt.savefig(os.path.join(output_dir, 'benchmark_throughput.png'))
    plt.close()
    
    # 2. Win Rate vs Sims (Quality)
    plt.figure(figsize=(10, 6))
    plt.plot(results["sims"], results["win_rate"], marker='s', linewidth=2, color='#ff7f0e')
    plt.title(f'Search Quality: Win Rate vs Search Depth')
    plt.xlabel('MCTS Simulations per Move')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, txt in enumerate(results["win_rate"]):
        plt.annotate(f"{txt:.1f}%", (results["sims"][i], results["win_rate"][i]), 
                     xytext=(0, 5), textcoords='offset points', ha='center')

    plt.savefig(os.path.join(output_dir, 'benchmark_winrate.png'))
    plt.close()

    print(f"\nGraphs saved to: {output_dir}")

if __name__ == "__main__":
    run_benchmark()
