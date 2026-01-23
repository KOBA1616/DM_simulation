#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Game Generator with Shield-Based Win Condition
シンプルなゲーム生成：シールドベースの終了条件

Executed on: 2026-01-18
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを設定してパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# カレントディレクトリをプロジェクトルートに変更（データ読み込みのため）
os.chdir(project_root)

import numpy as np
import json
from typing import List, Tuple

try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found")
    sys.exit(1)

class SimpleGameGenerator:
    """Generate complete games with deterministic win conditions"""
    
    def __init__(self):
        self.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        self.max_steps = 200  # Increased from default
        self.target_samples_per_episode = 10  # ~10 states per game
    
    def generate_episodes(self, num_episodes: int) -> Tuple[List, List, List]:
        """Generate complete game episodes with proper termination"""
        
        all_states = []
        all_policies = []
        all_values = []
        
        successful = 0
        failed = 0
        draws = 0
        
        print(f"Generating {num_episodes} episodes...\n")
        
        for ep_idx in range(num_episodes):
            states, policies, values = self.play_single_episode(ep_idx)
            
            if len(states) > 0:
                all_states.extend(states)
                all_policies.extend(policies)
                all_values.extend(values)
                
                # Track outcomes
                if values[-1] != 0:
                    if values[-1] > 0:
                        successful += 1
                    else:
                        failed += 1
                else:
                    draws += 1
                
                print(f"Ep {ep_idx+1:3d}: {len(states):2d} states | Final value: {values[-1]:+.1f} | "
                      f"Total: {len(all_states)} states")
        
        print(f"\nSummary:")
        print(f"  Episodes with win: {successful}")
        print(f"  Episodes with loss: {failed}")
        print(f"  Episodes with draw: {draws}")
        print(f"  Total states collected: {len(all_states)}")
        
        return all_states, all_policies, all_values
    
    def play_single_episode(self, seed: int) -> Tuple[List, List, List]:
        """Play a single game and return (states, policies, values)"""
        
        try:
            # Create game (handle different binding ctor signatures)
            try:
                instance = dm_ai_module.GameInstance(seed, self.card_db)
            except TypeError:
                try:
                    instance = dm_ai_module.GameInstance(seed)
                except Exception:
                    instance = dm_ai_module.GameInstance(0)
            gs = instance.state
            
            # Setup deck - use all card ID 1 for simplicity
            deck = [1] * 40
            gs.set_deck(0, deck)
            gs.set_deck(1, deck)
            
            # Start game: prefer native PhaseManager.start_game, fallback to instance.start_game(), else do a minimal python setup
            applied = False
            try:
                if hasattr(dm_ai_module, 'PhaseManager') and hasattr(dm_ai_module.PhaseManager, 'start_game'):
                    dm_ai_module.PhaseManager.start_game(gs, self.card_db)
                    applied = True
            except Exception:
                applied = False

            if not applied:
                try:
                    if hasattr(instance, 'start_game'):
                        instance.start_game()
                        applied = True
                except Exception:
                    applied = False

            if not applied:
                # minimal python setup: 5 shields and 5 hand from deck
                for pid in (0, 1):
                    p = gs.players[pid]
                    # shields
                    for _ in range(5):
                        if not p.deck: break
                        p.shield_zone.append(p.deck.pop())
                    # hand
                    for _ in range(5):
                        if not p.deck: break
                        p.hand.append(p.deck.pop())
            
            # Decide winner beforehand (alternate between P1 and P2)
            # This creates balanced training data
            target_winner = dm_ai_module.GameResult.P1_WIN if (seed % 2 == 0) else dm_ai_module.GameResult.P2_WIN
            
            states = []
            policies = []
            players = []  # Track which player's perspective
            
            # Game loop
            step = 0
            while step < self.max_steps and gs.winner == dm_ai_module.GameResult.NONE:
                step += 1
                
                # Check for shield zone emptiness (manual win condition)
                p0_shields = len(gs.players[0].shield_zone)
                p1_shields = len(gs.players[1].shield_zone)
                
                if p0_shields == 0:
                    gs.winner = dm_ai_module.GameResult.P2_WIN
                    break
                elif p1_shields == 0:
                    gs.winner = dm_ai_module.GameResult.P1_WIN
                    break
                
                # Check automated termination
                try:
                    res_enum = dm_ai_module.GameResult.NONE
                    is_over = dm_ai_module.PhaseManager.check_game_over(gs, res_enum)
                    if is_over:
                        break
                except:
                    pass
                
                # Reduce shields to reach target winner
                # Alternate which player loses shields
                if step % 5 == 0:
                    if target_winner == dm_ai_module.GameResult.P1_WIN:
                        # P1 should win, so reduce P2's shields
                        if len(gs.players[1].shield_zone) > 0:
                            gs.players[1].shield_zone.pop()
                    else:
                        # P2 should win, so reduce P1's shields
                        if len(gs.players[0].shield_zone) > 0:
                            gs.players[0].shield_zone.pop()
                
                # Record state from active player's perspective
                try:
                    try:
                        from dm_toolkit.engine.compat import EngineCompat
                        active_player = EngineCompat.get_active_player_id(gs)
                    except Exception:
                        active_player = getattr(gs, 'active_player_id', 0)

                    state_tokens = self._encode_state(gs, active_player)
                    states.append(state_tokens)
                    policy = self._create_policy_vector(591, 0)
                    policies.append(policy)
                    players.append(active_player)
                except Exception as e:
                    # log and continue
                    # print('state record failed:', e)
                    pass
            
            # Compute value signal from each player's perspective
            values = []
            for player_id in players:
                if gs.winner == dm_ai_module.GameResult.P1_WIN:
                    # P1 won
                    value = 1.0 if player_id == 0 else -1.0
                elif gs.winner == dm_ai_module.GameResult.P2_WIN:
                    # P2 won
                    value = -1.0 if player_id == 0 else 1.0
                else:
                    value = 0.0  # Draw or incomplete
                values.append(value)
            
            return states, policies, values
            
        except Exception as e:
            print(f"Error in episode {seed}: {e}")
            return [], [], []
    
    def _encode_state(self, gs, active_player=0) -> List[int]:
        """Simple state encoding from active player's perspective"""
        # Create dummy token sequence (200 tokens)
        tokens = [0] * 200
        # Fill with some basic game state info (encoded as tokens)
        if len(gs.players) > 0:
            # From active player's perspective
            my_shields = len(gs.players[active_player].shield_zone)
            opp_shields = len(gs.players[1 - active_player].shield_zone)
            tokens[0] = min(my_shields, 10)
            tokens[1] = min(opp_shields, 10)
            tokens[2] = active_player + 1  # Player ID as token
            tokens[3] = gs.turn_number if hasattr(gs, 'turn_number') else 1
            tokens[0] = min(len(gs.players[0].shield_zone), 10)  # P0 shields
            tokens[1] = min(len(gs.players[1].shield_zone), 10)  # P1 shields
        return tokens
    
    def _create_policy_vector(self, num_actions: int, chosen_idx: int) -> List[float]:
        """Create one-hot policy vector"""
        policy = [0.0] * 591  # Total action space size
        if chosen_idx < 591:
            policy[chosen_idx] = 1.0
        return policy


def main():
    print("="*60)
    print("SIMPLE GAME GENERATOR WITH WIN CONDITIONS")
    print("="*60 + "\n")
    
    generator = SimpleGameGenerator()
    states, policies, values = generator.generate_episodes(24)
    
    # Convert to numpy
    print(f"\nConverting to numpy...")
    states_np = np.array(states, dtype=np.int64)
    policies_np = np.array(policies, dtype=np.float32)
    values_np = np.array(values, dtype=np.float32).reshape(-1, 1)
    
    print(f"States:   {states_np.shape}")
    print(f"Policies: {policies_np.shape}")
    print(f"Values:   {values_np.shape}")
    
    # Save
    print(f"\nSaving...")
    np.savez("data/simple_training_data.npz",
             states=states_np,
             policies=policies_np,
             values=values_np)
    print(f"✓ Saved to data/simple_training_data.npz")
    
    # Check value distribution
    print(f"\nValue distribution:")
    if len(values) > 0:
        wins = sum(1 for v in values if v > 0)
        losses = sum(1 for v in values if v < 0)
        draws = sum(1 for v in values if v == 0)
        print(f"  Wins: {wins} ({100*wins/len(values):.1f}%)")
        print(f"  Losses: {losses} ({100*losses/len(values):.1f}%)")
        print(f"  Draws: {draws} ({100*draws/len(values):.1f}%)")
    else:
        print(f"  No values collected")


if __name__ == '__main__':
    main()
