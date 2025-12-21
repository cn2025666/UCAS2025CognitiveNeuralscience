import gymnasium as gym
import neurogym as ngym
import numpy as np

task = 'DelayPairedAssociation-v0'
# Try passing sigma (noise) and timing
try:
    # Standard neurogym tasks often accept kwargs in make, or via apply_wrapper
    # Let's try passing arguments to make
    env = gym.make(task, sigma=0.5, timing={'delay': ('uniform', [100, 500])})
    print("Successfully created env with kwargs")
    print(env.unwrapped)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
except Exception as e:
    print(f"Failed to create env with kwargs: {e}")

# Check if we can use wrappers
try:
    env = gym.make(task)
    env = ngym.wrappers.Noise(env, std_noise=0.1)
    print("Successfully applied Noise wrapper")
except Exception as e:
    print(f"Failed to apply Noise wrapper: {e}")
