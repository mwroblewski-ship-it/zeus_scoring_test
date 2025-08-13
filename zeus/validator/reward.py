# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import List, Optional
import numpy as np
import torch
import bittensor as bt
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import (
    MAX_STUPIDITY,
    REWARD_DIFFICULTY_SCALER,  
    REWARD_IMPROVEMENT_WEIGHT,
)


def help_format_miner_output(
    correct: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
    """
    Reshape or slice miner output if it is almost the correct shape.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
       Sliced/reshaped miner output.
    """
    if correct.shape == response.shape:
        return response
    
    try:
        if response.ndim - 1 == correct.ndim and response.shape[-1] == 1:
            # miner forgot to squeeze.
            response = response.squeeze(-1)
        
        return response
    except IndexError:
        # if miner's output is so wrong we cannot even index, do not try anymore
        return response


def get_shape_penalty(correct: torch.Tensor, response: torch.Tensor) -> bool:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: True if there is a shape penalty, False otherwise
    """
    penalty = False
    if response.shape != correct.shape:
        penalty = True
    elif not torch.isfinite(response).all():
        penalty = True

    return penalty

def rmse(
        output_data: torch.Tensor,
        prediction: torch.Tensor,
) -> float:
    """Calculates RMSE between miner prediction and correct output"""
    try:
        return ((prediction - output_data) ** 2).mean().sqrt().item()
    except:
        # shape error etc
        return -1.0

def set_penalties(
    output_data: torch.Tensor,
    miners_data: List[MinerData],
) -> List[MinerData]:
    """
    Calculates and sets penalities for miners based on correct shape and their prediction

    Args:
        output_data (torch.Tensor): ground truth, ONLY used for shape
        miners_data (List[MinerData]): List of MinerData objects containing predictions.

    Returns:
        List[MinerData]: List of MinerData objects with penalty fields
    """
    bt.logging.info(f"🔍 Checking {len(miners_data)} miner responses for penalties...")
    
    penalty_count = 0
    valid_count = 0
    
    for miner_data in miners_data:
        # potentially fix inputs for miners
        original_shape = list(miner_data.prediction.shape)
        miner_data.prediction = help_format_miner_output(output_data, miner_data.prediction)
        fixed_shape = list(miner_data.prediction.shape)
        
        shape_penalty = get_shape_penalty(output_data, miner_data.prediction)
        
        if shape_penalty:
            penalty_count += 1
            bt.logging.warning(f"❌ UID {miner_data.uid}: Shape penalty")
            bt.logging.warning(f"   Expected: {list(output_data.shape)}")
            bt.logging.warning(f"   Original: {original_shape}")
            bt.logging.warning(f"   After fix: {fixed_shape}")
            
            # Sprawdź przyczyny penalty
            if not torch.isfinite(miner_data.prediction).all():
                nan_count = torch.isnan(miner_data.prediction).sum().item()
                inf_count = torch.isinf(miner_data.prediction).sum().item()
                bt.logging.warning(f"   Contains NaN: {nan_count}, Inf: {inf_count}")
        else:
            valid_count += 1
            bt.logging.info(f"✅ UID {miner_data.uid}: Valid response shape {fixed_shape}")
        
        # set penalty, including rmse/reward if there is a penalty
        miner_data.shape_penalty = shape_penalty
    
    bt.logging.info(f"📊 Penalty Summary: {penalty_count} penalized, {valid_count} valid")
    return miners_data


def get_curved_scores(raw_scores: List[float], gamma: float) -> List[float]:
    """
    Given a list of raw float scores (can by any range),
    normalise them to 0-1 scores,
    and apply gamma correction to curve accordingly.

    Note that maximal error is capped at MAX_STUPIDITY * minimal_error if applicable,
    to prevent abuse through distribution shifting.

    This function assumes lower is better!
    """
    min_score = min(raw_scores)
    max_score = min(max(raw_scores), MAX_STUPIDITY * min_score)

    result = []
    for score in raw_scores:
        if max_score == min_score:
            result.append(1.0) # edge case, avoid division by 0
        else:
            # min to prevent getting negative score
            norm_score = (max_score - min(score, max_score)) / (max_score - min_score)
            result.append(np.power(norm_score, gamma)) # apply gamma correction
    
    return result
    

def set_rewards(
    output_data: torch.Tensor,
    miners_data: List[MinerData],
    baseline_data: Optional[torch.Tensor],
    difficulty_grid: np.ndarray,
    min_sota_delta: float
) -> List[MinerData]:
    """
    Calculates rewards for miner predictions based on RMSE and relative difficulty.
    NOTE: it is assumed penalties have already been scored and filtered out, 
      if not will remove them without scoring

    Args:
        output_data (torch.Tensor): The ground truth data.
        miners_data (List[MinerData]): List of MinerData objects containing predictions.
        baseline_data (torch.Tensor): OpenMeteo prediction, where additional incentive is awarded to beat this!
        difficulty_grid (np.ndarray): Difficulty grid for each coordinate.

    Returns:
        List[MinerData]: List of MinerData objects with updated rewards and metrics.
    """
    # Filtruj tylko prawidłowych minerów
    miners_data = [m for m in miners_data if not m.shape_penalty]

    if len(miners_data) == 0:
        bt.logging.warning("❌ No valid miners to reward")
        return miners_data

    bt.logging.info(f"🎁 Calculating rewards for {len(miners_data)} valid miners...")

    # old challenges have no baseline, use 0 to make it not affect scoring.
    baseline_rmse = 0
    if baseline_data is not None:
        baseline_rmse = rmse(output_data, baseline_data)
        bt.logging.info(f"🌐 Baseline (OpenMeteo) RMSE vs ERA5: {baseline_rmse:.4f}")
        
    avg_difficulty = difficulty_grid.mean()
    # make difficulty [-1, 1], then go between [1/scaler, scaler]
    gamma = np.power(REWARD_DIFFICULTY_SCALER, avg_difficulty * 2 - 1)
    bt.logging.info(f"🎲 Challenge difficulty: {avg_difficulty:.3f}, gamma: {gamma:.3f}")

    # compute unnormalised scores
    bt.logging.info(f"📊 Computing individual miner metrics...")
    
    miner_rmses = []
    miner_improvements = []
    
    for miner_data in miners_data:
        miner_data.rmse = rmse(output_data, miner_data.prediction)
        improvement = baseline_rmse - miner_data.rmse - min_sota_delta
        miner_data.baseline_improvement = max(0, improvement)
        
        miner_rmses.append(miner_data.rmse)
        miner_improvements.append(miner_data.baseline_improvement)
        
        bt.logging.info(f"   UID {miner_data.uid}: RMSE={miner_data.rmse:.4f}, improvement={miner_data.baseline_improvement:.4f}")

    # Loguj statystyki przed curvingiem
    bt.logging.info(f"📈 Raw Metrics Summary:")
    bt.logging.info(f"   RMSE range: [{min(miner_rmses):.4f}, {max(miner_rmses):.4f}]")
    bt.logging.info(f"   RMSE mean: {np.mean(miner_rmses):.4f}")
    bt.logging.info(f"   Improvements range: [{min(miner_improvements):.4f}, {max(miner_improvements):.4f}]")
    
    # Apply gamma correction (curving)
    bt.logging.info(f"🎚️  Applying gamma correction (γ={gamma:.3f})...")
    quality_scores = get_curved_scores(miner_rmses, gamma)
    # negative since curving assumes minimal is the best
    improvement_scores = get_curved_scores([-m for m in miner_improvements], gamma)
    
    bt.logging.info(f"   Quality scores range: [{min(quality_scores):.4f}, {max(quality_scores):.4f}]")
    bt.logging.info(f"   Improvement scores range: [{min(improvement_scores):.4f}, {max(improvement_scores):.4f}]")

    # Combine scores and assign final rewards
    bt.logging.info(f"⚖️  Combining scores (quality: {1-REWARD_IMPROVEMENT_WEIGHT:.1%}, improvement: {REWARD_IMPROVEMENT_WEIGHT:.1%})...")
    
    final_rewards = []
    for miner_data, quality, improvement in zip(miners_data, quality_scores, improvement_scores):
        miner_data.reward = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality + REWARD_IMPROVEMENT_WEIGHT * improvement
        final_rewards.append(miner_data.reward)
        
        bt.logging.info(f"   UID {miner_data.uid}: quality={quality:.4f}, improvement={improvement:.4f}, final={miner_data.reward:.6f}")

    # Final rewards summary
    bt.logging.info(f"🏆 Final Rewards Summary:")
    bt.logging.info(f"   Rewards range: [{min(final_rewards):.6f}, {max(final_rewards):.6f}]")
    bt.logging.info(f"   Rewards mean: {np.mean(final_rewards):.6f}")
    
    # Znajdź najlepszego i najgorszego
    best_miner = max(miners_data, key=lambda m: m.reward)
    worst_miner = min(miners_data, key=lambda m: m.reward)
    
    bt.logging.success(f"🥇 Best Performer: UID {best_miner.uid} - Reward: {best_miner.reward:.6f}, RMSE: {best_miner.rmse:.4f}")
    bt.logging.info(f"📊 Worst Performer: UID {worst_miner.uid} - Reward: {worst_miner.reward:.6f}, RMSE: {worst_miner.rmse:.4f}")

    return miners_data