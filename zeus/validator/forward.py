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
from functools import partial
import time
import bittensor as bt
import wandb
import numpy as np
import torch

from zeus.data.sample import Era5Sample
from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.utils.misc import split_list
from zeus.utils.time import timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.validator.reward import set_rewards, set_penalties, rmse
from zeus.validator.miner_data import MinerData
from zeus.utils.logging import maybe_reset_wandb
from zeus.base.validator import BaseValidatorNeuron
from zeus.validator.constants import FORWARD_DELAY_SECONDS, REWARD_IMPROVEMENT_MIN_DELTA
from zeus.validator.enhanced_validator import EnhancedERA5Validator


def log_detailed_comparison(sample: Era5Sample, baseline_data: Optional[torch.Tensor], miners_data: List[MinerData]):
    """
    Szczegółowe logowanie porównujące ground truth z ERA5 z predykcjami minerów i baseline
    """
    bt.logging.info("=" * 80)
    bt.logging.info("📊 SZCZEGÓŁOWE PORÓWNANIE PREDYKCJI")
    bt.logging.info("=" * 80)
    
    if sample.output_data is None:
        bt.logging.warning("❌ Brak ground truth data do porównania")
        return
    
    ground_truth = sample.output_data
    
    # Pokaż podstawowe info o challenge
    bt.logging.info(f"🎯 Challenge Info:")
    bt.logging.info(f"   Variable: {sample.variable}")
    bt.logging.info(f"   Grid shape: {ground_truth.shape}")
    bt.logging.info(f"   Time range: {timestamp_to_str(sample.start_timestamp)} -> {timestamp_to_str(sample.end_timestamp)}")
    bt.logging.info(f"   Location: {bbox_to_str(sample.get_bbox())}")
    bt.logging.info(f"   Predict hours: {sample.predict_hours}")
    
    # Statystyki ground truth
    gt_stats = {
        "mean": ground_truth.mean().item(),
        "std": ground_truth.std().item(),
        "min": ground_truth.min().item(),
        "max": ground_truth.max().item()
    }
    bt.logging.info(f"📏 Ground Truth Stats (ERA5): mean={gt_stats['mean']:.4f}, std={gt_stats['std']:.4f}, min={gt_stats['min']:.4f}, max={gt_stats['max']:.4f}")
    
    # Pokaż kilka przykładowych wartości z różnych miejsc i czasów
    if ground_truth.numel() > 0:
        bt.logging.info(f"🔍 Sample Ground Truth Values:")
        # Pierwsza godzina, pierwsze 3x3 punkty (jeśli istnieją)
        sample_vals = ground_truth[0, :min(3, ground_truth.shape[1]), :min(3, ground_truth.shape[2])]
        bt.logging.info(f"   Hour 0, top-left 3x3 grid: {sample_vals.flatten().tolist()}")
        
        if ground_truth.shape[0] > 1:
            # Ostatnia godzina
            sample_vals = ground_truth[-1, :min(3, ground_truth.shape[1]), :min(3, ground_truth.shape[2])]
            bt.logging.info(f"   Hour {ground_truth.shape[0]-1}, top-left 3x3 grid: {sample_vals.flatten().tolist()}")
    
    # Baseline (OpenMeteo) porównanie
    if baseline_data is not None and baseline_data.shape == ground_truth.shape:
        baseline_rmse = rmse(ground_truth, baseline_data)
        baseline_stats = {
            "mean": baseline_data.mean().item(),
            "std": baseline_data.std().item(),
            "min": baseline_data.min().item(),
            "max": baseline_data.max().item()
        }
        
        bt.logging.info(f"🌐 OpenMeteo Baseline:")
        bt.logging.info(f"   RMSE vs Ground Truth: {baseline_rmse:.4f}")
        bt.logging.info(f"   Stats: mean={baseline_stats['mean']:.4f}, std={baseline_stats['std']:.4f}, min={baseline_stats['min']:.4f}, max={baseline_stats['max']:.4f}")
        
        # Pokaż przykładowe wartości baseline
        if baseline_data.numel() > 0:
            sample_vals = baseline_data[0, :min(3, baseline_data.shape[1]), :min(3, baseline_data.shape[2])]
            bt.logging.info(f"   Sample values (hour 0): {sample_vals.flatten().tolist()}")
    else:
        bt.logging.warning("❌ Brak prawidłowych danych baseline do porównania")
    
    # Porównanie każdego minera
    bt.logging.info(f"⛏️  MINERS COMPARISON ({len(miners_data)} miners):")
    
    valid_miners = [m for m in miners_data if not m.shape_penalty]
    invalid_miners = [m for m in miners_data if m.shape_penalty]
    
    # Pokaż minerów z penalty
    if invalid_miners:
        bt.logging.info(f"❌ Miners with shape penalty ({len(invalid_miners)}):")
        for miner in invalid_miners:
            bt.logging.info(f"   UID {miner.uid}: shape={list(miner.prediction.shape)}, expected={list(ground_truth.shape)}")
    
    # Pokaż prawidłowych minerów
    if valid_miners:
        bt.logging.info(f"✅ Valid miners ({len(valid_miners)}):")
        
        # Sortuj minerów po RMSE (najlepsi pierwsi)
        valid_miners_sorted = sorted(valid_miners, key=lambda m: m.rmse if m.rmse is not None else float('inf'))
        
        for i, miner in enumerate(valid_miners_sorted):
            miner_stats = {
                "mean": miner.prediction.mean().item(),
                "std": miner.prediction.std().item(),
                "min": miner.prediction.min().item(),
                "max": miner.prediction.max().item()
            }
            
            ranking_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            
            bt.logging.info(f"   {ranking_symbol} UID {miner.uid} (Rank {i+1}):")
            if miner.rmse is not None:
                bt.logging.info(f"      RMSE: {miner.rmse:.4f}")
                bt.logging.info(f"      Reward: {miner.reward:.4f}")
            else:
                bt.logging.info("====== Miner RMSE and Reward is missing")
            
            bt.logging.info(f"      Stats: mean={miner_stats['mean']:.4f}, std={miner_stats['std']:.4f}")
            
            if hasattr(miner, 'baseline_improvement') and miner.baseline_improvement is not None:
                bt.logging.info(f"      Baseline improvement: {miner.baseline_improvement:.4f}")
            
            # Pokaż przykładowe wartości predykcji
            if miner.prediction.numel() > 0:
                sample_vals = miner.prediction[0, :min(3, miner.prediction.shape[1]), :min(3, miner.prediction.shape[2])]
                bt.logging.info(f"      Sample values (hour 0): {sample_vals.flatten().tolist()}")
        
        # Pokaż best vs worst porównanie
        if len(valid_miners_sorted) > 1:
            best_miner = valid_miners_sorted[0]
            worst_miner = valid_miners_sorted[-1]
            if miner.rmse is not None:
                improvement_pct = ((worst_miner.rmse - best_miner.rmse) / worst_miner.rmse) * 100
            
                bt.logging.info(f"🏆 Best vs Worst:")
                bt.logging.info(f"   Best UID {best_miner.uid}: RMSE {best_miner.rmse:.4f}")
                bt.logging.info(f"   Worst UID {worst_miner.uid}: RMSE {worst_miner.rmse:.4f}")
            bt.logging.info(f"   Improvement: {improvement_pct:.1f}% better RMSE")
    
    bt.logging.info("=" * 80)


async def forward(self: BaseValidatorNeuron):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    start_forward = time.time()
    # based on the block, we decide if we should score old stored predictions.
    if self.database.should_score(self.block):
        bt.logging.info(f"🔄 Potentially scoring stored predictions for live ERA5 data.")
        self.database.score_and_prune(score_func=partial(complete_challenge, self))
        return
    
    data_loader: Era5CDSLoader = self.cds_loader
    if not data_loader.is_ready():
        bt.logging.info("⏳ Data loader is not ready yet... Waiting until ERA5 data is downloaded.")
        time.sleep(20)  # Don't need to spam above message
        return

    bt.logging.info(f"🎲 Sampling data...")
    sample = data_loader.get_sample()
    bt.logging.success(
        f"✅ Data sampled with bounding box {bbox_to_str(sample.get_bbox())} for variable {sample.variable}"
    )
    bt.logging.success(
        f"⏰ Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Asked to predict {sample.predict_hours} hours ahead."
    )

    # get the baseline data, which we also store and check against
    bt.logging.info("🌐 Fetching OpenMeteo baseline")
    sample.output_data = self.open_meteo_loader.get_output(sample)
  
    miner_uids = [117]

    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    miner_hotkeys: List[str] = list([axon.hotkey for axon in axons])

    bt.logging.info(f"⛏️  Querying {len(miner_uids)} miners: {miner_uids}")
    start_request = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=sample.get_synapse(),
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    bt.logging.success(f"📨 Responses received in {time.time() - start_request:.2f}s")

    miners_data = parse_miner_inputs(self, sample, miner_hotkeys, responses)
    
    # 🔥 NOWE: Szczegółowe logowanie porównania przed scoringiem
    bt.logging.info("🔍 Analyzing responses vs baseline...")
    log_immediate_baseline_comparison(sample, miners_data)
    
    # Identify miners who should receive a penalty
    good_miners, bad_miners = split_list(miners_data, lambda m: not m.shape_penalty)

    # penalise 
    if len(bad_miners) > 0:
        uids = [miner.uid for miner in bad_miners]
        self.uid_tracker.mark_finished(uids, good=False)
        bt.logging.warning(f"❌ Punishing miners that did not respond properly: {uids}")
        for miner in bad_miners:
            bt.logging.warning(f"   UID {miner.uid}: shape={list(miner.prediction.shape)}, expected={list(sample.output_data.shape)}")
        
        self.update_scores(
            [miner.reward for miner in bad_miners],
            uids,
        )
        do_wandb_logging(self, sample, bad_miners)

    if len(good_miners) > 0:
        uids = [m.uid for m in good_miners]
        # store non-penalty miners for proxy
        self.uid_tracker.mark_finished(uids, good=True)
        hotkeys = [miner.hotkey for miner in good_miners]
        predictions = [miner.prediction for miner in good_miners]
      
        bt.logging.success(f"💾 Storing challenge and sensible miner responses in SQLite database: {uids}")
        bt.logging.info(f"⏱️  This challenge will be scored when ERA5 ground truth becomes available")
        self.database.insert(sample, hotkeys, predictions)

    # prevent W&B logs from becoming massive
    maybe_reset_wandb(self)
    # Introduce a delay to prevent spamming requests
    time.sleep(max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))


def log_immediate_baseline_comparison(sample: Era5Sample, miners_data: List[MinerData]):
    """
    Loguje natychmiastowe porównanie z baseline (OpenMeteo) - bez czekania na ERA5 ground truth
    """
    if sample.output_data is None:
        bt.logging.warning("❌ Brak baseline data do natychmiastowego porównania")
        return
    
    baseline_data = sample.output_data
    
    bt.logging.info("🔄" * 30)
    bt.logging.info("📊 NATYCHMIASTOWE PORÓWNANIE Z BASELINE (OpenMeteo)")
    bt.logging.info("🔄" * 30)
    
    # Statystyki baseline
    baseline_stats = {
        "mean": baseline_data.mean().item(),
        "std": baseline_data.std().item(),
        "min": baseline_data.min().item(),
        "max": baseline_data.max().item()
    }
    
    bt.logging.info(f"🌐 OpenMeteo Baseline Stats:")
    bt.logging.info(f"   Shape: {list(baseline_data.shape)}")
    bt.logging.info(f"   Mean: {baseline_stats['mean']:.4f}")
    bt.logging.info(f"   Std: {baseline_stats['std']:.4f}")
    bt.logging.info(f"   Range: [{baseline_stats['min']:.4f}, {baseline_stats['max']:.4f}]")
    
    # Pokaż sample wartości z baseline
    if baseline_data.numel() > 0:
        bt.logging.info(f"🔍 Sample Baseline Values:")
        # Pierwsza godzina, pierwsze punkty
        sample_vals = baseline_data[0, :min(2, baseline_data.shape[1]), :min(2, baseline_data.shape[2])]
        bt.logging.info(f"   Hour 0, top-left: {sample_vals.flatten().tolist()}")
        
        if baseline_data.shape[0] > 1:
            sample_vals = baseline_data[-1, :min(2, baseline_data.shape[1]), :min(2, baseline_data.shape[2])]
            bt.logging.info(f"   Hour {baseline_data.shape[0]-1}, top-left: {sample_vals.flatten().tolist()}")
    
    # Porównanie każdego minera z baseline
    valid_miners = [m for m in miners_data if not m.shape_penalty]
    
    if not valid_miners:
        bt.logging.warning("❌ Brak prawidłowych minerów do porównania")
        return
    
    bt.logging.info(f"⛏️  MINERS vs BASELINE COMPARISON ({len(valid_miners)} valid miners):")
    
    baseline_vs_miner_rmses = []
    
    for miner in valid_miners:
        try:
            # Oblicz RMSE względem baseline (nie ground truth!)
            miner_vs_baseline_rmse = rmse(baseline_data, miner.prediction)
            baseline_vs_miner_rmses.append(miner_vs_baseline_rmse)
            
            miner_stats = {
                "mean": miner.prediction.mean().item(),
                "std": miner.prediction.std().item(), 
                "min": miner.prediction.min().item(),
                "max": miner.prediction.max().item()
            }
            
            # Czy miner różni się znacząco od baseline?
            mean_diff = abs(miner_stats['mean'] - baseline_stats['mean'])
            std_diff = abs(miner_stats['std'] - baseline_stats['std'])
            
            status_icon = "🟢" if miner_vs_baseline_rmse < 1.0 else "🟡" if miner_vs_baseline_rmse < 3.0 else "🔴"
            
            bt.logging.info(f"   {status_icon} UID {miner.uid}:")
            bt.logging.info(f"      RMSE vs Baseline: {miner_vs_baseline_rmse:.4f}")
            bt.logging.info(f"      Mean: {miner_stats['mean']:.4f} (diff: {mean_diff:.4f})")
            bt.logging.info(f"      Std: {miner_stats['std']:.4f} (diff: {std_diff:.4f})")
            bt.logging.info(f"      Range: [{miner_stats['min']:.4f}, {miner_stats['max']:.4f}]")
            
            # Sample wartości
            if miner.prediction.numel() > 0:
                sample_vals = miner.prediction[0, :min(2, miner.prediction.shape[1]), :min(2, miner.prediction.shape[2])]
                bt.logging.info(f"      Sample values (hour 0): {sample_vals.flatten().tolist()}")
            
        except Exception as e:
            bt.logging.error(f"❌ Błąd porównania UID {miner.uid}: {e}")
    
    # Podsumowanie statystyk
    if baseline_vs_miner_rmses:
        avg_miner_rmse = np.mean(baseline_vs_miner_rmses)
        best_miner_rmse = min(baseline_vs_miner_rmses)
        worst_miner_rmse = max(baseline_vs_miner_rmses)
        
        bt.logging.info(f"📈 Miners vs Baseline Summary:")
        bt.logging.info(f"   Average RMSE vs baseline: {avg_miner_rmse:.4f}")
        bt.logging.info(f"   Best RMSE vs baseline: {best_miner_rmse:.4f}")
        bt.logging.info(f"   Worst RMSE vs baseline: {worst_miner_rmse:.4f}")
        
        # Sprawdź czy ktoś poprawił baseline
        better_than_baseline_count = sum(1 for rmse_val in baseline_vs_miner_rmses if rmse_val < 0.1)
        if better_than_baseline_count > 0:
            bt.logging.success(f"🎉 {better_than_baseline_count} miners performed similarly to baseline!")
        else:
            bt.logging.info(f"📊 No miners significantly improved over baseline yet")
    
    bt.logging.info("🔄" * 30)


def parse_miner_inputs(
    self,
    sample: Era5Sample,
    hotkeys: List[str],
    responses: List[torch.Tensor],
) -> List[MinerData]:
    """
    Convert input to MinerData and calculate (and populate) their penalty fields.
    Return a list of MinerData
    """
    lookup = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}

    # Make miner data for each miner that is still alive
    miners_data = []
    for hotkey, prediction in zip(hotkeys, responses):
        uid = lookup.get(hotkey, None)
        if uid is not None:
            miners_data.append(MinerData(uid=uid, hotkey=hotkey, prediction=prediction))

    # pre-calculate penalities since we need those to filter
    return set_penalties(
        output_data=sample.output_data,
        miners_data=miners_data
    )

def complete_challenge(
    self,
    sample: Era5Sample,
    baseline: Optional[torch.Tensor],
    hotkeys: List[str],
    predictions: List[torch.Tensor],
) -> Optional[List[MinerData]]:
    """
    Complete a challenge by reward all miners. 
    ENHANCED VERSION - uses real ERA5 data when possible
    """
    
    # 🆕 NOWE: Spróbuj Enhanced Validation
    try:
        bt.logging.info("🚀 Attempting Enhanced Validation with real ERA5 data...")
        
        # Utwórz enhanced validator
        enhanced_validator = EnhancedERA5Validator()
        
        # Parse miner inputs
        miners_data = []
        lookup = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}
        
        for hotkey, prediction in zip(hotkeys, predictions):
            uid = lookup.get(hotkey, None)
            if uid is not None:
                miner_data = MinerData(uid=uid, hotkey=hotkey, prediction=prediction)
                miners_data.append(miner_data)
        
        # Uruchom enhanced validation
        validation_results = enhanced_validator.enhanced_validation(
            sample, miners_data, baseline
        )
        
        # Zapisz raport
        enhanced_validator.save_validation_report(
            sample, validation_results, miners_data, baseline
        )
        
        # Jeśli udało się pobrać ground truth z ERA5, użyj go
        if validation_results["ground_truth_available"]:
            ground_truth = enhanced_validator.get_era5_ground_truth(sample)
            if ground_truth is not None:
                sample.output_data = ground_truth
                bt.logging.success("✅ Using real ERA5 data for final scoring!")
        
        # Kontynuuj z normalnym scoringiem
        miners_data = set_penalties(sample.output_data, miners_data)
        miners_data = set_rewards(
            output_data=sample.output_data,
            miners_data=miners_data,
            baseline_data=baseline,
            difficulty_grid=self.difficulty_loader.get_difficulty_grid(sample),
            min_sota_delta=REWARD_IMPROVEMENT_MIN_DELTA[sample.variable]
        )
        
        self.update_scores(
            [miner.reward for miner in miners_data],
            [miner.uid for miner in miners_data],
        )
        
        bt.logging.success(f"🏆 Enhanced scoring completed for UIDs: {[miner.uid for miner in miners_data]}")
        do_wandb_logging(self, sample, miners_data, baseline)
        return miners_data
        
    except Exception as e:
        bt.logging.warning(f"⚠️ Enhanced validation failed: {e}")
        bt.logging.info("🔄 Falling back to standard validation...")
    
    # FALLBACK: Standardowa walidacja (oryginalny kod)
    bt.logging.info(f"🌍 Fetching ERA5 ground truth for stored challenge...")
    era5_ground_truth = self.cds_loader.get_output(sample)
    
    if era5_ground_truth is None:
        bt.logging.warning(f"❌ ERA5 ground truth not yet available for challenge")
        return None
    
    sample.output_data = era5_ground_truth
    
    miners_data = parse_miner_inputs(self, sample, hotkeys, predictions)
    
    bt.logging.info("🎯 ERA5 GROUND TRUTH NOW AVAILABLE - FINAL SCORING!")
    log_detailed_comparison(sample, baseline, miners_data)
    
    miners_data = set_rewards(
        output_data=sample.output_data, 
        miners_data=miners_data, 
        baseline_data=baseline,
        difficulty_grid=self.difficulty_loader.get_difficulty_grid(sample),
        min_sota_delta=REWARD_IMPROVEMENT_MIN_DELTA[sample.variable]
    )

    self.update_scores(
        [miner.reward for miner in miners_data],
        [miner.uid for miner in miners_data],
    )
    
    bt.logging.success(f"🏆 Scored stored challenges for uids: {[miner.uid for miner in miners_data]}")
    log_final_era5_results(sample, baseline, miners_data)
    do_wandb_logging(self, sample, miners_data, baseline)
    return miners_data

def log_final_era5_results(sample: Era5Sample, baseline: Optional[torch.Tensor], miners_data: List[MinerData]):
    """
    Loguje finalne wyniki po scoringu z ERA5 ground truth
    """
    bt.logging.info("🏁" * 30)
    bt.logging.info("🎯 FINALNE WYNIKI - ERA5 GROUND TRUTH")
    bt.logging.info("🏁" * 30)
    
    ground_truth = sample.output_data
    
    # ERA5 vs OpenMeteo baseline
    if baseline is not None and baseline.shape == ground_truth.shape:
        baseline_rmse = rmse(ground_truth, baseline)
        bt.logging.info(f"🌐 OpenMeteo vs ERA5 Ground Truth:")
        bt.logging.info(f"   Baseline RMSE: {baseline_rmse:.4f}")
        
        # Pokaż różnice między baseline a ground truth
        diff_stats = (baseline - ground_truth)
        bt.logging.info(f"   Prediction Error Stats: mean={diff_stats.mean():.4f}, std={diff_stats.std():.4f}")
    
    # Miners vs ERA5 ground truth
    valid_miners = [m for m in miners_data if not m.shape_penalty and m.rmse is not None]
    
    if valid_miners:
        bt.logging.info(f"🏆 FINAL MINER RANKINGS vs ERA5:")
        
        # Sortuj po reward (najwyższy pierwszy)
        miners_sorted = sorted(valid_miners, key=lambda m: m.reward, reverse=True)
        
        for i, miner in enumerate(miners_sorted):
            rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            
            bt.logging.info(f"   {rank_symbol} UID {miner.uid}:")
            bt.logging.info(f"      Final Reward: {miner.reward:.6f}")
            if miner.rmse is not None:
                bt.logging.info(f"      RMSE vs ERA5: {miner.rmse:.4f}")
            
            if hasattr(miner, 'baseline_improvement') and miner.baseline_improvement is not None:
                improvement_status = "🎉" if miner.baseline_improvement > 0 else "📊"
                bt.logging.info(f"      {improvement_status} Baseline improvement: {miner.baseline_improvement:.4f}")
        
        # Podsumowanie
        avg_reward = np.mean([m.reward for m in miners_sorted])
        avg_rmse = np.mean([m.rmse for m in miners_sorted])
        best_reward = miners_sorted[0].reward
        best_rmse = min(m.rmse for m in miners_sorted)
        
        bt.logging.info(f"📈 Final Summary:")
        bt.logging.info(f"   Average reward: {avg_reward:.6f}")
        bt.logging.info(f"   Average RMSE: {avg_rmse:.4f}")
        bt.logging.info(f"   Best reward: {best_reward:.6f}")
        bt.logging.info(f"   Best RMSE: {best_rmse:.4f}")
        
        # Sprawdź kto pokonał baseline
        if baseline is not None:
            baseline_rmse = rmse(ground_truth, baseline)
            better_than_baseline = [m for m in miners_sorted if m.rmse < baseline_rmse]
            if better_than_baseline:
                bt.logging.success(f"🎉 {len(better_than_baseline)} miners beat OpenMeteo baseline!")
                for miner in better_than_baseline:
                    improvement = ((baseline_rmse - miner.rmse) / baseline_rmse) * 100
                    bt.logging.success(f"   UID {miner.uid}: {improvement:.1f}% better than baseline")
            else:
                bt.logging.info(f"📊 No miners beat baseline RMSE of {baseline_rmse:.4f}")
    
    bt.logging.info("🏁" * 30)


def do_wandb_logging(
        self, 
        challenge: Era5Sample, 
        miners_data: List[MinerData], 
        baseline: Optional[torch.Tensor] = None
    ):
    if self.config.wandb.off:
        return
    
    for miner in miners_data:
        wandb.log(
            {f"miner_{challenge.variable}_{miner.uid}_{key}": val for key, val in miner.metrics.items()},
            commit=False,  # All logging should be the same commit
        )

    uid_to_hotkey = {miner.uid: miner.hotkey for miner in miners_data}
    wandb_data = {
        "query_timestamp": challenge.query_timestamp,
        "variable": challenge.variable,
        "start_timestamp": challenge.start_timestamp,
        "end_timestamp": challenge.end_timestamp,
        "predict_hours": challenge.predict_hours,
        "lat_lon_bbox": challenge.get_bbox(),
        "uid_to_hotkey": uid_to_hotkey,
    }
    
    # Dodaj baseline RMSE jeśli dostępne
    if baseline is not None and challenge.output_data is not None:
        try:
            baseline_rmse_val = rmse(challenge.output_data, baseline)
            wandb_data["baseline_rmse"] = baseline_rmse_val
        except:
            pass
    
    wandb.log(wandb_data)