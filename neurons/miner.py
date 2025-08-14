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

import time
import torch
import typing
import bittensor as bt

import openmeteo_requests

import numpy as np
from zeus.data.converter import get_converter
from zeus.utils.config import get_device_str
from zeus.utils.time import to_timestamp, timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.protocol import TimePredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to OpenMeteo (https://open-meteo.com/) for predictions.
    You are encouraged to attempt to improve over this by changing the forward function.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        # TODO(miner): Anything specific to your use case you can do here
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()

    def _log_api_request(self, url: str, params: dict):
        """Loguj szczegóły żądania API"""
        try:
            import urllib.parse
            
            bt.logging.info(f"🌐 === MINER OPENMETEO API REQUEST ===")
            bt.logging.info(f"   URL: {url}")
            bt.logging.info(f"   Method: POST")
            
            # Pokaż parametry
            for key, value in params.items():
                if isinstance(value, list) and len(value) > 5:
                    bt.logging.info(f"   {key}: [{value[0]}, {value[1]}, {value[2]}, ...] ({len(value)} total)")
                else:
                    bt.logging.info(f"   {key}: {value}")
            
            # Stwórz pełny URL dla debugowania (jak GET)
            query_string = urllib.parse.urlencode(params, doseq=True)
            full_url = f"{url}?{query_string}"
            
            bt.logging.info(f"   Full URL (as GET): {full_url}")
            bt.logging.info(f"🌐 === END MINER REQUEST ===")
            
        except Exception as e:
            bt.logging.warning(f"Could not log API request details: {e}")
            bt.logging.info(f"🌐 MINER OPENMETEO API REQUEST: {url}")

    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse for a prediction.

        Args:
            synapse (TimePredictionSynapse): The synapse object containing the time range and coordinates

        Returns:
            TimePredictionSynapse: The synapse object with the 'predictions' field set".
        """
        # shape (lat, lon, 2) so a grid of locations
        coordinates = torch.Tensor(synapse.locations)
        start_time = to_timestamp(synapse.start_time)
        end_time = to_timestamp(synapse.end_time)
        
        # 🔥 NOWE: Szczegółowe logowanie żądania
        bt.logging.info("📥" * 20)
        bt.logging.info("📥 OTRZYMANO ŻĄDANIE OD VALIDATORA")
        bt.logging.info("📥" * 20)
        bt.logging.info(f"🎯 Challenge Details:")
        bt.logging.info(f"   Variable: {synapse.variable}")
        bt.logging.info(f"   Grid shape: {coordinates.shape}")
        bt.logging.info(f"   Requested hours: {synapse.requested_hours}")
        bt.logging.info(f"   Time range: {timestamp_to_str(synapse.start_time)} -> {timestamp_to_str(synapse.end_time)}")
        
        # Oblicz bbox dla logowania
        if coordinates.numel() > 0:
            lat_min, lat_max = coordinates[..., 0].min().item(), coordinates[..., 0].max().item()
            lon_min, lon_max = coordinates[..., 1].min().item(), coordinates[..., 1].max().item()
            bt.logging.info(f"   Location bbox: lat[{lat_min:.2f}, {lat_max:.2f}], lon[{lon_min:.2f}, {lon_max:.2f}]")
            
            # Pokaż sample coordinates
            bt.logging.info(f"🗺️  Sample coordinates:")
            bt.logging.info(f"   Top-left: ({coordinates[0, 0, 0]:.2f}, {coordinates[0, 0, 1]:.2f})")
            if coordinates.shape[0] > 1 and coordinates.shape[1] > 1:
                bt.logging.info(f"   Top-right: ({coordinates[0, -1, 0]:.2f}, {coordinates[0, -1, 1]:.2f})")
                bt.logging.info(f"   Bottom-left: ({coordinates[-1, 0, 0]:.2f}, {coordinates[-1, 0, 1]:.2f})")
                bt.logging.info(f"   Bottom-right: ({coordinates[-1, -1, 0]:.2f}, {coordinates[-1, -1, 1]:.2f})")

        ##########################################################################################################
        # TODO (miner) you likely want to improve over this baseline of calling OpenMeteo by changing this section
        
        bt.logging.info(f"🌐 Calling OpenMeteo API...")
        start_api_time = time.time()
        
        latitudes, longitudes = coordinates.view(-1, 2).T
        converter = get_converter(synapse.variable)
        
        bt.logging.info(f"   API request details:")
        bt.logging.info(f"   - Locations: {len(latitudes)} points")
        bt.logging.info(f"   - Variable: {synapse.variable} -> OpenMeteo: {converter.om_name}")
        bt.logging.info(f"   - Time range: {start_time.isoformat()} -> {end_time.isoformat()}")
        
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
        }
        
        # 🔥 NOWE: Loguj szczegóły API request
        self._log_api_request("https://api.open-meteo.com/v1/forecast", params)
        
        try:
            responses = self.openmeteo_api.weather_api(
                "https://api.open-meteo.com/v1/forecast", params=params, method="POST"
            )
            
            api_duration = time.time() - start_api_time
            bt.logging.success(f"✅ OpenMeteo API responded in {api_duration:.2f}s")
            bt.logging.info(f"   Received {len(responses)} location responses")

            # get output as grid of [time, lat, lon, variables]
            if output.shape[0] != synapse.requested_hours:
                bt.logging.warning(f"⚠️ API returned {output.shape[0]} hours, expected {synapse.requested_hours}")
                # Obetnij do wymaganej liczby godzin
                output = output[:synapse.requested_hours]

            output = torch.Tensor(np.stack(
                [
                    np.stack(
                        [
                            r.Hourly().Variables(i).ValuesAsNumpy() 
                            for i in range(r.Hourly().VariablesLength())
                        ],
                        axis=-1
                    )
                    for r in responses
                ],
                axis=1
            )).reshape(synapse.requested_hours, *coordinates.shape[:2], -1)
            # [time, lat, lon] in case of single variable output
            output = output.squeeze(dim=-1)
            # Convert variable(s) to ERA5 units, combines variables for windspeed
            output = converter.om_to_era5(output)
            
            # 🔥 NOWE: Szczegółowe logowanie odpowiedzi
            bt.logging.info(f"📊 Generated Prediction Analysis:")
            bt.logging.info(f"   Final shape: {list(output.shape)}")
            
            # Statystyki predykcji
            pred_stats = {
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item()
            }
            bt.logging.info(f"   Stats: mean={pred_stats['mean']:.4f}, std={pred_stats['std']:.4f}")
            bt.logging.info(f"   Range: [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]")
            
            # Sprawdź czy są NaN/Inf wartości
            nan_count = torch.isnan(output).sum().item()
            inf_count = torch.isinf(output).sum().item()
            if nan_count > 0 or inf_count > 0:
                bt.logging.warning(f"⚠️  Found {nan_count} NaN and {inf_count} Inf values!")
            else:
                bt.logging.success(f"✅ All values are finite")
            
            # Pokaż sample wartości dla różnych czasów
            if output.numel() > 0:
                bt.logging.info(f"🔍 Sample Prediction Values:")
                # Pierwsza godzina
                sample_vals = output[0, :min(2, output.shape[1]), :min(2, output.shape[2])]
                bt.logging.info(f"   Hour 0: {sample_vals.flatten().tolist()}")
                
                # Ostatnia godzina jeśli więcej niż 1
                if output.shape[0] > 1:
                    sample_vals = output[-1, :min(2, output.shape[1]), :min(2, output.shape[2])]
                    bt.logging.info(f"   Hour {output.shape[0]-1}: {sample_vals.flatten().tolist()}")
                    
                    # Sprawdź temporal consistency (czy wartości zmieniają się w czasie)
                    temporal_diff = (output[-1] - output[0]).abs().mean().item()
                    bt.logging.info(f"   Temporal variation: {temporal_diff:.4f}")
            
        except Exception as e:
            bt.logging.error(f"❌ OpenMeteo API error: {e}")
            # Detailed error logging
            bt.logging.error(f"   Exception type: {type(e).__name__}")
            import traceback
            bt.logging.error(f"   Traceback: {traceback.format_exc()}")
            
            # Fallback: zwróć tensor wypełniony średnimi wartościami
            fallback_shape = (synapse.requested_hours, *coordinates.shape[:2])
            
            # Użyj rozsądnych default wartości zależnie od zmiennej
            if synapse.variable == "2m_temperature":
                # Temperatura ~15°C = 288K
                output = torch.full(fallback_shape, 288.15)
            elif synapse.variable == "total_precipitation":
                # Minimalne opady
                output = torch.zeros(fallback_shape)
            else:
                # Inne zmienne - neutralne wartości
                output = torch.zeros(fallback_shape)
            
            bt.logging.warning(f"⚠️  Using fallback prediction: shape {list(output.shape)}, value {output[0,0,0].item():.4f}")

        ##########################################################################################################
        
        bt.logging.info(f"📤 Sending response with shape: {list(output.shape)}")
        bt.logging.info("📥" * 20)

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse
    

    async def blacklist(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority(self, synapse: TimePredictionSynapse) -> float:
        return await self._priority(synapse)
    
    

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"⛏️  Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)