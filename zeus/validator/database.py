from typing import List, Callable
import sqlite3
import time
import torch
import pandas as pd
import json
import bittensor as bt

from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.validator.constants import DATABASE_LOCATION
from zeus.data.sample import Era5Sample


class ResponseDatabase:

    def __init__(
        self,
        cds_loader: Era5CDSLoader,
        db_path: str = DATABASE_LOCATION,
    ):
        self.cds_loader = cds_loader
        self.db_path = db_path
        self.create_tables()
        # start at 0 so it always syncs at startup
        self.last_synced_block = 0

    def should_score(self, block: int) -> bool:
        """
        Check if the database should score its stored miner predictions.
        This is done roughly hourly, so with one block every 12 seconds this means
        if the current block is more than 300 blocks ahead of the last synced block, we should score.
        """
        if not self.cds_loader.is_ready():
            return False
        if block - self.last_synced_block > 5:
            self.last_synced_block = block
            return True
        return False

    def create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS challenges (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    lat_start REAL,
                    lat_end REAL,
                    lon_start REAL,
                    lon_end REAL,
                    start_timestamp REAL,
                    end_timestamp REAL,
                    hours_to_predict INTEGER,
                    baseline TEXT,
                    inserted_at REAL,
                    variable TEXT DEFAULT '2m_temperature'
                );
                """
            )

            # miner responses, we will use JSON for the tensor.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    miner_hotkey TEXT,
                    challenge_uid INTEGER,
                    prediction TEXT,
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
                """
            )
            conn.commit()

    def insert(
        self,
        sample: Era5Sample,
        miner_hotkeys: List[str],
        predictions: List[torch.Tensor],
    ):
        """
        Insert a challenge and responses into the database.
        """
        challenge_uid = self._insert_challenge(sample)
        self._insert_responses(challenge_uid, miner_hotkeys, predictions)
        
        # 🔥 NOWE: Szczegółowe logowanie zapisywania do bazy
        bt.logging.info(f"💾 ZAPISANO CHALLENGE DO BAZY DANYCH")
        bt.logging.info(f"   Challenge UID: {challenge_uid}")
        bt.logging.info(f"   Variable: {sample.variable}")
        bt.logging.info(f"   Miners stored: {len(miner_hotkeys)}")
        bt.logging.info(f"   Baseline shape: {list(sample.output_data.shape)}")
        
        # Pokaż kiedy challenge będzie scored
        end_time = pd.Timestamp(sample.end_timestamp, unit='s')
        era5_available_time = end_time + pd.Timedelta(days=5)  # ERA5 ma ~5 dni opóźnienia
        now = pd.Timestamp.now()
        time_to_scoring = era5_available_time - now
        
        if time_to_scoring.total_seconds() > 0:
            bt.logging.info(f"   ⏱️  ERA5 ground truth available in: {time_to_scoring}")
            bt.logging.info(f"   📅 Expected scoring time: {era5_available_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            bt.logging.info(f"   ✅ ERA5 ground truth should already be available!")

    def _insert_challenge(self, sample: Era5Sample) -> int:
        """
        Insert a sample into the database and return the challenge UID.
        Assumes the sample's output data is the baseline.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challenges (lat_start, lat_end, lon_start, lon_end, start_timestamp, end_timestamp, hours_to_predict, baseline, inserted_at, variable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    *sample.get_bbox(),
                    sample.start_timestamp,
                    sample.end_timestamp,
                    sample.predict_hours,
                    json.dumps(sample.output_data.tolist()),
                    sample.query_timestamp,
                    sample.variable
                ),
            )
            challenge_uid = cursor.lastrowid
            conn.commit()
            return challenge_uid

    def _insert_responses(
        self,
        challenge_uid: int,
        miner_hotkeys: List[str],
        predictions: List[torch.Tensor],
    ):
        """
        Insert the responses from the miners into the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            data_to_insert = []
            # prepare data for insertion
            for miner_hotkey, prediction in zip(miner_hotkeys, predictions):
                prediction_json = json.dumps(prediction.tolist())
                data_to_insert.append((miner_hotkey, challenge_uid, prediction_json))

            cursor.executemany(
                """
                INSERT INTO responses (miner_hotkey, challenge_uid, prediction)
                VALUES (?, ?, ?);
            """,
                data_to_insert,
            )
            conn.commit()

    def score_and_prune(
        self, score_func: Callable[[Era5Sample, torch.Tensor, List[str], List[torch.Tensor]], None]
    ):
        """
        Check the database for challenges and responses, and prune them if they are not needed anymore.

        If a challenge is found that should be finished, the correct output is fetched.
        Next, all miner predictions are loaded and the score_func is called with the sample, miner hotkeys and predictions.
        """
        latest_available = self.cds_loader.last_stored_timestamp.timestamp()

        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            # get all challenges that we can now score
            cursor.execute(
                """
                SELECT * FROM challenges WHERE end_timestamp <= ?;
            """,
                (latest_available,),
            )
            challenges = cursor.fetchall()

        bt.logging.info(f"🔍 Found {len(challenges)} challenges ready for scoring")

        for i, challenge in enumerate(challenges):
            # load the sample
            (
                challenge_uid,
                lat_start,
                lat_end,
                lon_start,
                lon_end,
                start_timestamp,
                end_timestamp,
                hours_to_predict,
                baseline,
                inserted_at,
                variable,
            ) = challenge

            # bt.logging.info(f"📋 Processing challenge {i+1}/{len(challenges)} (UID: {challenge_uid})")
            # bt.logging.info(f"   Variable: {variable}")
            # bt.logging.info(f"   Time: {pd.Timestamp(start_timestamp, unit='s')} -> {pd.Timestamp(end_timestamp, unit='s')}")
            # bt.logging.info(f"   Location: lat[{lat_start:.2f}, {lat_end:.2f}], lon[{lon_start:.2f}, {lon_end:.2f}]")

            sample = Era5Sample(
                variable=variable,
                query_timestamp=inserted_at,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                lat_start=lat_start,
                lat_end=lat_end,
                lon_start=lon_start,
                lon_end=lon_end,
                predict_hours=hours_to_predict,
            )
            
            # load the correct output and set it if it is available
            bt.logging.info(f"🌍 Fetching ERA5 ground truth...")
            output = self.cds_loader.get_output(sample)

            if output is None:
                days_old = (latest_available - end_timestamp) / (24 * 3600)
                bt.logging.warning(f"❌ Cannot get ERA5 ground truth (challenge {days_old:.1f} days old) - no data returned")
                
                if end_timestamp < (latest_available - pd.Timedelta(days=3).total_seconds()):
                    # challenge is unscore-able, delete it
                    bt.logging.info(f"🗑️  Deleting unscorable challenge (too old)")
                    self._delete_challenge(challenge_uid)
                continue

            sample.output_data = output

            if output.shape[0] != hours_to_predict:
                days_old = (latest_available - end_timestamp) / (24 * 3600)
                bt.logging.warning(f"❌ Cannot get ERA5 ground truth (challenge {days_old:.1f} days old) - wrong shape")
                
                if end_timestamp < (latest_available - pd.Timedelta(days=3).total_seconds()):
                    # challenge is unscore-able, delete it
                    bt.logging.info(f"🗑️  Deleting unscorable challenge (too old)")
                    self._delete_challenge(challenge_uid)
                continue

            
            bt.logging.success(f"✅ ERA5 ground truth loaded: shape {list(output.shape)}")
            
            # Pokaż statystyki ground truth
            if output.numel() == 0:
                bt.logging.warning(f"❌ ERA5 ground truth is empty (shape: {list(output.shape)})")
                bt.logging.info(f"🗑️  Deleting challenge with empty ground truth")
                self._delete_challenge(challenge_uid)
                continue
                
            gt_stats = {
                "mean": output.mean().item(),
                "std": output.std().item(), 
                "min": output.min().item(),
                "max": output.max().item()
            }
            bt.logging.info(f"📏 ERA5 Ground Truth Stats: mean={gt_stats['mean']:.4f}, std={gt_stats['std']:.4f}, range=[{gt_stats['min']:.4f}, {gt_stats['max']:.4f}]")
        
            baseline = torch.tensor(json.loads(baseline))
            bt.logging.info(f"🌐 Baseline shape: {list(baseline.shape)}")
            
            # load the miner predictions
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM responses WHERE challenge_uid = ?;
                """,
                    (challenge_uid,),
                )
                responses = cursor.fetchall()

                miner_hotkeys = [response[0] for response in responses]
                predictions = [
                    torch.tensor(json.loads(response[2])) for response in responses
                ]
            
            bt.logging.info(f"⛏️  Loaded {len(predictions)} miner predictions from database")
            
            # Pokaż podstawowe statystyki predykcji przed scoringiem
            for j, (hotkey, prediction) in enumerate(zip(miner_hotkeys, predictions)):
                pred_stats = {
                    "mean": prediction.mean().item(),
                    "std": prediction.std().item(),
                    "min": prediction.min().item(),
                    "max": prediction.max().item()
                }
                bt.logging.info(f"   Miner {j+1}: shape={list(prediction.shape)}, mean={pred_stats['mean']:.4f}")
            
            # don't score while database is open in case there is a metagraph delay.
            bt.logging.info(f"🎯 Starting final scoring process...")
            score_func(sample, baseline, miner_hotkeys, predictions)
            
            bt.logging.success(f"✅ Challenge {challenge_uid} scored and deleted")
            self._delete_challenge(challenge_uid)

            # don't score miners too quickly in succession and always wait after last scoring
            if i < len(challenges) - 1:  # Nie czekaj po ostatnim
                bt.logging.info(f"⏳ Waiting 4s before next challenge...")
                time.sleep(4)

    def _delete_challenge(self, challenge_uid: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # prune the challenge and the responses
            cursor.execute(
                """
                DELETE FROM challenges WHERE uid = ?;
            """,
                (challenge_uid,),
            )
            cursor.execute(
                """
                DELETE FROM responses WHERE challenge_uid = ?;
            """,
                (challenge_uid,),
            )
            conn.commit()

    def prune_hotkeys(self, hotkeys: List[str]):
        """
        Prune the database of hotkeys that are no longer participating.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM responses WHERE miner_hotkey IN ({});
                """.format(','.join('?' for _ in hotkeys)),
                hotkeys
            )
            conn.commit()


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns