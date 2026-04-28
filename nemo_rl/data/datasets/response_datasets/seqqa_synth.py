# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import random
from typing import Any, Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class SeqQASynthMCQDataset(RawDataset):
    """SeqQA synthetic biology multiple-choice dataset for GRPO.

    The source dataset stores the correct answer in ``ideal`` and three
    distractors separately. This adapter converts each row to the generic
    multiple-choice schema consumed by ``multichoice_qa_processor``.
    """

    def __init__(
        self,
        data_path: str = "hf-carbon/seqqa-synth",
        subset: Optional[str] = "v0_4x",
        split: Optional[str] = "train",
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "seqqa-synth-mcq"
        self.dataset = load_dataset_from_path(data_path, subset, split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    @staticmethod
    def _shuffle_seed(example_id: str) -> int:
        return int(hashlib.md5(example_id.encode("utf-8")).hexdigest(), 16)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        distractors = data["distractors"]
        if len(distractors) != 3:
            raise ValueError(
                f"SeqQASynthMCQDataset expected exactly 3 distractors, got {len(distractors)}"
            )

        choices = [(str(data["ideal"]), True)] + [
            (str(distractor), False) for distractor in distractors
        ]
        rng = random.Random(self._shuffle_seed(str(data["id"])))
        rng.shuffle(choices)

        letters = ("A", "B", "C", "D")
        options = {
            letter: option
            for letter, (option, _is_correct) in zip(letters, choices, strict=True)
        }
        answer = next(
            letter
            for letter, (_option, is_correct) in zip(letters, choices, strict=True)
            if is_correct
        )

        return {
            "question": data["question"],
            "options": options,
            "answer": answer,
            "task_name": self.task_name,
            "id": data["id"],
            "subtask": data["subtask"],
            "source": data["source"],
        }
