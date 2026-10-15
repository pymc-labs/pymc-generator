"""Publication adapter for disjoint, deterministic streaming task ownership.

Avoid a full run scan for every ten-world write. The orchestrator must ensure a
single attempt and disjoint task owners. Task-prefixed output/world identities
make cross-task identity collisions impossible; resume still validates the run.
Only a worker's writer thread may use this instance while generation is running.
"""

from __future__ import annotations

from pymc_generator.publication import (
    FilePublicationStore,
    PublicationError,
    _identifier,
    _read_json,
    _validate_record,
)

from . import publication as pub


class StreamPublicationStore(FilePublicationStore):
    def __init__(self, root, volume=None):
        super().__init__(root)
        self.volume = volume
        self._writing_task = None

    def _refresh(self):
        if self.volume is not None:
            self.volume.reload()

    def _persist(self):
        if self.volume is not None:
            self.volume.commit()

    def discover(self, run_id):
        if self._writing_task is None:
            return super().discover(run_id)
        self._refresh()
        directory = self.root / "runs" / _identifier(run_id) / "tasks" / self._writing_task
        outputs, problems = [], {}
        for path in sorted(directory.glob("attempts/*/completion.json")):
            reference = path.relative_to(self.root).as_posix()
            try:
                record = _read_json(path)
                _validate_record(record)
                if reference != self._reference(record) or record["run_id"] != run_id:
                    raise PublicationError("record identity does not match location")
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems[reference] = str(error)
            else:
                outputs.append({"reference": reference, "record": record})
        return outputs, problems

    def publish(self, metadata, corpus):
        task = metadata["task_id"]
        pub.task_index_of(task)
        if metadata["output_id"] != task or metadata["instruction_id"] != task:
            raise ValueError("streaming output identity must equal its task")
        if metadata["world_ids"] != [
            pub.completion_world_id(task, i) for i in range(metadata["counts"]["published"])
        ]:
            raise ValueError("streaming worlds must use task-prefixed row identities")
        if self._writing_task is not None:
            raise RuntimeError("only one writer may use a publication instance")
        self._writing_task = task
        try:
            return super().publish(metadata, corpus)
        finally:
            self._writing_task = None
