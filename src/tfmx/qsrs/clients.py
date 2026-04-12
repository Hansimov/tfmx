"""Production multi-machine QSR client."""

from .clients_core import _QSRClientsBase, _QSRClientsPipeline


class QSRClients(_QSRClientsBase):
    def __init__(self, endpoints: list[str], verbose: bool = False):
        self._verbose = verbose
        super().__init__(endpoints)
        self._pipeline = _QSRClientsPipeline(machine_scheduler=self.machine_scheduler)

    def __repr__(self) -> str:
        healthy = sum(1 for machine in self.machines if machine.healthy)
        return f"QSRClients(machines={len(self.machines)}, healthy={healthy})"
