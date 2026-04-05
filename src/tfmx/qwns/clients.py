"""Production multi-machine QWN client."""

from .clients_core import MachineScheduler, _QWNClientsBase, _QWNClientsPipeline


class QWNClients(_QWNClientsBase):
    def __init__(self, endpoints: list[str], verbose: bool = False):
        self._verbose = verbose
        super().__init__(endpoints)
        self._pipeline = _QWNClientsPipeline(machine_scheduler=self.machine_scheduler)

    def __repr__(self) -> str:
        healthy = sum(1 for machine in self.machines if machine.healthy)
        return f"QWNClients(machines={len(self.machines)}, healthy={healthy})"
