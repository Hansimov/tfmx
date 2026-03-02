"""QVL Clients - Production-Ready Multi-Machine QVL Client

Usage:
    from tfmx.qvls.clients import QVLClients

    clients = QVLClients(["http://machine1:29800", "http://machine2:29800"])
    response = clients.chat([{"role": "user", "content": "Hello!"}])
    responses = clients.chat_batch([{"messages": [{"role": "user", "content": "Hi"}]}])
"""

from .clients_core import _QVLClientsBase, _QVLClientsPipeline, MachineScheduler


class QVLClients(_QVLClientsBase):
    """Production multi-machine QVL client.

    Distributes chat completion requests across multiple machines
    using capacity-based scheduling and health recovery.
    """

    def __init__(self, endpoints: list[str], verbose: bool = False):
        self._verbose = verbose
        super().__init__(endpoints)
        self._pipeline = _QVLClientsPipeline(
            machine_scheduler=self.machine_scheduler,
        )

    def __repr__(self) -> str:
        healthy = sum(1 for m in self.machines if m.healthy)
        return f"QVLClients(machines={len(self.machines)}, " f"healthy={healthy})"
