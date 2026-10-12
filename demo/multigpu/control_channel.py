"""Private CPU control sockets; idle sessions do not wait in NCCL collectives."""
import os
import pickle
import socket
import struct
import time


class ControlChannel:
    def __init__(self, path, rank, world_size, timeout=1800):
        self.rank, self.connections = rank, {}
        self.timeout = timeout
        if rank == 0:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.settimeout(timeout)
                listener.bind(path)
                os.chmod(path, 0o600)
                listener.listen(world_size-1)
                for _ in range(world_size-1):
                    connection, _ = listener.accept()
                    connection.settimeout(timeout)
                    peer = self._receive(connection)
                    if peer not in range(1, world_size) or peer in self.connections:
                        connection.close()
                        raise ValueError('Invalid or duplicate peer rank')
                    self.connections[peer] = connection
        else:
            self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            deadline = time.monotonic()+timeout
            while True:
                try:
                    self.connection.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline: raise
                    time.sleep(0.1)
            self._send(self.connection, rank)
            # Intentional indefinite CPU idle wait; torchrun supervises failures.
            self.connection.settimeout(None)

    @staticmethod
    def _send(connection, value):
        payload = pickle.dumps(value, protocol=5)
        if len(payload) > 16*1024**2: raise ValueError('Control message too large')
        connection.sendall(struct.pack('!I', len(payload))+payload)

    @staticmethod
    def _receive(connection):
        def read(count):
            chunks = bytearray()
            while len(chunks) < count:
                data = connection.recv(count-len(chunks))
                if not data: raise EOFError('GPU control peer disconnected')
                chunks.extend(data)
            return bytes(chunks)
        size, = struct.unpack('!I', read(4))
        if size > 16*1024**2: raise ValueError('Control message too large')
        return pickle.loads(read(size))

    def message(self, value=None):
        if self.rank == 0:
            for rank in sorted(self.connections): self._send(self.connections[rank], value)
            return value
        return self._receive(self.connection)

    def reply(self, value):
        if self.rank == 0:
            return [value]+[self._receive(self.connections[r]) for r in sorted(self.connections)]
        self._send(self.connection, value)

    def close(self):
        if self.rank == 0:
            for connection in self.connections.values(): connection.close()
        else: self.connection.close()
