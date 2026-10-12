# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import contextlib
import ipaddress
import logging
import os
import socket

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str) -> tuple[int, socket.socket]:
    family = socket.AF_INET
    if is_valid_ipv6_address(address):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))

    port = sock.getsockname()[1]
    return port, sock


class _EmbeddedServer(uvicorn.Server):
    """uvicorn server running inside a host process (a Ray actor) that owns signal handling."""

    @contextlib.contextmanager
    def capture_signals(self):  # uvicorn >= 0.29
        yield

    def install_signal_handlers(self):  # uvicorn < 0.29
        pass


async def _serve_until_started(server: uvicorn.Server) -> asyncio.Task:
    """Run ``server`` as a background task and return once it is accepting connections.

    This used to set ``should_exit = True`` and await ``serve()``, relying on uvicorn returning
    right after startup with the listening sockets left open. uvicorn >= 0.41 calls ``shutdown()``
    on that path, which closes them: every vLLM server then looked healthy but refused all
    connections. Waiting on ``server.started`` works on every version.
    """

    async def _serve():
        try:
            await server.serve()
        except SystemExit as e:
            # uvicorn calls sys.exit(1) when it cannot bind. Inside a task, SystemExit would
            # escape to the event loop and take the whole actor down; surface it as an OSError
            # so the caller can retry on another port.
            raise OSError(f"uvicorn exited during startup (code {e.code})") from e

    task = asyncio.create_task(_serve())
    while not server.started:
        if task.done():
            task.result()  # re-raises the startup error
            raise OSError("uvicorn stopped before it started serving")
        await asyncio.sleep(0.05)
    return task


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        server_port, sock = get_free_port(server_address)
        try:
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server_task = await _serve_until_started(_EmbeddedServer(config))
            break
        except OSError as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
        finally:
            # The placeholder only reserves the port until uvicorn has bound it.
            sock.close()
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task
