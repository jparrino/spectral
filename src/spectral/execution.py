#
# SpectRAL - Spectral reconstruction analysis toolkit
# Copyright (C) 2026 Julian Parrino (https://github.com/jparrino)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
#

"""One explicit worker budget shared by numerical operations."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import io
import pickle
import mpmath as mp
from threadpoolctl import threadpool_limits


def _initialize_worker(threads):
    global _thread_limit
    _thread_limit = threadpool_limits(limits=threads)


@dataclass(frozen=True)
class Outcome:
    value: object = None
    error: str | None = None


def _invoke(task):
    function, args = task
    try:
        return Outcome(function(*args))
    except Exception as exc:
        return Outcome(error=f"{type(exc).__name__}: {exc}")


def _restore_mpf(value):
    return mp.make_mpf(value)


def _restore_mpc(value):
    return mp.make_mpc(value)


class _ExactPickler(pickle.Pickler):
    """Override context-sensitive mpmath pickle reducers, including nested values."""

    def reducer_override(self, value):
        if isinstance(value, mp.mpf):
            return _restore_mpf, (tuple(int(x) for x in value._mpf_),)
        if isinstance(value, mp.mpc):
            return _restore_mpc, (
                tuple(tuple(int(x) for x in part) for part in value._mpc_),
            )
        return NotImplemented


def _pack(value):
    buffer = io.BytesIO()
    _ExactPickler(buffer, protocol=5).dump(value)
    return buffer.getvalue()


def _invoke_packed(payload):
    # Both directions need exact transport: workers exit their numerical context
    # before returning, and the parent's ambient context may also be low precision.
    return _pack(_invoke(pickle.loads(payload)))


class Serial:
    workers = 1

    def map(self, tasks):
        return [_invoke(task) for task in tasks]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Processes:
    def __init__(self, workers="auto", *, threads_per_worker=1):
        if workers == "auto":
            workers = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
        if (
            not isinstance(workers, int)
            or workers < 1
            or not isinstance(threads_per_worker, int)
            or threads_per_worker < 1
        ):
            raise ValueError("worker and thread counts must be positive integers")
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self._pool = None

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError("execution context is already active")
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(self.threads_per_worker,),
        )
        return self

    def map(self, tasks):
        if self._pool is None:
            raise RuntimeError("use processes() inside a with block")
        try:
            # map keeps input ordering; worker scheduling never determines coordinates.
            return [
                pickle.loads(payload)
                for payload in self._pool.map(_invoke_packed, map(_pack, tasks))
            ]
        except Exception as exc:
            raise RuntimeError(
                "parallel execution failed; custom functions must be importable and picklable"
            ) from exc

    def __exit__(self, *args):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pool = None


def processes(workers="auto", *, threads_per_worker=1):
    return Processes(workers, threads_per_worker=threads_per_worker)


def serial():
    return Serial()
