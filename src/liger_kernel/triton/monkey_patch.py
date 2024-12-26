import os
import random

from triton.runtime.cache import FileCacheManager


class LigerTritonFileCacheManager(FileCacheManager):
    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = random.randint(0, 1000000)
        # we use the PID incase a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        os.removedirs(temp_dir)
        return filepath


def apply_liger_triton_cache_manager():
    """
    Experimental feature to get around transient FileNotFoundError in triton compilation.
    For more details please see https://github.com/triton-lang/triton/pull/4295
    """
    os.environ["TRITON_CACHE_MANAGER"] = "liger_kernel.triton.monkey_patch:LigerTritonFileCacheManager"
