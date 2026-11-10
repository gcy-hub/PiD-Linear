"""A failed encoder must be detected while its peers are still running."""
import subprocess
import sys
import time
from .cache_80k import wait_workers


def main():
    children=[subprocess.Popen([sys.executable,'-c',code]) for code in
              ['import time; time.sleep(30)','raise SystemExit(1)']]
    start=time.monotonic()
    try:
        codes=wait_workers(children)
        assert 1 in codes and time.monotonic()-start<5
        assert all(p.poll() is not None for p in children)
        children=[subprocess.Popen([sys.executable,'-c','pass']) for _ in range(2)]
        assert wait_workers(children)==[0,0]
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
            p.wait()
    print('Immediate worker failure detection and successful completion passed')


if __name__=='__main__':main()
