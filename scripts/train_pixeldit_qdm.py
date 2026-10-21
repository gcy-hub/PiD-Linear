"""Prompt-only PixelDiT-T2I, fixed quadrant memory and local Halo2."""
import sys
from pid._src.vdn_pid.training import main

DEFAULTS=['--backbone','pixeldit_t2i','--resolution','1024','--flow-shift','4','--shift-base-size','1024',
          '--memory-mode','fixed4','--local-mode','halo','--local-halo','2',
          '--region-size','8','--output-root','outputs/pixeldit-t2i-qdm-halo2','--grad-accum','1']
if __name__=='__main__':main(DEFAULTS+sys.argv[1:])
