import shlex
import subprocess
from pathlib import Path
root = Path.home() / 'qs3'
cmd = shlex.split(subprocess.check_output(['ninja', '-t', 'commands', 'qsfi.o'], cwd=root/'build', text=True).strip().splitlines()[-1])
cmd[cmd.index('../qsfi.cu')] = '../.prototypes/moe_sweep/qsfi.cu'
cmd[cmd.index('-o')+1] = '../.prototypes/moe_sweep/qsfi.o'
cmd.append('-I..')
subprocess.run(cmd, cwd=root/'build', check=True)
cmd = ['nvcc', '-std=c++17', '-arch=sm_121', '--expt-relaxed-constexpr', '-DQSFI_ENABLE_CHECKED_VALIDATION=0', '-I..', '../.prototypes/moe_sweep/sweep.cu', '../.prototypes/moe_sweep/qsfi.o', 'qscu_gdn.o', 'qscu.o', 'qscb.o', '-o', '../.prototypes/moe_sweep/sweep', '-lcuda', '-lcublas', '-lcublasLt']
subprocess.run(cmd, cwd=root/'build', check=True)
