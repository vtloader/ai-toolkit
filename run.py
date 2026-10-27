import os
import sys
import ctypes
from pathlib import Path
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()
# Default to the Hugging Face mirror; an explicit HF_ENDPOINT (environment or
# .env) still takes precedence. Must run before transformers/huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "0")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())

# The UI launches jobs with no console; keep anything we shell out to (torch
# compiles, HF git downloads) from flashing a console window. Must come before
# any import that might spawn a subprocess.
from toolkit.win_console import suppress_child_consoles
suppress_child_consoles()

# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# On Linux/aarch64, torch_npu loads libraries that consume most of the static
# TLS slots before OpenCV and transformers import their native dependencies.
# Loading libGLdispatch or scikit-learn's bundled libgomp afterwards can fail
# with "cannot allocate memory in static TLS block".  Load both libraries first
# so they can coexist in the job process.
def _preload_native_runtime_libraries():
    if not sys.platform.startswith('linux'):
        return

    for path in (
        '/lib/aarch64-linux-gnu/libGLdispatch.so.0',
        '/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0',
    ):
        if not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        break

    for path_entry in sys.path:
        if not path_entry:
            continue
        candidates = sorted(Path(path_entry).glob('scikit_learn.libs/libgomp-*.so.*'))
        if not candidates:
            continue
        try:
            ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        return


_preload_native_runtime_libraries()

# NPU throughput defaults: the Ascend task queue keeps kernel launches
# asynchronous, and COMBINED_ENABLE fuses adjacent small ops. Both must be set
# before torch_npu initializes. Opt out with AITK_NPU_PERF=0.
if os.environ.get("AITK_NPU_PERF", "1") != "0":
    try:
        import importlib.util

        if importlib.util.find_spec("torch_npu") is not None:
            os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
            os.environ.setdefault("COMBINED_ENABLE", "1")
    except Exception:
        pass

# set torch to trace mode
import torch
    
# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

# on unified memory devices (DGX Spark), offloading modules to cpu is useless and very slow
from toolkit.util.unified_memory import apply_unified_memory_patches
apply_unified_memory_patches()

# Tensor parallel: torchrun launches one process per NPU and exposes RANK /
# LOCAL_RANK / WORLD_SIZE. Bind this process to its chip and initialize HCCL
# before any model or Accelerator is created.
if os.environ.get("AITK_TP", "0") == "1":
    import numpy as _np
    import random as _random
    import torch.distributed as _dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "npu"):
        torch.npu.set_device(local_rank)
    if not _dist.is_initialized():
        _dist.init_process_group(backend="hccl")
    tp_world = _dist.get_world_size()
    tp_rank = _dist.get_rank()
    os.environ.setdefault("AITK_TP_SIZE", str(tp_world))
    os.environ["AITK_TP_RANK"] = str(tp_rank)
    # Every rank must see the same batch and noise for TP to be well defined.
    tp_seed = int(os.environ.get("SEED", "42"))
    os.environ.setdefault("SEED", str(tp_seed))
    _random.seed(tp_seed)
    _np.random.seed(tp_seed)
    torch.manual_seed(tp_seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(tp_seed)
    if tp_rank != 0:
        _TP_LOG = open(f"/tmp/aitk_tp_rank{tp_rank}.log", "w", buffering=1)
        sys.stdout = _TP_LOG
        sys.stderr = _TP_LOG
    else:
        print(f"Tensor parallel: {tp_world} ranks, rank {tp_rank} on npu:{local_rank}")

if seed is not None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        job = None
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            if job is not None:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            if job is not None:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_acc("")
                print_acc("========================================")
                print_acc("Job stopped")
                print_acc("========================================")
                sys.exit(0)


if __name__ == '__main__':
    main()
