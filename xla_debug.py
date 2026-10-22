import logging
import os

import torch_xla.debug.metrics as met

LOG = logging.getLogger(__name__)


def log_xla_metrics(step):
    """Compile/execute counts and CPU-fallback (aten::) counters, for checking recompilation and fallbacks."""
    compile_n = (met.metric_data("CompileTime") or (0,))[0]
    exec_n = (met.metric_data("ExecuteTime") or (0,))[0]
    aten = {n: met.counter_value(n) for n in met.counter_names() if n.startswith("aten::")}
    LOG.info(f"XLA step {step}: CompileTime={compile_n} ExecuteTime={exec_n} aten_fallbacks={aten}")


def write_xla_report(path=None):
    path = path or os.path.join(os.getcwd(), "xla_metrics_report.txt")
    with open(path, "w") as f:
        f.write(met.metrics_report())
    LOG.info(f"Wrote XLA metrics report to {path}")
