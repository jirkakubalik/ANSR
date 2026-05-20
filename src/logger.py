import os
import time
from colorama import Fore, Back, Style, init

init(autoreset=True)


def align_text(message: str, start_time: float = None) -> str:
    timestamp = f"[{time.time() - start_time:7.2f}] " if start_time else ""
    indent = " " * len(timestamp)

    lines = message.strip().splitlines()
    formatted_lines = [f"{timestamp}{lines[0]}"] + [
        f"{indent}{line}" for line in lines[1:]
    ]

    full_msg = "\n".join(formatted_lines)
    return full_msg


def log(
    message: str,
    start_time: float = None,
    file_path=None,
    level="INFO",
    flush_file: bool = False,
):
    """Prints aligned multi-line log messages either to file or standard output."""

    COLORS = {
        "INFO": Fore.WHITE,
        "HIGHLIGHT_INFO": Back.GREEN + Fore.WHITE,
        "INFO_MAGENTA": Back.LIGHTMAGENTA_EX + Fore.BLACK,
        "WARNING": Back.YELLOW + Fore.BLACK,
        "HIGHLIGHT_GREEN": Back.GREEN + Fore.BLACK,
        "HIGHLIGHT_RED": Back.RED + Fore.BLACK,
        "HIGHLIGHT_CYAN": Back.LIGHTCYAN_EX + Fore.BLACK,
        "HIGHLIGHT_MAGENTA": Back.LIGHTMAGENTA_EX + Fore.BLACK,
        "DEBUG": Back.RED + Fore.BLACK,
        "ERROR": Back.RED + Fore.BLACK,
    }

    if "INFO" in level.upper():
        # if True:
        full_msg = align_text(message, start_time)
        print(f"{COLORS.get(level.upper(), '')}{full_msg}{Style.RESET_ALL}", flush=True)

        if file_path:
            with open(file_path, "a") as f:
                f.write(full_msg + "\n")
                if flush_file:
                    f.flush()
                    os.fsync(f.fileno())


def log_population_state(
    population,
    start_time,
    title,
    file_path=None,
    tick_time=None,
    tick_backprops=None,
    tick_ids=None,
):
    """Logs the current state of a population."""
    header_line = f"{title}:"
    header_line += (
        f" tick_time={time.time() - tick_time:7.2f}," if tick_time is not None else ""
    )
    header_line += (
        f" tick_backprops={tick_backprops}," if tick_backprops is not None else ""
    )
    header_line += f" tick_ids={tick_ids}" if tick_ids is not None else ""
    lines = [header_line]
    for i, ind in enumerate(population):
        perf = list(ind.perf_history.values())[-1]
        lines.append(
            f"{i}.\t id: {ind.id}, age: {ind.age}, backprop_iters: {ind.backprop_iters}, "
            f"valid_loss: {round(perf.performance['valid_loss'], 9)}, "
            f"rmse_constr: {round(perf.performance['rmse_constr'], 9)}, "
            f"complexity: {perf.performance['complexity']}, "
            f"subjectToTune: {ind.subjectToTune}"
        )
    lines.append("   ")

    message = "\n".join(lines)
    log(message, start_time, file_path=file_path, level="INFO")
