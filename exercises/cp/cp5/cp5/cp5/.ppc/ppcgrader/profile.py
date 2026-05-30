from typing import Dict, Optional, Callable, Union
from dataclasses import dataclass

from ppcgrader.quantity import Quantity, Unit
from ppcgrader.doc_builder import DocumentBuilder, Document, ListBuilder, strong
from ppcgrader.info_utils import get_grader_domain

CACHE_LINE = 64


@dataclass
class ProfileData:
    exclude_kernel: bool = None
    kernel_time_warning: bool = None
    generic_cache: bool = None
    wallclock: Quantity = None
    cpu_time: Quantity = None
    sys_time: Quantity = None
    num_threads: Quantity = None
    cycles: Quantity = None
    freq: Quantity = None
    instructions: Quantity = None
    ipc: Quantity = None
    instr_per_wall: Quantity = None
    instr_per_cpu: Quantity = None
    branches: Quantity = None
    branch_pct: Quantity = None
    branch_misses: Quantity = None
    branch_miss_pct: Quantity = None
    sys_pct: Quantity = None
    l3_refs: Quantity = None
    l3_misses: Quantity = None
    l3_miss_pct: Quantity = None
    l3_read: Quantity = None
    l3_read_rate: Quantity = None
    ram_read: Quantity = None
    ram_read_rate: Quantity = None
    l1_refs: Quantity = None
    l1_misses: Quantity = None
    l1_miss_pct: Quantity = None
    l2_read: Quantity = None
    l2_read_rate: Quantity = None
    l1_inst_per_access: Quantity = None
    page_faults: Quantity = None
    page_fault_rate: Quantity = None
    context_switches: Quantity = None
    context_switch_rate: Quantity = None
    cpu_migrations: Quantity = None
    cpu_migration_rate: Quantity = None

    # task-specific quantities
    operations: Quantity = None
    operations_name: str = None
    ops_per_sec: Quantity = None
    instr_per_op: Quantity = None


def optional_binary_op(a: Optional[float], b: Optional[float],
                       op: Callable[[float, float], float]) -> Optional[float]:
    if a is None or b is None:
        return None

    return op(a, b)


def optional_div(num: Optional[float],
                 den: Optional[float]) -> Optional[float]:
    if den == 0:
        return None
    return optional_binary_op(num, den, lambda x, y: x / y)


def optional_gibi(num: Optional[float]) -> Optional[float]:
    return optional_div(num, 1024 * 1024 * 1024)


def optional_product(num: Optional[float],
                     den: Optional[float]) -> Optional[float]:
    return optional_binary_op(num, den, lambda x, y: x * y)


def optional_percent(num: Optional[float], den: Optional[float]) -> Quantity:
    if num is None or den is None or den == 0:
        return Quantity(None, Unit.Percent)
    else:
        return Quantity(100 * num / den, Unit.Percent)


def generate_derived_statistics(stat: Dict[str, float]) -> ProfileData:
    wallclock = stat.get('perf_wall_clock_ns', None)
    enabled = stat.get('perf_time_enabled_ns', None)
    running = stat.get('perf_time_running_ns', None)
    usr_time = stat.get('perf_time_usr_ns', None)
    sys_time = stat.get('perf_time_sys_ns', None)
    instrs = stat.get('perf_instructions', None)
    cycles = stat.get('perf_cycles', None)
    branches = stat.get('perf_branches', None)
    branch_misses = stat.get('perf_branch_misses', None)

    if wallclock is None or wallclock == 0:
        return ProfileData()

    wallclock_secs = optional_div(wallclock, 1e9)
    exclude_kernel = stat.get('perf_exclude_kernel', True)

    if usr_time is not None and sys_time is not None:
        total_time = usr_time + sys_time
        if exclude_kernel:
            running = usr_time
    elif running is not None:
        total_time = running
    else:
        # one last try: this might be an old record, where the total running time
        # is recorded in perf_cpu_time_ns
        running = stat.get('perf_cpu_time_ns', None)
        total_time = running

    result = ProfileData()
    result.exclude_kernel = exclude_kernel
    result.kernel_time_warning = False
    result.wallclock = Quantity(wallclock_secs, Unit.Seconds)
    result.cpu_time = Quantity(optional_div(total_time, 1e9), Unit.Seconds)
    result.sys_time = Quantity(optional_div(sys_time, 1e9), Unit.Seconds)
    result.num_threads = Quantity(optional_div(total_time, wallclock),
                                  Unit.Count)
    result.cycles = Quantity(cycles, Unit.Event)
    result.freq = Quantity(optional_div(cycles, optional_div(running, 1e9)),
                           Unit.Hertz)
    result.instructions = Quantity(instrs, Unit.Event)
    result.ipc = Quantity(optional_div(instrs, cycles), Unit.EventRate)
    result.instr_per_wall = Quantity(optional_div(instrs, wallclock),
                                     Unit.EventRate)
    result.instr_per_cpu = Quantity(optional_div(instrs, running),
                                    Unit.EventRate)
    result.branches = Quantity(branches, Unit.Count)
    result.branch_pct = optional_percent(branches, instrs)
    result.branch_misses = Quantity(branch_misses, Unit.Count)
    result.branch_miss_pct = optional_percent(branch_misses, branches)

    # only record sys_pct if there is a reasonably large fraction
    if sys_time is not None and total_time is not None and sys_time > 0.01 * total_time and sys_time >= 1e7:
        result.sys_pct = optional_percent(sys_time, total_time)
        result.kernel_time_warning = exclude_kernel
    else:
        result.sys_pct = Quantity(None, Unit.Percent)

    generic_cache = False
    l3_refs = stat.get('perf_l3_read_refs', None)
    l3_misses = stat.get('perf_l3_read_misses', None)
    if l3_refs is None:
        # the generic cache events. their meaning is a bit diffuse, so they are only here as a fallback
        l3_refs = stat.get('perf_cache_refs', None)
        l3_misses = stat.get('perf_cache_misses', None)
        if l3_refs is not None:
            generic_cache = True

    result.generic_cache = generic_cache
    read_bytes = Quantity(optional_product(l3_refs, CACHE_LINE), Unit.Bytes)
    miss_bytes = Quantity(optional_product(l3_misses, CACHE_LINE), Unit.Bytes)

    result.l3_refs = Quantity(l3_refs, Unit.Event)
    result.l3_misses = Quantity(l3_misses, Unit.Event)
    result.l3_miss_pct = optional_percent(l3_misses, l3_refs)
    result.l3_read = read_bytes
    result.l3_read_rate = Quantity(
        optional_div(read_bytes.value, wallclock_secs), Unit.BytesPerSecond)
    result.ram_read = miss_bytes
    result.ram_read_rate = Quantity(
        optional_div(miss_bytes.value, wallclock_secs), Unit.BytesPerSecond)

    l1_refs = stat.get('perf_l1_read_refs', None)
    l1_misses = stat.get('perf_l1_read_misses', None)

    miss_bytes = optional_product(l1_misses, CACHE_LINE)
    result.l1_refs = Quantity(l1_refs, Unit.Event)
    result.l1_misses = Quantity(l1_misses, Unit.Event)
    result.l1_miss_pct = optional_percent(l1_misses, l1_refs)
    result.l2_read = Quantity(miss_bytes, Unit.Bytes)
    result.l2_read_rate = Quantity(optional_div(miss_bytes, wallclock_secs),
                                   Unit.BytesPerSecond)
    result.l1_inst_per_access = Quantity(optional_div(instrs, l1_refs),
                                         Unit.EventRate)

    page_faults = stat.get('perf_page_faults', None)
    result.page_faults = Quantity(page_faults, Unit.Event)
    result.page_fault_rate = Quantity(
        optional_div(page_faults, wallclock_secs), Unit.EventRate)

    context_switches = stat.get('perf_context_switches', None)
    result.context_switches = Quantity(context_switches, Unit.Event)
    result.context_switch_rate = Quantity(
        optional_div(context_switches, wallclock_secs), Unit.EventRate)

    cpu_migrations = stat.get('perf_cpu_migrations', None)
    result.cpu_migrations = Quantity(cpu_migrations, Unit.Event)
    result.cpu_migration_rate = Quantity(
        optional_div(cpu_migrations, wallclock_secs), Unit.EventRate)

    ops = stat.get('operations', None)
    result.operations_name = stat.get('operations_name', None)
    result.operations = Quantity(ops, Unit.Count)
    result.ops_per_sec = Quantity(optional_div(ops, wallclock_secs),
                                  Unit.EventRate)
    result.instr_per_op = Quantity(optional_div(instrs, ops), Unit.EventRate)
    return result


def explain_task(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    if s.operations_name == "useful arithmetic operation":
        with lst.item() as item:
            item += grader.gettext(
                "profile.task.useful.expect",
                operations=s.operations,
            )
            if lst.mode == "term":
                item += grader.gettext(
                    "profile.task.useful.throughput.term",
                    ops_per_sec=s.ops_per_sec,
                )
            else:
                item += grader.gettext(
                    "profile.task.useful.throughput.web.prefix",
                    wallclock=f"{s.wallclock:.2f}",
                )
                item += grader.gettext(
                    "profile.task.useful.throughput.web.mid", )
                item += strong(s.ops_per_sec)
                item += grader.gettext(
                    "profile.task.useful.throughput.web.suffix", )

    if s.operations_name == "rectangle evaluation":
        with lst.item() as item:
            item += grader.gettext(
                "profile.task.rectangle.expect",
                operations=s.operations,
            )
            if lst.mode == "term":
                item += grader.gettext(
                    "profile.task.rectangle.throughput.term",
                    ops_per_sec=s.ops_per_sec,
                )
            else:
                item += grader.gettext(
                    "profile.task.rectangle.throughput.web.prefix",
                    wallclock=f"{s.wallclock:.2f}",
                )
                item += grader.gettext(
                    "profile.task.rectangle.throughput.web.mid", )
                item += strong(s.ops_per_sec)
                item += grader.gettext(
                    "profile.task.rectangle.throughput.web.suffix", )


def localized_operations_name(operations_name: Optional[str],
                              grader) -> Optional[str]:
    if operations_name == "useful arithmetic operation":
        key = "profile.operation.useful_arithmetic"
        translated = grader.gettext("profile.operation.useful_arithmetic")
        return operations_name if translated == key else translated
    if operations_name == "rectangle evaluation":
        key = "profile.operation.rectangle_evaluation"
        translated = grader.gettext("profile.operation.rectangle_evaluation")
        return operations_name if translated == key else translated
    return operations_name


def explain_time(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    with lst.item() as item:
        wallclock_str = f"{s.wallclock:.3f}"
        cpu_time_str = f"{s.cpu_time:.3f}"
        num_threads_str = f"{s.num_threads:.1f}"

        item += grader.gettext(
            "profile.time.usage.prefix",
            wallclock=wallclock_str,
            cpu_time=cpu_time_str,
        )
        if lst.mode == "web":
            item += grader.gettext("profile.time.usage.web_mid", )
        else:
            item += grader.gettext("profile.time.usage.term_mid", )
        item += strong(num_threads_str)
        item += grader.gettext("profile.time.usage.suffix", )
        if s.sys_pct and s.sys_pct > 1:
            sys_time_str = f"{s.sys_time:.3f}"
            sys_pct_str = f"{s.sys_pct}"
            if lst.mode == "web":
                item += grader.gettext(
                    "profile.time.sys_share.web",
                    sys_time=sys_time_str,
                    sys_pct=sys_pct_str,
                )
            else:
                item += grader.gettext(
                    "profile.time.sys_share.term",
                    sys_time=sys_time_str,
                    sys_pct=sys_pct_str,
                )
    if s.kernel_time_warning:
        with lst.item() as item:
            if lst.mode == "web":
                item += grader.gettext(
                    "profile.time.kernel_warning.prefix.web", )
            else:
                item += grader.gettext(
                    "profile.time.kernel_warning.prefix.term", )
            item += grader.gettext("profile.time.kernel_warning.body", )
            item += strong(
                grader.gettext(
                    "profile.time.kernel_warning.inaccurate_label", ))
            item += grader.gettext(
                "profile.time.kernel_warning.inaccurate_suffix", )


def explain_freq(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    with lst.item() as item:
        cycles_str = f"{s.cycles}"
        freq_str = f"{s.freq}"

        item += grader.gettext(
            "profile.freq.prefix",
            cycles=cycles_str,
        )
        if lst.mode == "web":
            item += grader.gettext("profile.freq.web_mid", )
        else:
            item += grader.gettext("profile.freq.term_mid", )
        item += strong(freq_str)
        item += grader.gettext("profile.freq.suffix", )


def explain_inst(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)
    operations_name = localized_operations_name(s.operations_name, grader)

    with lst.item() as item:
        instructions_str = f"{s.instructions}"
        instr_per_wall_str = f"{s.instr_per_wall}"
        instr_per_cpu_str = f"{s.instr_per_cpu}"
        ipc_str = f"{s.ipc}"

        item += grader.gettext(
            "profile.inst.summary.header",
            instructions=instructions_str,
            instr_per_wall=instr_per_wall_str,
            instr_per_cpu=instr_per_cpu_str,
        )
        item += grader.gettext("profile.inst.summary.ipc.prefix", )
        item += strong(ipc_str)
        item += grader.gettext("profile.inst.summary.ipc.suffix", )
        if operations_name is not None and lst.mode == "term":
            item += grader.gettext("profile.inst.per_operation.term.prefix", )
            item += strong(f"{s.instr_per_op}")
            item += grader.gettext(
                "profile.inst.per_operation.term.suffix",
                operations_name=operations_name,
            )

    if operations_name is not None and lst.mode == "web":
        with lst.item() as item:
            item += grader.gettext("profile.inst.per_operation.web.prefix", )
            item += strong(s.instr_per_op)
            item += grader.gettext(
                "profile.inst.per_operation.web.suffix",
                operations_name=operations_name,
            )

    with lst.item() as item:
        branch_pct_str = f"{s.branch_pct}"
        item += grader.gettext(
            "profile.branch.summary.prefix",
            branch_pct=branch_pct_str,
        )
        item += grader.gettext("profile.branch.summary.middle", )
        item += strong(s.branch_miss_pct)
        item += grader.gettext("profile.branch.summary.suffix", )


def explain_switches(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    with lst.item() as item:
        item += grader.gettext(
            "profile.switches.summary",
            cpu_migrations=s.cpu_migrations,
            cpu_migration_rate=s.cpu_migration_rate,
            context_switches=s.context_switches,
            context_switch_rate=s.context_switch_rate,
        )


def explain_page_faults(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    with lst.item() as item:
        item += grader.gettext(
            "profile.page_faults.prefix",
            page_faults=s.page_faults,
        )
        item += grader.gettext("profile.page_faults.middle", )
        item += strong(s.page_fault_rate)
        item += grader.gettext("profile.page_faults.suffix", )


def explain_cache(s: ProfileData, lst: ListBuilder):
    grader = get_grader_domain(lst.mode)

    with lst.item() as item:
        item += grader.gettext(
            "profile.cache.l3.header",
            l3_refs=s.l3_refs,
        )
        if lst.mode == "web":
            item += grader.gettext("profile.cache.l3.miss.web_prefix", )
        else:
            item += grader.gettext("profile.cache.l3.miss.term_prefix", )
        item += strong(s.l3_miss_pct)
        item += grader.gettext("profile.cache.l3.miss.suffix", )
        if lst.mode == "web":
            item += grader.gettext("profile.cache.l3.bytes.web_prefix", )
        else:
            item += grader.gettext("profile.cache.l3.bytes.term_prefix", )
        item += strong(s.l3_read)
        item += grader.gettext(
            "profile.cache.l3.bytes.middle",
            l3_read_rate=s.l3_read_rate,
        )
        item += grader.gettext("profile.cache.l3.bytes.l3l2_suffix", )
        if lst.mode == "web":
            item += grader.gettext("profile.cache.l3.ram.web_prefix", )
        else:
            item += grader.gettext("profile.cache.l3.ram.term_prefix", )
        item += strong(s.ram_read)
        item += grader.gettext(
            "profile.cache.l3.ram.middle",
            ram_read_rate=s.ram_read_rate,
        )
        item += grader.gettext("profile.cache.l3.ram_suffix", )

    if s.generic_cache:
        with lst.item() as item:
            item += grader.gettext("profile.cache.generic.header", )
            item += grader.gettext("profile.cache.generic.footer", )

    with lst.item() as item:
        item += grader.gettext(
            "profile.cache.l1.header",
            l1_refs=s.l1_refs,
        )
        if lst.mode == "web":
            item += grader.gettext("profile.cache.l1.miss.web_prefix", )
        else:
            item += grader.gettext("profile.cache.l1.miss.term_prefix", )
        item += strong(s.l1_miss_pct)
        item += grader.gettext("profile.cache.l1.miss.suffix", )
        if lst.mode == "web":
            item += grader.gettext("profile.cache.l1.bytes.web_prefix", )
        else:
            item += grader.gettext("profile.cache.l1.bytes.term_prefix", )
        item += strong(s.l2_read)
        item += grader.gettext(
            "profile.cache.l1.bytes.middle",
            l2_read_rate=s.l2_read_rate,
        )
        item += grader.gettext("profile.cache.l1.bytes.suffix", )
        item += grader.gettext("profile.cache.l1.insts.prefix", )
        item += strong(s.l1_inst_per_access)
        item += grader.gettext("profile.cache.l1.insts.suffix", )


def explain_profiling(stat: ProfileData, mode: str) -> Document:
    grader = get_grader_domain(mode)

    builder = DocumentBuilder(mode)
    if stat is None or not stat.wallclock:
        with builder.text() as txt:
            txt += grader.gettext("profile.missing_timing", )
    elif stat.wallclock < 0.001:
        with builder.text() as txt:
            txt += grader.gettext("profile.too_fast.message", )
    else:
        with builder.list() as lst:
            explain_task(stat, lst)
            explain_time(stat, lst)
            explain_freq(stat, lst)
            explain_inst(stat, lst)
            explain_switches(stat, lst)
            explain_page_faults(stat, lst)
            explain_cache(stat, lst)

    return builder.build()
