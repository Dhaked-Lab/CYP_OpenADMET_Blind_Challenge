from __future__ import annotations
from .registry import Registry, RegistryError


class CycleError(RegistryError):
    pass


def validate_graph(reg: Registry):
    for s in reg.specs():
        for d in s.depends_on:
            if d not in reg:
                raise RegistryError(f"step {s.id!r} depends on unknown step {d!r}")
    topo_order(reg, reg.ids())     # raises on cycle


def topo_order(reg: Registry, targets, include_deps=True) -> list[str]:
    """Deterministic topological order (registration order breaks ties) of targets + transitive deps."""
    needed, stack = set(), list(targets)
    for t in targets:
        reg.get(t)
    while stack:
        n = stack.pop()
        if n in needed:
            continue
        needed.add(n)
        if include_deps:
            stack.extend(reg.get(n).depends_on)
    order, state = [], {}
    ids = [i for i in reg.ids() if i in needed]

    def visit(n, path):
        st = state.get(n)
        if st == 2:
            return
        if st == 1:
            cyc = path[path.index(n):] + [n]
            raise CycleError("circular step dependency: " + " -> ".join(cyc))
        state[n] = 1
        for d in reg.get(n).depends_on:
            if d in needed:
                visit(d, path + [n])
        state[n] = 2
        order.append(n)

    for n in ids:
        visit(n, [])
    return order


def downstream(reg: Registry, sid: str) -> list[str]:
    out, changed = {sid}, True
    while changed:
        changed = False
        for s in reg.specs():
            if s.id not in out and any(d in out for d in s.depends_on):
                out.add(s.id); changed = True
    return [i for i in reg.ids() if i in out and i != sid]
