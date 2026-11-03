"""Step registry: every step declares id, description, dependencies, contracts, version."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class StepSpec:
    id: str
    description: str
    run: Callable
    depends_on: tuple = ()
    version: str = "1"
    config_keys: tuple = ()                 # dotted config keys that affect this step
    outputs: tuple = ()                     # relative file names written in the step directory
    inputs_doc: str = ""                    # input contract (human readable)
    outputs_doc: str = ""
    inputs_fn: Callable | None = None       # ctx -> list[Path] of external input files to fingerprint
    validate_fn: Callable | None = None     # ctx -> None, raises on invalid outputs/inputs
    resources: dict = field(default_factory=dict)
    requires: tuple = ()                    # heavy frameworks this step genuinely needs, e.g. ('torch',)


class Registry:
    def __init__(self):
        self._steps: dict[str, StepSpec] = {}

    def register(self, spec: StepSpec):
        if spec.id in self._steps:
            raise RegistryError(f"duplicate step id {spec.id!r}")
        if not spec.id.isidentifier():
            raise RegistryError(f"invalid step id {spec.id!r}")
        if not spec.outputs:
            raise RegistryError(f"step {spec.id!r} declares no outputs")
        self._steps[spec.id] = spec
        return spec

    def get(self, sid: str) -> StepSpec:
        if sid not in self._steps:
            raise RegistryError(f"unknown step {sid!r}; known: {list(self._steps)}")
        return self._steps[sid]

    def ids(self):
        return list(self._steps)

    def specs(self):
        return list(self._steps.values())

    def __contains__(self, sid):
        return sid in self._steps


REGISTRY = Registry()


def step(id, description, depends_on=(), version="1", config_keys=(), outputs=(), inputs_doc="",
         outputs_doc="", inputs_fn=None, validate_fn=None, resources=None, requires=(), registry: Registry = REGISTRY):
    def deco(fn):
        registry.register(StepSpec(id=id, description=description, run=fn, depends_on=tuple(depends_on),
                                   version=version, config_keys=tuple(config_keys), outputs=tuple(outputs),
                                   inputs_doc=inputs_doc, outputs_doc=outputs_doc, inputs_fn=inputs_fn,
                                   validate_fn=validate_fn, resources=resources or {}, requires=tuple(requires)))
        return fn
    return deco
