from collections.abc import Iterable
from typing import Callable, Protocol

class _Decorator(Protocol):
    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...

class Type:
    pass

class Argument(Type):
    def __call__(self, *_: Argument) -> Type: ...

class Scalar(Argument):
    def __getitem__(self, val: slice | tuple[slice, ...]) -> Argument: ...

int64: Scalar
uint8: Scalar
float64: Scalar
void: Argument

def njit(sig: Type, /) -> _Decorator: ...
def prange(start: int) -> Iterable[int]: ...
