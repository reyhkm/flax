---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

```{code-cell} ipython3
from flax import nnx
import jax
import jax.numpy as jnp
import dataclasses
```

## Pytree

```{code-cell} ipython3
class Linear(nnx.Pytree):
  def __init__(self, din: int, dout: int):
    self.din = din
    self.dout = dout
    self.w = jnp.ones((din, dout))
    self.b = jnp.zeros((dout,))

pytree = Linear(1, 2)

print("pytree structure:")
for path, value in jax.tree.leaves_with_path(pytree):
  print(f"- pytree{jax.tree_util.keystr(path)} = {value!r}")
```

### Classifying Attributes

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self, i: int):
    self.i = nnx.data(i)  # explicit data
    self.s = nnx.static("Hi" + "!" * i)  # explicit static
    self.x = jnp.array(42 * i)  # auto data
    self.a = hash(i)  # auto static

class Bar(nnx.Pytree):
  def __init__(self):
    self.ls = [Foo(i) for i in range(3)]  # auto data
    self.shapes = [8, 16, 32]  # auto static

pytree = Bar()

print("pytree structure:")
for path, value in jax.tree.leaves_with_path(pytree):
  print(f"- pytree{jax.tree_util.keystr(path)} = {value!r}")
```

* mention `nnx.is_data_type`
* mention `nnx.register_data_type`

```{code-cell} ipython3
@dataclasses.dataclass
class Foo(nnx.Pytree):
  i: nnx.Data[int]
  s: nnx.Static[str]
  x: jax.Array
  a: int

@dataclasses.dataclass
class Bar(nnx.Pytree):
  ls: list[Foo]
  shapes: list[int]

pytree = Bar(
  ls=[Foo(i, "Hi" + "!" * i, jnp.array(42 * i), hash(i)) for i in range(3)],
  shapes=[8, 16, 32]
)

print("pytree structure:")
for path, value in jax.tree.leaves_with_path(pytree):
  print(f"- pytree{jax.tree_util.keystr(path)} = {value!r}")
```

#### When to use explicit annotations?

```{code-cell} ipython3
class Bar(nnx.Pytree):
  def __init__(self, x, num_layers: int, use_bias: bool):
    self.x = nnx.data(x)  # force inputs (e.g. user could pass Array or ShapeDtypeStruct)
    self.ls = nnx.data([ # on potentially empty pytrees (e.g. num_layers = 0)
      jnp.array(i) for i in range(num_layers)
    ])
    if use_bias:
      self.bias = nnx.Param(jnp.array(0.0))
    else:
      self.bias = nnx.data(None)  # on empty pytrees

bar = Bar(1.0, 3, True)

print("pytree structure:")
for path, value in jax.tree.leaves_with_path(bar):
  print(f"- bar{jax.tree_util.keystr(path)} = {value!r}")
```

#### Trace-level awareness

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.count = nnx.data(0)

foo = Foo()

@jax.vmap  # or jit, grad, shard_map, pmap, scan, etc.
def increment(n):
  foo.count += 1

try:
  increment(jnp.arange(5))
except Exception as e:
  print(f"Error: {e}")
```

### Post `__init__` attribute checks

```{code-cell} ipython3
class Foo(nnx.Pytree):
  def __init__(self):
    self.ls = []  # no data when setting the attribute!!
    for i in range(5):
      self.ls.append(jnp.array(i))

    print("num nodes before:", len(jax.tree.leaves(self)))  # ls is not data

foo = Foo()  # attributes checked after __init__ is done

print("num nodes after:", len(jax.tree.leaves(foo)))  # ls added as data
```

## Module

+++

### set_attributes

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.mode = 1
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    return nnx.relu(self.dropout(self.bn(self.linear(x))))
  
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

print("train:")
print(f"  {model.mode = }")
print(f"  {model.bn.use_running_average = }")
print(f"  {model.dropout.deterministic = }")

# Set attributes for evaluation
model.set_attributes(deterministic=True, use_running_average=True, mode=2)

print("eval:")
print(f"  {model.mode = }")
print(f"  {model.bn.use_running_average = }")
print(f"  {model.dropout.deterministic = }")
```

```{code-cell} ipython3
model = Block(din=1, dout=2, rngs=nnx.Rngs(0))

model.eval(mode=2)  # .set_attributes(deterministic=True, use_running_average=True, mode=2)
print("eval:")
print(f"  {model.mode = }")
print(f"  {model.bn.use_running_average = }")
print(f"  {model.dropout.deterministic = }")

model.train(mode=1)  # .set_attributes(deterministic=False, use_running_average=False, mode=1)
print("train:")
print(f"  {model.mode = }")
print(f"  {model.bn.use_running_average = }")
print(f"  {model.dropout.deterministic = }")
```

### sow

```{code-cell} ipython3
class Block(nnx.Module):
  def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dout, rngs=rngs)
    self.bn = nnx.BatchNorm(dout, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, rngs=rngs)

  def __call__(self, x):
    y = nnx.relu(self.dropout(self.bn(self.linear(x))))
    self.sow(nnx.Intermediate, "y_mean", jnp.mean(y))
    return y

class MLP(nnx.Module):
  def __init__(self, num_layers, dim, rngs: nnx.Rngs):
    self.blocks = [Block(dim, dim, rngs) for _ in range(num_layers)]

  def __call__(self, x):
    for block in self.blocks:
      x = block(x)
    return x


model = MLP(num_layers=3, dim=20, rngs=nnx.Rngs(0))
x = jnp.ones((10, 20))
y = model(x)
intermediates = nnx.pop(model, nnx.Intermediate) # extract intermediate values

print(intermediates)
```
