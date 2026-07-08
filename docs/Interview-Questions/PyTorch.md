---
title: PyTorch Interview Questions
description: 100+ PyTorch interview questions for cracking Machine Learning, Deep Learning, and ML Engineer interviews
---

# PyTorch Interview Questions

<!-- [TOC] -->

This document provides a curated list of PyTorch interview questions commonly asked in technical interviews for Machine Learning Engineer, Deep Learning Researcher, and Senior ML Engineer roles. It covers tensors and autograd, building models with nn.Module, training loops and optimizers, data loading, and the performance and deployment tooling in PyTorch 2.x.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### How Do You Create Tensors and Control Their Dtype? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Tensors`, `Dtypes` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    PyTorch offers several **constructors**, and the one you pick determines both the values and the default **dtype**.

    - `torch.tensor(data)`: copies from a Python list or NumPy array and **infers** the dtype.
    - `torch.zeros`, `torch.ones`, `torch.empty`: shape based, default dtype is `float32`.
    - `torch.arange`, `torch.linspace`: ranges; `arange` infers integer or float from its arguments.
    - `torch.randn`, `torch.rand`: sampled tensors, always floating point.

    A common trap: `torch.Tensor` (capital T) is a legacy alias for `torch.FloatTensor` and always yields `float32`, ignoring the input values. Prefer lowercase `torch.tensor`.

    ```python
    import torch

    a = torch.tensor([1, 2, 3])          # inferred int64
    b = torch.tensor([1.0, 2.0, 3.0])    # inferred float32
    c = torch.tensor([1, 2, 3], dtype=torch.float32)
    d = torch.zeros(2, 3)                # float32 by default

    print(a.dtype, b.dtype, c.dtype)     # torch.int64 torch.float32 torch.float32

    # Cast without copying data semantics changing unexpectedly
    e = a.to(torch.float32)              # explicit cast
    f = a.float()                        # shorthand for float32
    ```

    **Dtype matters for autograd:** only floating point and complex tensors can require gradients. Trying to set `requires_grad=True` on an `int64` tensor raises an error. Mixed dtype arithmetic follows **type promotion** rules, so `int + float` promotes to float.

    !!! tip "Interviewer's Insight"
        - Knows `torch.tensor` **infers** dtype while `torch.Tensor` forces `float32`
        - Understands only **floating point** tensors can track gradients
        - Real-world: **NVIDIA** teams pick `float16` or `bfloat16` deliberately to cut memory on large models

---

### What Does requires_grad Do and How Is the Graph Built? - Meta, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Autograd`, `Computational Graph` | **Asked by:** Meta, OpenAI, Google

??? success "View Answer"

    Setting **`requires_grad=True`** on a tensor tells autograd to **track every operation** applied to it so gradients can flow back later. PyTorch builds a **dynamic computational graph** on the fly: each operation creates a new tensor whose `grad_fn` points to the function that produced it.

    - The graph is a **DAG** of `Function` objects recorded during the forward pass.
    - It is **define by run**: the graph is rebuilt on every forward pass, so Python control flow (if, for, while) is fully supported.
    - Leaf tensors (created by the user with `requires_grad=True`) accumulate gradients in `.grad`; intermediate tensors do not, unless you call `.retain_grad()`.

    ```python
    import torch

    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x        # y.grad_fn -> AddBackward0
    z = y.mean()

    print(x.is_leaf, y.is_leaf)   # True False
    print(y.grad_fn)              # <AddBackward0 object ...>

    z.backward()
    print(x.grad)                # d(z)/dx = 2x + 3 = 7.0
    ```

    **Key points:**

    - If any input to an operation has `requires_grad=True`, the output does too.
    - After `backward()`, the graph is **freed** by default. Call `backward(retain_graph=True)` to keep it for a second backward pass.
    - The dynamic nature means each iteration can have a **different graph shape**, which is what makes PyTorch friendly for RNNs and variable length inputs.

    !!! tip "Interviewer's Insight"
        - Explains the **define by run** dynamic graph versus static graph frameworks
        - Distinguishes **leaf** tensors that store `.grad` from intermediates
        - Real-world: **Meta** relies on dynamic graphs for research models with data dependent control flow

---

### Why Do Gradients Accumulate and When Do You Call zero_grad? - Amazon, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Autograd`, `Training Loop` | **Asked by:** Amazon, Uber, Netflix

??? success "View Answer"

    Calling **`backward()`** does not overwrite `.grad`; it **adds** the newly computed gradient into whatever is already there. This accumulation is deliberate: it lets you sum gradients across multiple backward passes, for example to simulate a large batch by **gradient accumulation**.

    The consequence is that in a normal training loop you must **reset gradients** each step, otherwise gradients from previous iterations leak into the current update.

    ```python
    import torch

    x = torch.tensor([1.0], requires_grad=True)

    for step in range(3):
        y = (x ** 2).sum()
        y.backward()
        print(x.grad)     # 2.0, then 4.0, then 6.0 without zeroing

    # Correct training loop
    optimizer = torch.optim.SGD([x], lr=0.1)
    for step in range(3):
        optimizer.zero_grad()   # clears .grad before backward
        loss = (x ** 2).sum()
        loss.backward()
        optimizer.step()
    ```

    **Gradient accumulation pattern** (large effective batch on limited memory):

    ```python
    accum_steps = 4
    optimizer.zero_grad()
    for i, (xb, yb) in enumerate(loader):
        loss = criterion(model(xb), yb) / accum_steps
        loss.backward()                       # accumulates
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    ```

    `optimizer.zero_grad(set_to_none=True)` is the default in PyTorch 2.x and is slightly faster because it sets `.grad` to `None` instead of writing zeros.

    !!! tip "Interviewer's Insight"
        - Knows gradients **accumulate** so `zero_grad` is mandatory each step
        - Can turn the behavior into a feature with **gradient accumulation** for big batches
        - Real-world: **Uber** and others accumulate gradients to train large models on modest GPUs

---

### no_grad Versus detach: What Is the Difference? - OpenAI, Apple Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Autograd`, `Inference` | **Asked by:** OpenAI, Apple, Google

??? success "View Answer"

    Both stop gradients, but at **different granularity**.

    | Aspect | `torch.no_grad()` | `.detach()` |
    |--------|-------------------|-------------|
    | Scope | Context manager over a block | Single tensor |
    | Effect | New tensors inside have `requires_grad=False` | Returns a new tensor sharing storage, cut from the graph |
    | Storage | N/A | **Shares** the same underlying data |
    | Typical use | Inference, validation loops | Detaching a value to use as a constant |

    ```python
    import torch

    x = torch.tensor([2.0], requires_grad=True)

    # no_grad: nothing inside is tracked
    with torch.no_grad():
        y = x * 3
        print(y.requires_grad)   # False

    # detach: y2 shares data with x but has no grad history
    y2 = x.detach()
    print(y2.requires_grad)      # False
    y2 += 1                      # WARNING: also mutates x's data (shared storage)
    print(x)                     # tensor([3.], requires_grad=True)
    ```

    **Practical guidance:**

    - Use `no_grad()` around **evaluation and inference** to save memory and speed up, since no graph is built.
    - Use `detach()` when you need a tensor's **value** but want to block gradients through it, for example when computing a target in a bootstrapping loss or logging a metric.
    - Because `detach()` shares storage, an in-place edit on the detached tensor also changes the original. Add `.clone()` if you need an independent copy.

    !!! tip "Interviewer's Insight"
        - Frames `no_grad` as **block scoped** and `detach` as **tensor scoped**
        - Warns that `detach` **shares storage**, so in-place edits leak back
        - Real-world: **Apple** on device inference wraps forward passes in `no_grad` to cut memory

---

### What Are the Pitfalls of In-Place Operations with Autograd? - NVIDIA, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Autograd`, `In-Place Ops` | **Asked by:** NVIDIA, Meta, Amazon

??? success "View Answer"

    In-place operations, marked by a trailing **underscore** (`add_`, `relu_`, `mul_`) or by `+=`, modify a tensor's storage without allocating a new one. They save memory but can **break autograd** because the backward pass may need the original, unmodified values.

    **The core problem:** some backward functions save input or output tensors for the gradient computation. If you overwrite those values in place before `backward()` runs, autograd detects the change through a **version counter** and raises:

    ```
    RuntimeError: one of the variables needed for gradient computation
    has been modified by an inplace operation
    ```

    ```python
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2          # backward for pow saves x
    # y += 1            # fine, y is not needed by pow's backward
    # x.sigmoid_()      # would corrupt saved x -> error on backward

    z = y.sum()
    z.backward()
    print(x.grad)       # 2x = [2., 4., 6.]
    ```

    **Guidelines:**

    - You cannot run an in-place op on a **leaf tensor that requires grad**; it raises immediately.
    - In-place is safe when the overwritten tensor is **not required** by any saved backward function (autograd tracks this with version counters).
    - Prefer out-of-place ops in ambiguous cases; the memory win rarely justifies subtle bugs.
    - `torch.autograd.set_detect_anomaly(True)` pinpoints the offending op during debugging.

    ```python
    # Safe, common example: in-place ReLU inside a module where input is not reused
    import torch.nn as nn
    activation = nn.ReLU(inplace=True)   # valid because ReLU backward needs only output sign
    ```

    !!! tip "Interviewer's Insight"
        - Explains the **version counter** mechanism behind the in-place error
        - Knows leaf tensors requiring grad **cannot** be edited in place
        - Real-world: **NVIDIA** uses `inplace=True` activations carefully to trim activation memory

---

### Views Versus Copies: How Does contiguous() Fit In? - Google, NVIDIA Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory`, `Views` | **Asked by:** Google, NVIDIA, Microsoft

??? success "View Answer"

    A **view** is a tensor that shares the same underlying storage as another tensor but with different **shape or stride** metadata. Operations like `view`, `reshape` (when possible), `transpose`, `permute`, `narrow`, and slicing return views, not copies. Mutating a view mutates the original.

    **Strides** describe how many elements to step in storage to advance one position along each dimension. Operations like `transpose` do not move data; they just swap strides, producing a **non contiguous** tensor whose memory layout no longer matches row major order.

    ```python
    import torch

    x = torch.arange(6).reshape(2, 3)
    print(x.is_contiguous())     # True
    print(x.stride())            # (3, 1)

    xt = x.transpose(0, 1)       # view, shares storage
    print(xt.is_contiguous())    # False
    print(xt.stride())           # (1, 3)

    xt[0, 0] = 99                # also changes x, shared storage
    print(x[0, 0])               # tensor(99)
    ```

    **Why `contiguous()` matters:**

    - `view()` requires a contiguous tensor and errors on a transposed one; `reshape()` silently falls back to a **copy** when needed.
    - Calling `.contiguous()` returns a **copy** laid out in standard row major order (a no-op if already contiguous).
    - Some kernels and `.view()` demand contiguous memory, so the idiom `x.transpose(0, 1).contiguous().view(...)` is common.

    ```python
    # This raises: view on non-contiguous tensor
    # xt.view(6)
    flat = xt.contiguous().view(6)   # works after making a contiguous copy
    ```

    !!! tip "Interviewer's Insight"
        - Explains **strides** and how transpose only swaps metadata
        - Knows `view` needs contiguity while `reshape` may copy
        - Real-world: **NVIDIA** kernel authors track contiguity to avoid hidden copies in hot loops

---

### How Do You Manage Devices with .to and cuda? - Amazon, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Devices`, `GPU` | **Asked by:** Amazon, Netflix, Apple

??? success "View Answer"

    Tensors and modules live on a **device**: CPU or a specific GPU. Operations require all operands on the **same device**, so managing placement is a routine part of training.

    - `tensor.to(device)`: returns a copy on the target device (a no-op if already there for tensors).
    - `tensor.cuda()` / `tensor.cpu()`: shorthands for GPU and CPU.
    - `model.to(device)`: moves parameters **in place** for modules.

    ```python
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.Linear(10, 2).to(device)   # moves parameters in place
    x = torch.randn(4, 10).to(device)           # tensor copy on device

    out = model(x)                              # both on same device, works
    print(out.device)                           # cuda:0 or cpu
    ```

    **Key distinctions:**

    - For **tensors**, `.to()` returns a new tensor; you must reassign: `x = x.to(device)`.
    - For **modules**, `.to()` mutates the module in place, so `model.to(device)` without reassignment is correct.
    - Mixing devices raises `RuntimeError: Expected all tensors to be on the same device`.
    - `.to()` can also cast dtype and move device in one call: `x.to(device, dtype=torch.float16)`.
    - Use `non_blocking=True` with **pinned memory** to overlap host to device transfers with compute.

    !!! tip "Interviewer's Insight"
        - Knows tensor `.to()` **returns** a copy while module `.to()` is **in place**
        - Can combine device and dtype in one `.to()` call
        - Real-world: **Netflix** pipelines pin memory and use `non_blocking` transfers to keep GPUs fed

---

### How Do You Make PyTorch Training Reproducible? - Microsoft, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Reproducibility`, `Seeding` | **Asked by:** Microsoft, Airbnb, Google

??? success "View Answer"

    Reproducibility requires seeding **every source of randomness** and constraining nondeterministic GPU kernels. Randomness enters through weight initialization, data shuffling, dropout, and augmentation.

    ```python
    import torch
    import numpy as np
    import random

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)            # CPU and current GPU
        torch.cuda.manual_seed_all(seed)   # all GPUs

    set_seed(42)

    # Force deterministic algorithms where available
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # benchmark picks fastest, varies run to run
    ```

    **Things people miss:**

    - **DataLoader workers** each need seeding. Pass a `worker_init_fn` and a `generator` so shuffling is repeatable across worker processes.
    - `cudnn.benchmark=True` speeds up fixed size inputs but chooses kernels **nondeterministically**; turn it off for exact reproducibility.
    - Some CUDA ops have no deterministic implementation; `use_deterministic_algorithms(True)` will **raise** rather than silently diverge, which is the desired behavior.

    ```python
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True,
        worker_init_fn=seed_worker, generator=g,
    )
    ```

    !!! tip "Interviewer's Insight"
        - Seeds **Python, NumPy, and torch** plus the CUDA generators
        - Knows to disable **cudnn.benchmark** and seed **DataLoader workers**
        - Real-world: **Microsoft** research requires deterministic runs to compare ablations fairly

---

### Explain Broadcasting Semantics in PyTorch - Meta, Apple Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Broadcasting`, `Tensors` | **Asked by:** Meta, Apple, Amazon

??? success "View Answer"

    **Broadcasting** lets tensors of different shapes participate in elementwise operations without explicit copying. PyTorch follows the same rules as NumPy.

    Two dimensions are **compatible** when, comparing shapes from the **trailing** dimension backward, they are equal or one of them is 1. Missing leading dimensions are treated as 1.

    ```python
    import torch

    a = torch.ones(3, 1)      # shape (3, 1)
    b = torch.ones(1, 4)      # shape (1, 4)
    c = a + b                 # broadcasts to (3, 4)
    print(c.shape)            # torch.Size([3, 4])

    # Add a bias vector to each row of a batch
    x = torch.randn(32, 10)   # (batch, features)
    bias = torch.randn(10)    # (features,) -> treated as (1, 10)
    y = x + bias              # (32, 10)
    ```

    **Rules step by step** for shapes `(3, 1)` and `(1, 4)`:

    - Align trailing dims: `1` vs `4` -> one is 1, compatible, result 4.
    - Next: `3` vs `1` -> one is 1, compatible, result 3.
    - Final shape `(3, 4)`.

    **Common gotchas:**

    - Broadcasting does **not copy memory**; the expanded dimension has stride 0, so it is memory efficient.
    - Shapes like `(3,)` and `(4,)` are **incompatible** and raise a `RuntimeError`.
    - A frequent bug: adding predictions of shape `(N, 1)` to targets of shape `(N,)` silently broadcasts to `(N, N)`. Use `squeeze` or `unsqueeze` to align intent explicitly.

    ```python
    pred = torch.randn(4, 1)
    target = torch.randn(4)
    loss = (pred - target) ** 2      # SILENTLY becomes (4, 4), likely a bug
    fixed = (pred.squeeze(1) - target) ** 2   # (4,) as intended
    ```

    !!! tip "Interviewer's Insight"
        - States the **trailing dimension** rule precisely
        - Knows broadcast dims use **stride 0**, so no memory is copied
        - Real-world: **Meta** code reviews flag `(N,1)` vs `(N,)` mismatches that inflate loss tensors

---

### Anatomy of nn.Module: __init__, forward, and parameters - Google, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `nn.Module`, `Basics` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    An **nn.Module** is the base class for every model, layer, and loss in PyTorch. Three pieces matter:

    - **`__init__`**: register submodules and parameters as attributes. Anything you assign that is an `nn.Module` or `nn.Parameter` is tracked automatically.
    - **`forward`**: define the computation. You call the module like `model(x)`, which invokes `__call__`, which runs hooks and then `forward`.
    - **`parameters()`**: returns an iterator over all learnable tensors, collected recursively from every registered submodule. This is what you hand to the optimizer.

    ```python
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()  # required before assigning submodules
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, out_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = MLP(10, 32, 2)
    print(sum(p.numel() for p in model.parameters()))  # total params
    for name, p in model.named_parameters():
        print(name, p.shape, p.requires_grad)
    ```

    **Key points:**

    - Always call `super().__init__()` first, otherwise attribute registration fails.
    - Never call `model.forward(x)` directly; call `model(x)` so hooks and autograd bookkeeping run.
    - Use `nn.Parameter` for a raw learnable tensor; a plain tensor attribute is NOT registered and will not train.
    - For non-learnable state that should move with `.to(device)` and appear in the `state_dict`, use `self.register_buffer(...)`.

    !!! tip "Interviewer's Insight"
        - Explains why assigning a submodule in `__init__` makes it show up in `parameters()`
        - Knows the difference between `nn.Parameter` and `register_buffer`
        - Real-world: **Meta's PyTorch models rely on this auto-registration to sync parameters across GPUs**

---

### Write the Canonical PyTorch Training Loop - Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Training Loop`, `Autograd` | **Asked by:** Amazon, Microsoft, Uber

??? success "View Answer"

    Every supervised training loop in PyTorch follows the same five steps per batch: **forward, loss, zero_grad, backward, step**.

    ```python
    import torch
    import torch.nn as nn

    model = nn.Linear(20, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()      # clear old gradients
            logits = model(xb)         # forward pass
            loss = criterion(logits, yb)
            loss.backward()            # compute gradients
            optimizer.step()           # update weights
    ```

    **Why each step matters:**

    - **`zero_grad()`**: PyTorch accumulates gradients into `.grad` by default. Without clearing, gradients from previous batches add up and corrupt the update.
    - **`backward()`**: traverses the autograd graph built during the forward pass and fills every `param.grad`.
    - **`step()`**: applies the optimizer update rule using the current `.grad` values.

    **Common ordering mistake:** calling `zero_grad()` after `step()` in a way that clears gradients you still need, or forgetting it entirely. The safe order is `zero_grad -> forward -> loss -> backward -> step`.

    Wrap validation in `torch.no_grad()` and `model.eval()` to skip graph building and disable dropout:

    ```python
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb.to(device))
    ```

    !!! tip "Interviewer's Insight"
        - Can explain gradient accumulation and why `zero_grad()` is mandatory
        - Places `model.train()` / `model.eval()` correctly around the loop
        - Real-world: **Amazon uses this exact skeleton scaled with mixed precision and gradient scaling**

---

### CrossEntropyLoss and Its Logits Expectation - OpenAI, NVIDIA Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Loss Functions`, `Classification` | **Asked by:** OpenAI, NVIDIA, Google

??? success "View Answer"

    **`nn.CrossEntropyLoss` expects raw logits, not probabilities.** It internally applies `log_softmax` followed by negative log likelihood, so you must NOT put a `softmax` in your model's final layer.

    ```python
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    logits = torch.randn(4, 3)          # (batch, num_classes), unnormalized
    targets = torch.tensor([0, 2, 1, 0])  # class indices, dtype long
    loss = criterion(logits, targets)
    ```

    **Rules to remember:**

    - Input shape is `(N, C)` for classification; targets are **integer class indices** of shape `(N,)`, dtype `long`, NOT one-hot vectors.
    - Applying `softmax` before this loss double-counts the normalization and hurts training. Feed logits directly.
    - For binary or multi-label problems use **`nn.BCEWithLogitsLoss`**, which fuses a sigmoid and is numerically stable.

    | Loss | Feed it | Target |
    |------|---------|--------|
    | `CrossEntropyLoss` | logits `(N, C)` | class indices `(N,)` |
    | `NLLLoss` | `log_softmax` output | class indices `(N,)` |
    | `BCEWithLogitsLoss` | logits | float 0/1, same shape |

    Useful arguments: `weight=` for class imbalance, `label_smoothing=0.1` for regularization, and `ignore_index=` to skip padding tokens in sequence tasks.

    **Why fused?** Combining `log_softmax` and NLL in one op avoids computing an intermediate `softmax` that can overflow or underflow, giving better numerical stability than doing the two steps separately.

    !!! tip "Interviewer's Insight"
        - States that logits go in, and no softmax belongs in the model head
        - Knows targets are `long` class indices, not one-hot
        - Real-world: **OpenAI trains classifiers and language model heads on raw logits for numerical stability**

---

### SGD vs Adam vs AdamW: Which Optimizer and Why - Meta, Netflix Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Optimizers`, `Training` | **Asked by:** Meta, Netflix, Apple

??? success "View Answer"

    All three update weights from gradients, but they differ in how they scale and regularize.

    - **SGD (with momentum)**: `w -= lr * (momentum_buffer)`. Simple, memory light, generalizes well for vision when tuned, but sensitive to learning rate and slow to converge.
    - **Adam**: adapts the per-parameter step using running estimates of the first and second moments of the gradient. Fast, robust to LR choice, but its L2 regularization interacts badly with the adaptive scaling.
    - **AdamW**: Adam with **decoupled weight decay**. The decay is applied directly to the weights instead of being folded into the gradient, which fixes Adam's flawed regularization. It is the default for transformers.

    ```python
    import torch

    sgd   = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    adam  = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    adamw = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ```

    | Optimizer | Adaptive LR | Weight decay | Typical use |
    |-----------|-------------|--------------|-------------|
    | SGD+momentum | No | Coupled (L2) | CNNs, when you can tune |
    | Adam | Yes | Coupled (buggy) | Quick baselines |
    | AdamW | Yes | Decoupled (correct) | Transformers, LLMs |

    **Key distinction:** in Adam, `weight_decay` adds `wd * w` to the gradient, so the adaptive denominator shrinks the effective decay unevenly. AdamW subtracts `lr * wd * w` from the weight directly, keeping decay independent of the gradient magnitude. Prefer **AdamW** whenever you use weight decay with an adaptive optimizer.

    !!! tip "Interviewer's Insight"
        - Explains decoupled weight decay as the reason AdamW beats Adam
        - Knows SGD can generalize better on vision but needs careful tuning
        - Real-world: **Meta and Netflix train large transformers with AdamW plus a warmup schedule**

---

### model.train() vs model.eval(): Dropout and BatchNorm - Apple, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Training Mode`, `BatchNorm` | **Asked by:** Apple, Amazon, Microsoft

??? success "View Answer"

    `model.train()` and `model.eval()` flip a single boolean, `self.training`, on every submodule. Two layer types behave differently based on it: **Dropout** and **BatchNorm**.

    **Dropout:**

    - In `train()`: randomly zeros a fraction `p` of activations and scales the rest by `1/(1-p)` (inverted dropout), so the expected value is preserved.
    - In `eval()`: becomes an identity function; no units are dropped.

    **BatchNorm:**

    - In `train()`: normalizes each batch using the **current batch mean and variance**, and updates running statistics via a momentum average.
    - In `eval()`: normalizes using the stored **running mean and variance**, so outputs are deterministic and independent of other samples in the batch.

    ```python
    model.train()   # dropout active, batchnorm uses batch stats
    # ... training loop ...

    model.eval()    # dropout off, batchnorm uses running stats
    with torch.no_grad():
        preds = model(x_val)
    ```

    **Two separate concerns, often confused:**

    - `model.eval()` changes **layer behavior** (dropout/batchnorm).
    - `torch.no_grad()` disables **gradient tracking** to save memory and compute.

    You typically want both during validation and inference. Forgetting `eval()` is a classic bug: validation accuracy looks noisy or worse because dropout is still firing and BatchNorm is using tiny-batch statistics.

    !!! tip "Interviewer's Insight"
        - Separates `eval()` (layer mode) from `no_grad()` (autograd off)
        - Explains exactly how BatchNorm switches from batch to running statistics
        - Real-world: **Apple ships on-device models where forgetting `eval()` causes flaky predictions**

---

### Weight Initialization: Why It Matters and How to Do It - NVIDIA, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Initialization`, `Training` | **Asked by:** NVIDIA, Google, OpenAI

??? success "View Answer"

    Poor initialization causes **vanishing or exploding activations** across depth, stalling training before it starts. Good init keeps the variance of activations and gradients roughly constant layer to layer.

    - **Xavier/Glorot** (`nn.init.xavier_uniform_`): scales by fan-in and fan-out; suited to symmetric activations like `tanh`.
    - **Kaiming/He** (`nn.init.kaiming_normal_`): scales by fan-in and accounts for the `ReLU` nonlinearity, which zeros half the inputs. Default choice for ReLU nets.
    - **Bias**: usually initialized to zero.

    ```python
    import torch.nn as nn

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    model.apply(init_weights)  # recursively applies to every submodule
    ```

    **Key points:**

    - `model.apply(fn)` walks the module tree and calls `fn` on each submodule, the idiomatic way to init a whole network.
    - PyTorch layers already ship with sensible defaults (Linear uses a Kaiming-uniform variant), so custom init is most valuable for deep or unusual architectures.
    - Match the init to the activation: **Kaiming for ReLU family, Xavier for tanh/sigmoid**.
    - Initializing all weights to the same constant breaks **symmetry**: every neuron computes the same gradient and they never differentiate. Randomness is essential.

    !!! tip "Interviewer's Insight"
        - Connects the init scheme to the activation function
        - Uses `model.apply` and explains the symmetry-breaking argument
        - Real-world: **NVIDIA tunes initialization carefully to train very deep networks in mixed precision**

---

### Learning-Rate Schedulers: Warmup, Step, and Cosine - Microsoft, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Schedulers`, `Training` | **Asked by:** Microsoft, Uber, Meta

??? success "View Answer"

    A **learning-rate scheduler** adjusts the optimizer's LR over training. A high LR early helps escape bad regions; a decaying LR late helps settle into a minimum.

    ```python
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()   # step once per epoch for most schedulers
    ```

    **Common schedulers:**

    - **StepLR**: multiply LR by `gamma` every `step_size` epochs. Simple staircase decay.
    - **CosineAnnealingLR**: smoothly anneal LR following a cosine curve down to near zero.
    - **OneCycleLR**: warm up then anneal within a single run; call `scheduler.step()` every batch, not every epoch.
    - **ReduceLROnPlateau**: drop LR when a monitored metric stops improving; pass the metric: `scheduler.step(val_loss)`.

    **Critical ordering rule:** call `optimizer.step()` **before** `scheduler.step()`. Reversing them applies the schedule one step early and triggers a PyTorch warning.

    **Warmup** linearly ramps LR from near zero over the first few hundred steps, which stabilizes adaptive optimizers on transformers. Chain it with a decay using `SequentialLR` or `LambdaLR`.

    **Stepping cadence matters:** per-epoch schedulers (StepLR, CosineAnnealingLR) step once per epoch, while per-batch schedulers (OneCycleLR) step every iteration. Mixing these up silently ruins the schedule.

    !!! tip "Interviewer's Insight"
        - Gets the `optimizer.step()` then `scheduler.step()` ordering right
        - Knows which schedulers step per batch vs per epoch
        - Real-world: **Microsoft trains transformers with linear warmup then cosine decay**

---

### Custom Dataset and DataLoader: num_workers, pin_memory, collate_fn - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Data Loading`, `Dataset` | **Asked by:** Netflix, Airbnb, Amazon

??? success "View Answer"

    A **`Dataset`** defines how to fetch one sample; a **`DataLoader`** batches, shuffles, and parallelizes fetching. A map-style dataset implements `__len__` and `__getitem__`.

    ```python
    import torch
    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    loader = DataLoader(
        MyDataset(X, y),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=None,   # default stacks tensors
    )
    ```

    **What each argument does:**

    - **`num_workers`**: number of subprocesses that prefetch batches in parallel. `0` loads in the main process (slow); a value like 4 to 8 overlaps data prep with GPU compute. Too many workers can exhaust RAM and thrash.
    - **`pin_memory=True`**: places fetched batches in page-locked host memory so the CPU-to-GPU transfer is faster and can be async with `non_blocking=True`. Use it when training on GPU.
    - **`collate_fn`**: controls how a list of samples becomes a batch. The default stacks equal-shaped tensors. Override it for **variable-length sequences** (pad them) or structured samples.

    ```python
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        seqs, labels = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True)  # pad to longest in batch
        return padded, torch.tensor(labels)
    ```

    **Gotchas:**

    - With `num_workers > 0`, `__getitem__` runs in child processes, so avoid non-picklable state and heavy per-item setup; keep worker code fork-safe.
    - Set `drop_last=True` for BatchNorm-sensitive training so a tiny final batch does not skew statistics.
    - Combine `pin_memory=True` with `xb.to(device, non_blocking=True)` for overlapping transfer.

    !!! tip "Interviewer's Insight"
        - Explains `num_workers` prefetching and `pin_memory` async transfer together
        - Writes a `collate_fn` to pad variable-length sequences
        - Real-world: **Netflix and Airbnb build custom collate functions for ragged feature and sequence data**

---

### state_dict vs Whole-Model Saving: The Right Way to Checkpoint - OpenAI, Apple Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Serialization`, `Checkpointing` | **Asked by:** OpenAI, Apple, NVIDIA

??? success "View Answer"

    A **`state_dict`** is a plain Python dict mapping each parameter and buffer name to its tensor. Saving the state_dict is the **recommended** approach; saving the whole model pickles the class and is fragile across refactors.

    ```python
    import torch

    # Recommended: save the state_dict
    torch.save(model.state_dict(), 'model.pt')

    # Load: rebuild the architecture first, then load weights
    model = MyModel(...)
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model.eval()
    ```

    **Why state_dict over whole model:**

    - Saving the whole model (`torch.save(model, ...)`) pickles the class definition and file paths. If you rename or move the class, loading breaks. The state_dict stores only tensors, so it is portable and future proof.
    - You must reconstruct the architecture in code before loading a state_dict, which keeps the source of truth in your model definition.

    **Full training checkpoint** should include optimizer and scheduler state so you can resume exactly:

    ```python
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, 'ckpt.pt')

    ckpt = torch.load('ckpt.pt', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    ```

    **Important details:**

    - Use **`map_location`** to load a GPU-trained checkpoint onto CPU or a different device.
    - For a `DataParallel` or `DistributedDataParallel` model, the keys are prefixed with `module.`; unwrap with `model.module.state_dict()` or strip the prefix on load.
    - Set **`weights_only=True`** in `torch.load` for untrusted checkpoints to avoid arbitrary code execution during unpickling.
    - Call `model.eval()` after loading for inference so dropout and batchnorm behave correctly.

    !!! tip "Interviewer's Insight"
        - Recommends state_dict and explains the pickling fragility of whole-model saves
        - Checkpoints optimizer and scheduler state to resume training exactly
        - Real-world: **OpenAI checkpoints model, optimizer, and scheduler state to resume large runs after preemption**

---

### DataParallel vs DistributedDataParallel - Meta, NVIDIA Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Distributed`, `Scaling` | **Asked by:** Meta, NVIDIA, Google

??? success "View Answer"

    Both spread a model across multiple GPUs, but they work very differently and only one is recommended today.

    **`DataParallel` (DP):** single process, multiple threads. It replicates the model on every forward pass, scatters the input batch, and gathers outputs on GPU 0. It is easy to enable but slow because of the Python GIL, repeated replication, and an imbalanced load on the primary GPU.

    **`DistributedDataParallel` (DDP):** one process per GPU. Each process holds its own model replica and gradients are synchronized with an efficient **all-reduce** during the backward pass. There is no single bottleneck GPU, and it scales across multiple machines.

    | Aspect | DataParallel | DistributedDataParallel |
    |--------|--------------|--------------------------|
    | Processes | One (multithreaded) | One per GPU |
    | GIL contention | Yes | No |
    | Multi-node | No | Yes |
    | Gradient sync | Gather on GPU 0 | Ring all-reduce |
    | Recommended | No | Yes |

    ```python
    import torch
    import torch.nn as nn

    # DataParallel: simple but discouraged
    model = nn.DataParallel(MyModel()).cuda()

    # DistributedDataParallel: preferred, one process per GPU
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp_model = DDP(MyModel().to(local_rank), device_ids=[local_rank])
    ```

    **Rule of thumb:** always reach for **DDP**, even on a single machine with several GPUs. `DataParallel` remains only for quick prototypes.

---

### What Is Automatic Mixed Precision and How Do autocast and GradScaler Work? - NVIDIA, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `AMP`, `Performance` | **Asked by:** NVIDIA, Google, Meta

??? success "View Answer"

    **Automatic Mixed Precision (AMP)** runs parts of the model in lower precision (float16 or bfloat16) while keeping numerically sensitive parts in float32. On modern GPUs with Tensor Cores this gives large **speedups** and roughly **half the memory** for activations, with negligible accuracy loss.

    Two pieces cooperate:

    - **`torch.autocast`:** a context manager that automatically chooses the right precision per operation. Matrix multiplies and convolutions run in float16 or bfloat16; reductions and losses stay in float32.
    - **`torch.amp.GradScaler`:** float16 has a narrow range, so tiny gradients can **underflow to zero**. The scaler multiplies the loss by a large factor before backward, then unscales gradients before the optimizer step. It is not needed for bfloat16, which has the same exponent range as float32.

    ```python
    import torch

    scaler = torch.amp.GradScaler("cuda")
    for x, y in loader:
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(x)
            loss = loss_fn(preds, y)
        scaler.scale(loss).backward()   # scale up to avoid underflow
        scaler.step(optimizer)          # unscale, then step if no inf/nan
        scaler.update()                 # adjust scale factor
    ```

    **float16 vs bfloat16:**

    | | float16 | bfloat16 |
    |--|---------|----------|
    | Exponent range | Narrow | Same as float32 |
    | Needs GradScaler | Yes | No |
    | Hardware | Older and newer GPUs | Ampere and newer, TPUs |

    **Interview tip:** if asked why loss becomes NaN with AMP, the usual culprit is float16 overflow or a missing GradScaler. Switching to **bfloat16** on Ampere or newer hardware often removes the problem entirely.

---

### What Does torch.compile Do in PyTorch 2.x? - OpenAI, NVIDIA Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `torch.compile`, `Performance` | **Asked by:** OpenAI, NVIDIA, Apple

??? success "View Answer"

    **`torch.compile`** is the headline feature of PyTorch 2.x. It keeps eager mode's flexibility but transparently **JIT compiles** your model into optimized kernels, often yielding meaningful speedups by wrapping a single call around the model.

    Under the hood it uses three components:

    - **TorchDynamo:** hooks into CPython bytecode to capture an **FX graph** of your model with minimal code changes.
    - **AOTAutograd:** traces both forward and backward so the compiler can optimize the training graph too.
    - **TorchInductor:** the default backend that generates fused **Triton** kernels for GPU and C++ or OpenMP for CPU.

    ```python
    import torch

    model = MyModel().cuda()
    compiled = torch.compile(model)          # default mode

    # Common modes:
    # compiled = torch.compile(model, mode="max-autotune")  # tune kernels harder
    # compiled = torch.compile(model, mode="reduce-overhead")  # CUDA graphs

    out = compiled(inputs)  # first call traces and compiles, later calls are fast
    ```

    **Key ideas to mention:**

    - The **first call is slow** because it compiles; subsequent calls are fast.
    - A **graph break** happens when Dynamo hits code it cannot trace (data-dependent Python control flow, unsupported ops). Fewer breaks means better performance. Use `torch._dynamo.explain` to find them.
    - A shape change can trigger a **recompilation**; mark axes as dynamic to avoid this.

    **Fusion** (combining several elementwise ops into one kernel) and reduced Python overhead are where most of the gains come from. It composes with AMP and DDP, so you usually wrap the model once and keep the rest of the training loop unchanged.

---

### TorchScript, torch.export, and ONNX: How Do You Ship a PyTorch Model? - Apple, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deployment`, `Export` | **Asked by:** Apple, Microsoft, Amazon

??? success "View Answer"

    To run a model outside a Python training loop you need a **portable, serialized graph**. PyTorch offers three main paths.

    **TorchScript (legacy but still common):** converts a model into a serializable representation runnable from C++ with no Python dependency.

    - `torch.jit.trace` records ops for one example input. Fast, but it **bakes in control flow** and misses data-dependent branches.
    - `torch.jit.script` compiles the actual Python source, so it preserves loops and if statements.

    **`torch.export` (the modern approach):** produces a clean, full-graph **ExportedProgram** with a stable intermediate representation. It is the recommended foundation for ahead-of-time deployment in current PyTorch.

    **ONNX:** an open, framework-neutral format so a model can run in ONNX Runtime, TensorRT, or mobile and edge runtimes.

    ```python
    import torch

    model = MyModel().eval()
    example = torch.randn(1, 3, 224, 224)

    # TorchScript
    scripted = torch.jit.script(model)
    scripted.save("model.pt")

    # torch.export (modern IR)
    exported = torch.export.export(model, (example,))

    # ONNX via the newer dynamo-based exporter
    onnx_program = torch.onnx.export(model, (example,), dynamo=True)
    onnx_program.save("model.onnx")
    ```

    | Path | Runs without Python | Cross-framework | Status |
    |------|--------------------|-----------------|--------|
    | TorchScript | Yes (LibTorch) | No | Maintenance |
    | torch.export | Yes | No (PyTorch IR) | Recommended |
    | ONNX | Yes | Yes | Widely used |

    **Interview tip:** call `model.eval()` before exporting so dropout and batch norm behave in inference mode, and pass **dynamic shapes** if the serving batch size or sequence length varies.

---

### What Quantization Approaches Does PyTorch Support? - Meta, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Quantization`, `Optimization` | **Asked by:** Meta, Amazon, Apple

??? success "View Answer"

    **Quantization** stores and computes with low-precision integers (usually int8) instead of float32. It shrinks the model roughly **4x** and speeds up inference on CPUs and supported accelerators, at the cost of some accuracy. There are three classic approaches, and a newer export-based flow.

    **1. Dynamic quantization.** Weights are quantized ahead of time; activations are quantized on the fly at runtime. Easiest to apply, great for **LSTMs and Transformers** where linear layers dominate.

    ```python
    import torch
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ```

    **2. Static (post-training) quantization.** Both weights and activations are quantized. It needs a **calibration** pass over representative data to record activation ranges. Faster than dynamic at inference because activations are precomputed to int8.

    **3. Quantization-aware training (QAT).** Simulates quantization with **fake-quant** nodes during training so the model learns to compensate. Best accuracy, especially for aggressive int8, but requires retraining.

    | Approach | Needs calibration | Needs retraining | Typical accuracy |
    |----------|-------------------|------------------|------------------|
    | Dynamic | No | No | Good for Transformers |
    | Static | Yes | No | Better |
    | QAT | Yes (during training) | Yes | Best |

    **Key concepts to mention:**

    - **Per-tensor vs per-channel** scales: per-channel keeps more accuracy for convolution and linear weights.
    - **Symmetric vs affine** quantization schemes trade a zero-point for range coverage.
    - Modern PyTorch is moving toward the **`torch.export` plus PT2E** workflow (`prepare_pt2e`, `convert_pt2e`) that replaces the older eager and FX graph modes.

    **Rule of thumb:** start with dynamic quantization for a quick win on Transformers; move to static or QAT when you need maximum speed or need to recover accuracy lost by naive int8.

---

### How Do You Profile a PyTorch Model and Find Bottlenecks? - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Profiling`, `Performance` | **Asked by:** Netflix, Uber, Microsoft

??? success "View Answer"

    The right tool is **`torch.profiler`**, which records CPU and CUDA time, memory, and kernel-level detail, and exports a trace you can view in TensorBoard or Chrome's trace viewer.

    ```python
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, (x, y) in enumerate(loader):
            loss = loss_fn(model(x.cuda()), y.cuda())
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    ```

    **Common bottlenecks the trace reveals:**

    - **Data loading starvation:** the GPU sits idle waiting for the CPU. Fix with more `num_workers`, `pin_memory=True`, and `prefetch_factor`.
    - **Small, unfused kernels:** many tiny GPU launches dominated by launch overhead. Fix with `torch.compile` or larger batches.
    - **Host-device sync points:** calls like `.item()`, `.cpu()`, or printing a tensor force the GPU to flush and stall the pipeline.
    - **CPU-bound preprocessing** blocking the training thread.

    **Key idea:** because CUDA kernels run **asynchronously**, naive `time.time()` around a forward pass is misleading. Either use the profiler or call `torch.cuda.synchronize()` before measuring. Look for **gaps** in the GPU timeline, which almost always point to a data or synchronization stall rather than slow math.

---

### How Do You Debug a CUDA Out-of-Memory Error? - OpenAI, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Memory`, `Debugging` | **Asked by:** OpenAI, Netflix, NVIDIA

??? success "View Answer"

    A CUDA **out-of-memory (OOM)** error means the requested allocation exceeded free GPU memory. Work through it systematically rather than blindly calling `empty_cache`.

    **First, understand `empty_cache`.** PyTorch uses a **caching allocator**: freed tensors are kept in a pool for reuse rather than returned to the driver. `torch.cuda.empty_cache()` releases that cached pool back to the GPU. It does **not** free memory still referenced by live tensors, so it rarely fixes a genuine OOM; it mainly helps another process or framework grab memory.

    **Techniques to reduce memory, in rough order:**

    - **Lower the batch size**, or use **gradient accumulation** to keep an effective large batch with a small per-step footprint.
    - Enable **mixed precision** (autocast) to halve activation memory.
    - Use **gradient checkpointing** to trade compute for memory.
    - Wrap inference in `torch.no_grad()` so no activation graph is stored.
    - Avoid accumulating history: append `loss.item()`, not `loss`, to running logs, or you keep the whole graph alive.

    ```python
    import torch

    # See what is actually allocated vs reserved by the caching allocator
    print(torch.cuda.memory_allocated() / 1e9, "GB tensors")
    print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    print(torch.cuda.max_memory_allocated() / 1e9, "GB peak")

    # A full breakdown for debugging leaks
    print(torch.cuda.memory_summary())
    ```

    **Classic gotcha:** keeping references to tensors across iterations (for example storing the raw loss tensor for a plot) prevents the graph from being freed and causes a slow memory creep that ends in OOM. Detach or convert with `.item()` before storing.

---

### Explain Gradient Checkpointing and the Compute-Memory Trade-off - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Memory`, `Training` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Gradient checkpointing** (activation checkpointing) reduces training memory by **not storing every intermediate activation** during the forward pass. Normally backprop needs all forward activations cached to compute gradients, which for deep networks and long sequences dominates memory. Checkpointing keeps only a few **checkpoint** activations and **recomputes** the rest on the fly during the backward pass.

    **The trade-off:** you save memory but pay extra compute, typically about **one additional forward pass** (roughly 20 to 30 percent slower training). For a network of N layers, memory for activations can drop from O(N) toward O(sqrt(N)) with well-placed checkpoints. This is what lets teams fit larger models or longer sequences on the same GPU.

    ```python
    import torch
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential

    class Block(torch.nn.Module):
        def forward(self, x):
            # wrap the expensive submodule
            return checkpoint(self.layers, x, use_reentrant=False)

    # Or checkpoint a Sequential in chunks
    out = checkpoint_sequential(model.layers, segments=4, input=x,
                                use_reentrant=False)
    ```

    **Important details for interviews:**

    - Set **`use_reentrant=False`**, the modern non-reentrant implementation, which composes better with features like DDP and handles inputs that do not require grad.
    - Because activations are recomputed, any **randomness** (dropout) must be deterministic between the two passes; the checkpoint utility handles RNG state for you.
    - It composes with **mixed precision** and **DDP**, and is a standard ingredient in large Transformer training alongside sharding techniques such as FSDP.

    **When to use it:** memory-bound training where you are already at minimal batch size and still hitting OOM. When you are compute-bound instead, checkpointing only makes things slower, so it is the wrong lever.

---

### PyTorch vs TensorFlow: When Would You Choose Each? - Airbnb, Uber Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Frameworks`, `Comparison` | **Asked by:** Airbnb, Uber, Google

??? success "View Answer"

    Both are mature deep learning frameworks, and the gap has narrowed a lot. A strong answer covers **where each still has an edge** rather than declaring one universally better.

    | Dimension | PyTorch | TensorFlow |
    |-----------|---------|------------|
    | Execution | Eager by default, `torch.compile` for graphs | Eager, `tf.function` for graphs |
    | Research adoption | Dominant | Smaller share |
    | Mobile and edge | ExecuTorch, growing | TF Lite, mature |
    | Browser | Limited | TensorFlow.js |
    | Serving | TorchServe, Triton | TF Serving, mature |
    | High-level API | Lightning, native modules | Keras, tightly integrated |

    **Choose PyTorch when:**

    - You are doing **research or fast iteration**; its Pythonic, define-by-run style makes debugging natural.
    - You want the largest **model ecosystem**, since most new papers and Hugging Face weights ship PyTorch first.

    **Choose TensorFlow when:**

    - You need a **mature end-to-end production stack**: TF Serving, TFX pipelines, and TF Lite for mobile.
    - You are targeting the **browser** with TensorFlow.js or leaning on Google Cloud and TPU tooling.

    **Balanced takeaway:** PyTorch has become the default for research and much of production, especially after **PyTorch 2.x** closed the performance gap with `torch.compile`. TensorFlow remains strong where its deployment tooling and Keras integration are already embedded. In an interview, note that **team familiarity and existing infrastructure** often matter more than raw framework differences.

---

## Quick Reference: 100 PyTorch Questions

| # | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|---|----------------|----------------|------------------|------------|--------|
| 1 | What is a PyTorch tensor and how does it differ from a NumPy array? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Google, Amazon | Easy | Fundamentals, Tensors |
| 2 | How do you create tensors from Python lists and NumPy arrays? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Amazon, Microsoft | Easy | Fundamentals, Tensors |
| 3 | What are the common tensor data types (dtypes) in PyTorch? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Meta, Apple | Easy | Fundamentals, Tensors |
| 4 | How do you check and change the shape of a tensor? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Google, Netflix | Easy | Tensors, Reshaping |
| 5 | What is the difference between view() and reshape()? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Meta, Amazon | Medium | Tensors, Memory |
| 6 | Explain broadcasting rules in PyTorch. | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Google, Microsoft | Medium | Tensors, Broadcasting |
| 7 | What is the difference between torch.Tensor and torch.tensor? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Apple, Amazon | Easy | Fundamentals, Tensors |
| 8 | How do contiguous and non-contiguous tensors differ? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Meta, Nvidia | Medium | Tensors, Memory |
| 9 | How do you move a tensor between CPU and GPU? | [PyTorch Docs: CUDA](https://docs.pytorch.org/docs/stable/cuda.html) | Nvidia, Meta | Easy | Tensors, CUDA |
| 10 | What is torch.device and how is it used? | [PyTorch Docs: CUDA](https://docs.pytorch.org/docs/stable/cuda.html) | Amazon, Google | Easy | CUDA, Devices |
| 11 | How do you check if CUDA is available? | [PyTorch Docs: CUDA](https://docs.pytorch.org/docs/stable/cuda.html) | Nvidia, Microsoft | Easy | CUDA, Environment |
| 12 | What is the difference between in-place and out-of-place operations? | [PyTorch Docs: Tensors](https://docs.pytorch.org/docs/stable/tensors.html) | Meta, Apple | Medium | Tensors, Autograd |
| 13 | Why can in-place operations break autograd? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Google, Meta | Hard | Autograd, Debugging |
| 14 | Explain how automatic differentiation works in PyTorch. | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Google, Amazon | Medium | Autograd, Fundamentals |
| 15 | What is a computational graph and when is it built? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Meta, Microsoft | Medium | Autograd, Graph |
| 16 | What does requires_grad do? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Amazon, Apple | Easy | Autograd, Basics |
| 17 | How does loss.backward() compute gradients? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Google, Meta | Medium | Autograd, Training |
| 18 | What is the purpose of torch.no_grad()? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Amazon, Netflix | Easy | Autograd, Inference |
| 19 | What is the difference between detach() and no_grad()? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Meta, Google | Medium | Autograd, Inference |
| 20 | Why do gradients accumulate and how do you zero them? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Amazon, Microsoft | Medium | Autograd, Training |
| 21 | What does retain_graph=True do in backward()? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Google, Nvidia | Hard | Autograd, Graph |
| 22 | How do you compute higher-order gradients? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Meta, Apple | Hard | Autograd, Advanced |
| 23 | What is a leaf tensor in the autograd graph? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Google, Amazon | Medium | Autograd, Graph |
| 24 | How do you write a custom autograd Function? | [PyTorch Docs: Autograd](https://docs.pytorch.org/docs/stable/autograd.html) | Nvidia, Meta | Hard | Autograd, Custom |
| 25 | What is torch.nn.Module and why subclass it? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Google | Easy | nn, Models |
| 26 | What is the difference between a parameter and a buffer? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Apple | Medium | nn, Parameters |
| 27 | How does the forward() method work in a Module? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Microsoft | Easy | nn, Models |
| 28 | What is nn.Parameter and when do you use it? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Meta | Medium | nn, Parameters |
| 29 | What is the difference between nn.Sequential and a custom Module? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Netflix | Easy | nn, Models |
| 30 | How do model.train() and model.eval() change behavior? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Meta | Medium | nn, Training |
| 31 | Why do Dropout and BatchNorm behave differently in eval mode? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Amazon | Medium | nn, Regularization |
| 32 | What is nn.functional and how does it differ from nn modules? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Apple | Medium | nn, API |
| 33 | How do you implement a simple feedforward neural network? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Microsoft | Easy | nn, Models |
| 34 | What is weight initialization and why does it matter? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Google | Medium | nn, Initialization |
| 35 | How do you apply custom weight initialization? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Nvidia, Amazon | Medium | nn, Initialization |
| 36 | What is the difference between nn.Linear and manual matmul? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Apple | Easy | nn, Layers |
| 37 | How do convolutional layers (nn.Conv2d) work? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Nvidia | Medium | nn, CNN |
| 38 | What are common pooling layers and their purpose? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Google | Easy | nn, CNN |
| 39 | How do recurrent layers like nn.LSTM work in PyTorch? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Apple | Medium | nn, RNN |
| 40 | How do you implement an embedding layer with nn.Embedding? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Amazon | Medium | nn, NLP |
| 41 | What is batch normalization and how is it implemented? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Nvidia | Medium | nn, Normalization |
| 42 | What is layer normalization and when is it preferred? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Amazon | Medium | nn, Normalization |
| 43 | What are common activation functions available in PyTorch? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Apple | Easy | nn, Activations |
| 44 | How do you build a training loop from scratch? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Google, Amazon | Easy | Training, Basics |
| 45 | What is the role of the optimizer in training? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Amazon, Google | Easy | Optim, Training |
| 46 | How does torch.optim.SGD work? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Meta, Apple | Easy | Optim, SGD |
| 47 | What is the Adam optimizer and when do you use it? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Google, Amazon | Medium | Optim, Adam |
| 48 | What is the difference between Adam and AdamW? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Meta, Nvidia | Medium | Optim, Regularization |
| 49 | Why must you call optimizer.zero_grad() each step? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Amazon, Google | Easy | Optim, Training |
| 50 | What does optimizer.step() actually do? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Apple, Meta | Easy | Optim, Training |
| 51 | What is a learning rate scheduler and why use one? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Google, Amazon | Medium | Optim, Scheduling |
| 52 | How does StepLR differ from CosineAnnealingLR? | [PyTorch Docs: torch.optim](https://docs.pytorch.org/docs/stable/optim.html) | Meta, Microsoft | Medium | Optim, Scheduling |
| 53 | What is gradient clipping and when is it needed? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Nvidia | Medium | Training, Stability |
| 54 | How do you implement early stopping? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Amazon, Apple | Medium | Training, Regularization |
| 55 | What are common loss functions in PyTorch? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Amazon | Easy | Loss, Training |
| 56 | What is the difference between CrossEntropyLoss and NLLLoss? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Microsoft | Medium | Loss, Classification |
| 57 | Why does CrossEntropyLoss expect raw logits? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Google, Amazon | Medium | Loss, Classification |
| 58 | When would you use BCEWithLogitsLoss over BCELoss? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Apple | Medium | Loss, Classification |
| 59 | How do you implement a custom loss function? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Amazon, Nvidia | Medium | Loss, Custom |
| 60 | What is a Dataset in PyTorch and how do you build a custom one? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Google, Amazon | Easy | Data, Loading |
| 61 | What is the difference between map-style and iterable-style datasets? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Meta, Microsoft | Medium | Data, Loading |
| 62 | What is a DataLoader and what does it provide? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Amazon, Google | Easy | Data, Loading |
| 63 | What does the num_workers argument control? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Meta, Nvidia | Medium | Data, Performance |
| 64 | What is a collate_fn and when do you customize it? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Google, Apple | Medium | Data, Batching |
| 65 | How do you handle variable-length sequences in a batch? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Meta, Amazon | Hard | Data, NLP |
| 66 | What does pin_memory=True do? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Nvidia, Google | Medium | Data, Performance |
| 67 | How do you shuffle data correctly across epochs? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Amazon, Meta | Easy | Data, Loading |
| 68 | What is a Sampler and how does it customize batching? | [PyTorch Docs: Data](https://docs.pytorch.org/docs/stable/data.html) | Google, Apple | Medium | Data, Sampling |
| 69 | How do you apply data transforms and augmentation? | [torchvision Docs](https://docs.pytorch.org/vision/stable/index.html) | Meta, Amazon | Medium | Data, Augmentation |
| 70 | How do you save and load a model's state_dict? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Google, Amazon | Easy | Serialization, Deployment |
| 71 | Why is saving state_dict preferred over saving the whole model? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Meta, Microsoft | Medium | Serialization, Deployment |
| 72 | How do you save and resume a full training checkpoint? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Amazon, Google | Medium | Serialization, Training |
| 73 | What is map_location when loading a checkpoint? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Nvidia, Meta | Medium | Serialization, Devices |
| 74 | How do you load a model saved on GPU onto a CPU? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Google, Apple | Medium | Serialization, Devices |
| 75 | What is TorchScript and why use it? | [PyTorch Docs: TorchScript](https://docs.pytorch.org/docs/stable/jit.html) | Meta, Amazon | Hard | Deployment, JIT |
| 76 | What is the difference between torch.jit.script and torch.jit.trace? | [PyTorch Docs: TorchScript](https://docs.pytorch.org/docs/stable/jit.html) | Google, Nvidia | Hard | Deployment, JIT |
| 77 | What is torch.compile and how does it speed up models? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Meta, Google | Hard | Performance, Compilation |
| 78 | How do you export a PyTorch model to ONNX? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Amazon, Microsoft | Medium | Deployment, ONNX |
| 79 | What is Automatic Mixed Precision (AMP)? | [PyTorch Docs: AMP](https://docs.pytorch.org/docs/stable/amp.html) | Nvidia, Meta | Medium | Performance, AMP |
| 80 | How does autocast work in mixed precision training? | [PyTorch Docs: AMP](https://docs.pytorch.org/docs/stable/amp.html) | Google, Nvidia | Hard | Performance, AMP |
| 81 | What is a GradScaler and why is it needed with AMP? | [PyTorch Docs: AMP](https://docs.pytorch.org/docs/stable/amp.html) | Meta, Amazon | Hard | Performance, AMP |
| 82 | How do you profile a PyTorch model's performance? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Google, Nvidia | Hard | Performance, Profiling |
| 83 | What causes CUDA out-of-memory errors and how do you fix them? | [PyTorch Docs: CUDA](https://docs.pytorch.org/docs/stable/cuda.html) | Amazon, Meta | Hard | Debugging, CUDA |
| 84 | How does gradient accumulation help with large batch sizes? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Google, Nvidia | Medium | Training, Memory |
| 85 | What is gradient checkpointing and what does it trade off? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Meta, Amazon | Hard | Performance, Memory |
| 86 | Why are CUDA operations asynchronous and how does that affect timing? | [PyTorch Docs: CUDA](https://docs.pytorch.org/docs/stable/cuda.html) | Nvidia, Google | Hard | Performance, CUDA |
| 87 | What is the difference between DataParallel and DistributedDataParallel? | [PyTorch Docs: Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Meta, Google | Hard | Distributed, Training |
| 88 | How does DistributedDataParallel synchronize gradients? | [PyTorch Docs: Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Google, Nvidia | Hard | Distributed, Training |
| 89 | What is a process group in distributed training? | [PyTorch Docs: Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Amazon, Meta | Hard | Distributed, Training |
| 90 | What are NCCL and Gloo backends used for? | [PyTorch Docs: Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Nvidia, Google | Hard | Distributed, Backends |
| 91 | What is a DistributedSampler and why is it required? | [PyTorch Docs: Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | Meta, Amazon | Hard | Distributed, Data |
| 92 | How do you make training reproducible in PyTorch? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Google, Apple | Medium | Reproducibility, Debugging |
| 93 | What does torch.manual_seed control and what does it miss? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Amazon, Meta | Medium | Reproducibility, Debugging |
| 94 | How do you debug NaN or exploding loss values? | [Stack Overflow: pytorch](https://stackoverflow.com/questions/tagged/pytorch) | Google, Microsoft | Hard | Debugging, Stability |
| 95 | How do you use hooks to inspect activations and gradients? | [PyTorch Docs: torch.nn](https://docs.pytorch.org/docs/stable/nn.html) | Meta, Nvidia | Hard | Debugging, Hooks |
| 96 | How do you freeze layers for transfer learning? | [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | Amazon, Google | Medium | Transfer Learning, Training |
| 97 | What is torchvision and what does it provide? | [torchvision Docs](https://docs.pytorch.org/vision/stable/index.html) | Google, Meta | Easy | Ecosystem, Vision |
| 98 | How do you load a pretrained model from torchvision? | [torchvision Docs](https://docs.pytorch.org/vision/stable/index.html) | Amazon, Apple | Easy | Ecosystem, Transfer Learning |
| 99 | What are torchaudio and torchtext used for? | [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html) | Google, Meta | Easy | Ecosystem, Domains |
| 100 | What is PyTorch Lightning and what problem does it solve? | [Stack Overflow: pytorch](https://stackoverflow.com/questions/tagged/pytorch) | Amazon, Microsoft | Medium | Ecosystem, Training |

## Code Examples

#### 1. Minimal End-to-End Training Script

**Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Toy dataset: 1000 samples, 20 features, 3 classes
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    ).to(device)

    criterion = nn.CrossEntropyLoss()          # expects raw logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch}: loss={running / len(loader):.4f}")

    # Save just the learned weights
    torch.save(model.state_dict(), "model.pt")
    ```

#### 2. Custom Component: nn.Module with a Learnable Parameter and Buffer

**Difficulty:** 🟡 Medium | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import torch
    import torch.nn as nn

    class ScaledResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.act = nn.GELU()
            # A learnable scalar that gates the residual branch
            self.alpha = nn.Parameter(torch.ones(1))
            # Non-learnable state that still moves with .to(device) and saves
            self.register_buffer("call_count", torch.zeros(1))

        def forward(self, x):
            self.call_count += 1
            h = self.fc2(self.act(self.fc1(x)))
            return x + self.alpha * h

    block = ScaledResidualBlock(32)
    out = block(torch.randn(8, 32))
    print(out.shape)                                   # torch.Size([8, 32])
    print([n for n, _ in block.named_parameters()])    # includes 'alpha'
    print(block.call_count)                            # buffer, saved in state_dict
    ```

#### 3. Data Pipeline and Inference Export

**Difficulty:** 🟡 Medium | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence

    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    def collate_fn(batch):
        seqs, labels = zip(*batch)
        padded = pad_sequence(seqs, batch_first=True)      # pad to longest in batch
        lengths = torch.tensor([len(s) for s in seqs])
        return padded, lengths, torch.tensor(labels)

    seqs = [torch.randn(torch.randint(3, 8, (1,)).item(), 16) for _ in range(20)]
    labels = torch.randint(0, 2, (20,))
    loader = DataLoader(
        SequenceDataset(seqs, labels),
        batch_size=4, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=collate_fn,
    )

    padded, lengths, y = next(iter(loader))
    print(padded.shape, lengths, y.shape)

    # Export a trained model for portable, ahead-of-time serving
    model = torch.nn.Linear(16, 2).eval()
    example = torch.randn(1, 16)
    exported = torch.export.export(model, (example,))
    print(exported.graph_signature)
    ```

---

## Questions asked in Google interview
- Explain the difference between view() and reshape() and when each copies memory
- How does autograd build the dynamic computational graph during the forward pass?
- Write a training loop from scratch and justify each of the five steps
- Why does CrossEntropyLoss expect logits rather than probabilities?
- How would you make a PyTorch run fully reproducible across GPUs?
- What is torch.compile and where do its speedups come from?
- How do you debug a NaN or exploding loss?
- Explain broadcasting rules with a concrete shape mismatch example
- How does gradient checkpointing trade compute for memory?
- When would you reach for hooks to inspect activations or gradients?

## Questions asked in Amazon interview
- Walk through zero_grad, backward, and step and why the order matters
- How do you implement gradient accumulation for a large effective batch?
- Build a custom Dataset and explain num_workers and pin_memory
- What is the difference between saving a state_dict and the whole model?
- How do you resume a full training checkpoint including optimizer state?
- How do you move tensors and modules between CPU and GPU correctly?
- What causes a CUDA out-of-memory error and how do you fix it?
- Compare Adam and AdamW and explain decoupled weight decay
- How would you quantize a Transformer for cheaper CPU inference?
- Explain map-style versus iterable-style datasets

## Questions asked in Meta interview
- What does requires_grad do and which tensors accumulate gradients?
- Explain how model.train() and model.eval() change Dropout and BatchNorm
- Why can in-place operations break autograd, and how does the version counter catch it?
- Compare DataParallel and DistributedDataParallel and say which you would pick
- How does DistributedDataParallel synchronize gradients across processes?
- When do you prefer LayerNorm over BatchNorm?
- Write a custom autograd Function with forward and backward
- How do you handle variable-length sequences in a batch?
- Explain the difference between nn.Parameter and register_buffer
- How would you scale training to many GPUs and multiple nodes?

## Questions asked in Microsoft interview
- Design a learning-rate schedule with warmup followed by cosine decay
- What does map_location do when loading a checkpoint across devices?
- How do you export a PyTorch model to ONNX for serving?
- Explain torch.jit.script versus torch.jit.trace
- How do you profile a model and find the real bottleneck?
- What is the difference between CrossEntropyLoss and NLLLoss?
- How do you implement early stopping cleanly?
- Explain gradient clipping and when it is needed
- What does torch.manual_seed control and what does it miss?
- How would you make a training pipeline resumable after preemption?

## Questions asked in OpenAI interview
- Explain the difference between torch.no_grad() and detach()
- How does Automatic Mixed Precision work and why is a GradScaler needed?
- When would you use bfloat16 instead of float16?
- How does gradient checkpointing help fit larger Transformers?
- Walk through a CUDA out-of-memory investigation step by step
- Why are CUDA operations asynchronous and how does that affect timing?
- How do you checkpoint model, optimizer, and scheduler to survive a restart?
- What is torch.export and how does it differ from TorchScript?
- How would you implement a custom loss function numerically stably?
- Explain how weight initialization affects training of deep networks

---

## Additional Resources

- [PyTorch Official Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch Official Tutorials](https://docs.pytorch.org/tutorials/)
- [Deep Learning with PyTorch (Free Book, Stevens, Antiga, Viehmann)](https://www.manning.com/books/deep-learning-with-pytorch)
- [Fast.ai Practical Deep Learning for Coders](https://course.fast.ai/)
- [torchvision Documentation](https://docs.pytorch.org/vision/stable/index.html)
- [PyTorch Forums](https://discuss.pytorch.org/)
