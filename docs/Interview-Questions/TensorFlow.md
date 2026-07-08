---
title: TensorFlow Interview Questions
description: 100+ TensorFlow interview questions for cracking Machine Learning, Deep Learning, and ML Engineer interviews
---

# TensorFlow Interview Questions

<!-- [TOC] -->

This document provides a curated list of TensorFlow interview questions commonly asked in technical interviews for Machine Learning Engineer, Deep Learning, and Senior Data Scientist roles. It covers fundamental concepts of tensors and automatic differentiation, the Keras model APIs, high-performance input pipelines with tf.data, and production topics like distributed training and serving.

This is updated frequently but right now this is the most exhaustive list of type of questions being asked.

---

## Premium Interview Questions

### What Is a Tensor and How Does It Differ from a NumPy Array? - Google, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Tensors`, `Fundamentals` | **Asked by:** Google, Amazon, Netflix

??? success "View Answer"

    A **tensor** is TensorFlow's core data structure: an immutable, n-dimensional array with a uniform data type (`dtype`) and a `shape`. Scalars are rank-0 tensors, vectors are rank-1, matrices are rank-2, and so on.

    **Key differences from a NumPy array:**

    - Tensors are **immutable**: operations return new tensors rather than mutating in place. Use `tf.Variable` when you need mutable state.
    - Tensors can live on a **GPU or TPU**, not just host memory.
    - Tensors integrate with **automatic differentiation** and graph tracing via `tf.function`.

    ```python
    import tensorflow as tf
    import numpy as np

    t = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    print(t.shape)   # (2, 2)
    print(t.dtype)   # <dtype: 'float32'>
    print(t.ndim)    # 2

    # Interop is zero-copy on CPU
    arr = t.numpy()          # tensor -> ndarray
    back = tf.convert_to_tensor(arr)  # ndarray -> tensor
    ```

    **Interop note:** Many TensorFlow ops accept NumPy arrays directly and NumPy functions accept tensors, because tensors implement the array interface. The conversion shares memory on CPU when possible, so it is cheap but not free on GPU (data must be copied to host).

    !!! tip "Interviewer's Insight"
        - Explains **immutability** and why `tf.Variable` exists for trainable state
        - Mentions **device placement** and **autodiff** as the real reasons to use tensors over ndarrays
        - Real-world: **Netflix serves recommendation models where tensors move between GPU compute and NumPy post-processing**

---

### Eager Execution vs tf.function Graphs: What Changes? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `tf.function`, `Graphs` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Eager execution** (the TensorFlow 2.x default) runs operations immediately, like normal Python, which makes debugging with prints and breakpoints easy. **`tf.function`** traces your Python function once and compiles it into a static dataflow **graph** that runs faster and can be serialized, distributed, and deployed.

    | Aspect | Eager | tf.function graph |
    |--------|-------|-------------------|
    | Execution | Op-by-op immediately | Compiled graph |
    | Debugging | Easy (print, pdb) | Harder (runs once at trace) |
    | Speed | Slower | Faster, fused ops |
    | Python side effects | Run every call | Run only during tracing |

    ```python
    import tensorflow as tf

    @tf.function
    def add(a, b):
        print("tracing")      # prints only during tracing
        return a + b

    add(tf.constant(1), tf.constant(2))  # prints "tracing"
    add(tf.constant(3), tf.constant(4))  # no print, reuses graph
    ```

    **Gotcha:** Python side effects (printing, appending to lists, mutating globals) execute only during **tracing**, not on later calls. Use `tf.print` for graph-time printing and keep functions pure with respect to Python state.

    !!! tip "Interviewer's Insight"
        - Distinguishes **trace-time** from **run-time** behavior
        - Knows Python side effects fire only during tracing, a classic bug source
        - Real-world: **OpenAI wraps training steps in tf.function to fuse ops and cut per-step Python overhead**

---

### Explain tf.function Retracing and How to Avoid It - Meta, NVIDIA Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `tf.function`, `Tracing`, `Performance` | **Asked by:** Meta, NVIDIA, Google

??? success "View Answer"

    A `tf.function` creates a separate **ConcreteFunction** (graph) for each distinct **input signature**, defined by the dtype and shape of tensor arguments and the identity of Python arguments. Every new signature triggers a **retrace**, which is expensive and, in loops, can silently destroy performance.

    **Common causes of excessive retracing:**

    - Passing **Python scalars** (like `1`, `2`, `3`) instead of tensors, since each value is a new signature.
    - Feeding tensors with **varying shapes** (variable-length batches or sequences).
    - Passing **new Python objects** each call.

    ```python
    import tensorflow as tf

    # Bad: Python ints cause a retrace per value
    @tf.function
    def f(x):
        return x * 2

    for i in range(3):
        f(i)   # 3 traces

    # Fix 1: pass tensors, not Python scalars
    for i in range(3):
        f(tf.constant(i))   # 1 trace

    # Fix 2: fix the signature for variable shapes
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def g(x):
        return tf.reduce_sum(x)
    ```

    **Diagnostics:** enable `tf.data.experimental.enable_debug_mode()` or set `@tf.function(experimental_get_tracing_count)` style checks, and watch for repeated tracing warnings. An `input_signature` with `None` dimensions makes the graph shape-polymorphic so variable batch sizes reuse one graph.

    **Practical rule:** keep the number of distinct signatures small and bounded; if you see retracing every iteration, your loop is running in eager-equivalent cost with graph overhead on top.

    !!! tip "Interviewer's Insight"
        - Defines retracing in terms of **input signatures** (dtype + shape + Python identity)
        - Uses **input_signature with None dims** to make graphs shape-polymorphic
        - Real-world: **NVIDIA tunes tf.function signatures so dynamic sequence lengths do not retrace on every GPU step**

---

### tf.Variable vs tf.constant: When to Use Each? - Amazon, Apple Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Variables`, `Fundamentals` | **Asked by:** Amazon, Apple, Microsoft

??? success "View Answer"

    **`tf.constant`** creates an **immutable** tensor whose value is fixed at creation. **`tf.Variable`** creates a **mutable** tensor whose value can be updated in place with `assign`, `assign_add`, and similar methods. Trainable model parameters (weights, biases) are always variables.

    ```python
    import tensorflow as tf

    c = tf.constant([1.0, 2.0])
    # c.assign(...) would fail: constants are immutable

    w = tf.Variable([1.0, 2.0])
    w.assign([3.0, 4.0])       # replace value
    w.assign_add([1.0, 1.0])   # in-place add -> [4.0, 5.0]
    print(w.trainable)         # True by default
    ```

    **Key points:**

    - Variables are **tracked automatically** by `GradientTape` for gradient computation.
    - Set `trainable=False` for state you update manually but do not optimize (for example, batch-norm moving averages or step counters).
    - Constants get **baked into the graph**, so large constants bloat serialized models; prefer variables or inputs for big data.

    !!! tip "Interviewer's Insight"
        - Connects **mutability** to trainable parameters and optimizer updates
        - Knows `trainable=False` for non-optimized state like BN statistics
        - Real-world: **Apple on-device models freeze most variables with trainable=False and fine-tune only a small head**

---

### How Does GradientTape Compute Gradients? - OpenAI, NVIDIA Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Autodiff`, `GradientTape` | **Asked by:** OpenAI, NVIDIA, Google

??? success "View Answer"

    `tf.GradientTape` records operations onto a **tape** during the forward pass, then applies **reverse-mode automatic differentiation** to compute gradients of a target with respect to sources. By default it watches all trainable `tf.Variable` objects touched inside the context.

    ```python
    import tensorflow as tf

    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        y = x ** 2 + 2 * x        # y = x^2 + 2x
    dy_dx = tape.gradient(y, x)    # dy/dx = 2x + 2 = 8.0
    print(dy_dx.numpy())           # 8.0
    ```

    **Important behaviors:**

    - The tape is **consumed** after one `gradient` call. Use `persistent=True` to compute multiple gradients.
    - Constants are **not watched** by default; call `tape.watch(t)` to differentiate with respect to a non-variable tensor.
    - Only ops executed **inside** the `with` block are recorded.

    ```python
    x = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x ** 2
        z = x ** 3
    print(tape.gradient(y, x).numpy())  # 6.0
    print(tape.gradient(z, x).numpy())  # 27.0
    del tape
    ```

    **Higher-order gradients** come from nesting tapes: an outer tape differentiates the gradient produced by an inner tape.

    !!! tip "Interviewer's Insight"
        - Explains **reverse-mode autodiff** and why the tape records the forward pass
        - Knows `persistent=True` and `tape.watch` for constants and multiple targets
        - Real-world: **OpenAI custom training loops use GradientTape to apply gradient clipping and mixed-precision scaling before the optimizer step**

---

### Explain Broadcasting Rules in TensorFlow - Microsoft, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Shapes`, `Broadcasting` | **Asked by:** Microsoft, Meta, Amazon

??? success "View Answer"

    **Broadcasting** lets ops combine tensors of different shapes without copying data, by virtually stretching dimensions. TensorFlow follows the same rules as NumPy: align shapes from the **trailing** (rightmost) dimension and, for each dimension, they are compatible if they are **equal** or one of them is **1**.

    ```python
    import tensorflow as tf

    a = tf.constant([[1], [2], [3]])   # shape (3, 1)
    b = tf.constant([10, 20, 30])      # shape (3,)
    print((a + b).shape)               # (3, 3)
    ```

    Here `a` is `(3, 1)` and `b` is treated as `(1, 3)`; both stretch to `(3, 3)`.

    **Step-by-step for `(3, 1)` and `(3,)`:**

    - Right-align: `(3, 1)` and `(_, 3)` where the missing left dim is treated as 1, giving `(1, 3)`.
    - Dimension 0: `3` vs `1` -> stretch to `3`.
    - Dimension 1: `1` vs `3` -> stretch to `3`.
    - Result: `(3, 3)`.

    **Common mistakes:**

    - Shapes like `(4,)` and `(3,)` are **incompatible** (neither is 1 and they are unequal) and raise an error.
    - Silent broadcasting can hide bugs; a `(N, 1)` label accidentally broadcasting against `(N, N)` predictions produces a wrong loss without any error.

    Use `tf.broadcast_to` to make broadcasting explicit and `tf.reshape` or `tf.expand_dims` to add size-1 dimensions deliberately.

    !!! tip "Interviewer's Insight"
        - States the **trailing-dimension alignment** rule precisely
        - Flags **silent broadcasting** as a real correctness hazard in loss functions
        - Real-world: **Microsoft debugging pipelines add explicit shape asserts to catch unintended broadcasts before they corrupt training**

---

### What Are Ragged and Sparse Tensors and When Do You Use Them? - Airbnb, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ragged`, `Sparse`, `Tensors` | **Asked by:** Airbnb, Uber, Google

??? success "View Answer"

    Standard tensors are dense and rectangular. **Ragged** and **sparse** tensors handle two different irregular-data cases efficiently.

    **RaggedTensor:** rows of **variable length**, such as tokenized sentences or user session sequences. It avoids padding waste while still supporting batched ops.

    ```python
    import tensorflow as tf

    rt = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
    print(rt.shape)          # (3, None)
    print(rt.row_lengths())  # [3, 1, 2]
    print((rt + 1).to_list())  # [[2, 3, 4], [5], [6, 7]]
    ```

    **SparseTensor:** mostly zeros stored as `indices`, `values`, and `dense_shape`, ideal for one-hot categorical features, bag-of-words, or huge embedding lookups.

    ```python
    st = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=[10, 20],
        dense_shape=[3, 4],
    )
    dense = tf.sparse.to_dense(st)   # zeros everywhere except the two entries
    ```

    | Type | Irregularity | Storage | Typical use |
    |------|-------------|---------|-------------|
    | Ragged | Variable row length | Values + row splits | Text, sequences |
    | Sparse | Mostly zeros | Indices + values | One-hot, embeddings |

    **Rule of thumb:** ragged for **variable length**, sparse for **mostly empty**. Both convert to dense when a downstream op requires it, but converting a huge sparse tensor to dense can blow up memory.

    !!! tip "Interviewer's Insight"
        - Separates **variable-length** (ragged) from **mostly-zero** (sparse) cleanly
        - Warns that dense conversion of large sparse tensors is a memory trap
        - Real-world: **Uber and Airbnb feature stores use sparse tensors for high-cardinality categorical inputs feeding embedding layers**

---

### How Do You Control Device Placement Across CPU and GPU? - NVIDIA, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Devices`, `GPU` | **Asked by:** NVIDIA, Netflix, Amazon

??? success "View Answer"

    TensorFlow places ops on the **GPU automatically** when a compatible GPU is visible, falling back to CPU otherwise. You override placement explicitly with `tf.device`.

    ```python
    import tensorflow as tf

    print(tf.config.list_physical_devices('GPU'))

    with tf.device('/CPU:0'):
        a = tf.constant([1.0, 2.0])

    with tf.device('/GPU:0'):
        b = tf.constant([3.0, 4.0])
        c = b * 2            # runs on GPU if available
    ```

    **Key points:**

    - Enable `tf.debugging.set_log_device_placement(True)` to log where each op runs.
    - Ops that lack a GPU kernel automatically **fall back to CPU**, which can cause hidden host-device copies that slow training.
    - Use `tf.config.set_visible_devices` to restrict which GPUs a process sees, and `set_memory_growth` to avoid grabbing all GPU memory at startup.

    ```python
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    ```

    **Interop caution:** moving data between CPU and GPU (for example, calling `.numpy()` inside a training loop) forces a copy and a synchronization, so keep hot-path computation on one device.

    !!! tip "Interviewer's Insight"
        - Knows **automatic placement** plus explicit `tf.device` overrides
        - Flags **hidden host-device copies** and CPU fallback as performance killers
        - Real-world: **NVIDIA profiling flows use set_memory_growth and device-placement logs to keep the full training step on the GPU**

---

### How Do You Ensure Reproducibility with Random Seeds? - Google, Apple Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Reproducibility`, `Seeds` | **Asked by:** Google, Apple, Microsoft

??? success "View Answer"

    Reproducibility requires controlling **every source of randomness**: weight initialization, data shuffling, dropout, and augmentation. TensorFlow has both a **global seed** and **operation-level seeds**, and results depend on both.

    ```python
    import tensorflow as tf

    tf.random.set_seed(42)          # global seed
    print(tf.random.uniform([2]))   # deterministic given the global seed
    ```

    **How the two seeds interact:**

    - With only a **global seed**, each op still derives its own deterministic stream, so repeated identical programs match.
    - An **op-level seed** (`tf.random.uniform(..., seed=1)`) makes that specific op deterministic regardless of surrounding ops.
    - Setting neither gives fully nondeterministic results.

    **For full determinism, also:**

    ```python
    import numpy as np, random, os
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()  # deterministic GPU kernels
    ```

    **Remaining gotchas:**

    - Some **GPU kernels** (certain reductions, atomics) are nondeterministic unless `enable_op_determinism()` is set, at a speed cost.
    - `tf.data` pipelines must set `shuffle(..., seed=...)` and avoid `num_parallel_calls` nondeterminism by setting `deterministic=True`.
    - Use the stateless generators (`tf.random.stateless_uniform`) when you need results that depend only on an explicit seed tensor, which is ideal for reproducible per-example augmentation.

    !!! tip "Interviewer's Insight"
        - Distinguishes **global vs op-level seeds** and how they combine
        - Knows **enable_op_determinism** and the deterministic tf.data flags for true reproducibility
        - Real-world: **Google research paper reproductions pin seeds, enable op determinism, and use stateless RNGs so runs match bit for bit**

---

### Sequential vs Functional vs Subclassed Models - Google, Meta Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Keras`, `Model API` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    Keras offers **three ways to build models**, each trading simplicity for flexibility.

    | API | Best for | Limitation |
    |-----|----------|------------|
    | **Sequential** | Plain stacks of layers | Single input, single output, no branching |
    | **Functional** | Multi-input/output, shared layers, DAGs | Static graph, no dynamic control flow |
    | **Subclassed** | Fully custom `call`, dynamic behavior | No automatic graph, harder to serialize |

    **Sequential:** a linear stack where each layer feeds the next.

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    seq = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    ```

    **Functional:** you connect layers as callables on tensors, so branching and merging are natural.

    ```python
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(128, activation="relu")(inputs)
    outputs = layers.Dense(10, activation="softmax")(x)
    func = keras.Model(inputs, outputs)
    ```

    **Subclassed:** you write imperative logic in `call`, useful for research and dynamic graphs.

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = layers.Dense(128, activation="relu")
            self.d2 = layers.Dense(10, activation="softmax")

        def call(self, x):
            return self.d2(self.d1(x))
    ```

    **Rule of thumb:** reach for **Sequential** for prototypes, **Functional** for most production models because it stays serializable and inspectable, and **subclassing** only when you need imperative control flow.

    !!! tip "Interviewer's Insight"
        - Knows the **Functional API is a DAG of layers**, not arbitrary Python
        - Understands **subclassed models lose static shape inference** and `model.summary()` until built
        - Real-world: **most Google production Keras models use the Functional API for tooling support**

---

### What Do compile, fit, evaluate, and predict Do? - Amazon, Microsoft Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Keras`, `Training` | **Asked by:** Amazon, Microsoft, Apple

??? success "View Answer"

    These four methods form the **core Keras training workflow**.

    - **`compile`** configures the model for training: it attaches the **optimizer**, **loss function**, and **metrics**. Nothing trains yet; it just wires up the objective.
    - **`fit`** runs the actual training loop over batches for a number of epochs, updating weights via backpropagation.
    - **`evaluate`** runs a forward pass over a dataset and returns the loss and metrics with **no weight updates**.
    - **`predict`** returns raw model outputs (probabilities, logits, regressions) for inference.

    ```python
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10, batch_size=32,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    probs = model.predict(x_new)
    ```

    **Key distinctions:**

    - `fit` returns a **`History`** object whose `.history` dict holds per-epoch loss and metric values.
    - `evaluate` uses the **same loss and metrics** set in `compile`, but in inference mode (dropout off, BatchNorm using running stats).
    - `predict` does not need labels and never touches the loss.

    !!! tip "Interviewer's Insight"
        - Knows **compile does not run computation**, it only configures
        - Understands **layers behave differently in training vs inference** (dropout, BatchNorm)
        - Real-world: **Amazon teams log `History` objects to compare hyperparameter sweeps**

---

### How Do Keras Callbacks Work? - Netflix, Uber Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Keras`, `Callbacks` | **Asked by:** Netflix, Uber, Google

??? success "View Answer"

    **Callbacks** are objects passed to `fit` that hook into the training loop at defined points (epoch start/end, batch start/end) to observe or alter behavior without rewriting the loop.

    Three you should know cold:

    - **`EarlyStopping`** halts training when a monitored metric stops improving, avoiding wasted epochs and overfitting.
    - **`ModelCheckpoint`** saves the model (or just weights) periodically, typically the best so far.
    - **`TensorBoard`** logs scalars, histograms, and graphs for visualization.

    ```python
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "best.keras", monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.TensorBoard(log_dir="./logs"),
    ]

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=50, callbacks=callbacks)
    ```

    **Important flags:**

    - `restore_best_weights=True` rolls the model back to the epoch with the best monitored value, not the last one.
    - `patience` is how many epochs of no improvement to tolerate before stopping.
    - `save_best_only=True` overwrites the checkpoint only when the metric improves.

    You can also write a **custom callback** by subclassing `keras.callbacks.Callback` and overriding methods like `on_epoch_end`.

    !!! tip "Interviewer's Insight"
        - Knows `restore_best_weights` matters, otherwise you keep the **overfit final epoch**
        - Understands callbacks compose; order matters when one modifies state
        - Real-world: **Netflix uses ModelCheckpoint to resume long recommender training runs**

---

### How Do You Write a Custom Keras Layer? - Meta, NVIDIA Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Keras`, `Custom Layers` | **Asked by:** Meta, NVIDIA, OpenAI

??? success "View Answer"

    Subclass **`keras.layers.Layer`** and implement three methods:

    - **`__init__`** stores configuration (units, activation) but should not create weights that depend on input shape.
    - **`build(input_shape)`** creates weights lazily once the input shape is known, using `self.add_weight`.
    - **`call(inputs)`** defines the forward computation.

    ```python
    import tensorflow as tf
    from tensorflow import keras

    class DenseWithScale(keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform", trainable=True, name="w",
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros",
                trainable=True, name="b",
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config
    ```

    **Why `build` instead of `__init__`?** It defers weight creation until the layer sees real input, so you do not hardcode the input dimension. Keras calls `build` automatically on the first forward pass.

    **Why `get_config`?** It makes the layer **serializable**, so a saved model can be reloaded with `keras.models.load_model` when you pass `custom_objects`.

    Use `self.add_weight(trainable=False)` for non-trained state like running statistics.

    !!! tip "Interviewer's Insight"
        - Knows **`build` enables shape-agnostic layers** and lazy weight creation
        - Implements **`get_config` for round-trip serialization**
        - Real-world: **NVIDIA writes custom fused layers to hit specific GPU kernels**

---

### How Do You Define Custom Losses and Metrics? - Microsoft, Airbnb Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Keras`, `Losses` | **Asked by:** Microsoft, Airbnb, Amazon

??? success "View Answer"

    A **custom loss** is any callable taking `(y_true, y_pred)` and returning a scalar (or per-sample tensor Keras reduces).

    ```python
    import tensorflow as tf
    from tensorflow import keras

    def huber_loss(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + delta * linear
    ```

    For a loss with **hyperparameters or state**, subclass `keras.losses.Loss` so it serializes cleanly.

    ```python
    class HuberLoss(keras.losses.Loss):
        def __init__(self, delta=1.0, **kwargs):
            super().__init__(**kwargs)
            self.delta = delta

        def call(self, y_true, y_pred):
            error = y_true - y_pred
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, self.delta)
            linear = abs_error - quadratic
            return 0.5 * quadratic**2 + self.delta * linear

        def get_config(self):
            return {**super().get_config(), "delta": self.delta}
    ```

    **Custom metrics** differ from losses: they **accumulate state across batches**. Subclass `keras.metrics.Metric` and use `add_weight` plus `update_state`, `result`, and `reset_state`.

    ```python
    class F1Score(keras.metrics.Metric):
        def __init__(self, name="f1", **kwargs):
            super().__init__(name=name, **kwargs)
            self.tp = self.add_weight(name="tp", initializer="zeros")
            self.fp = self.add_weight(name="fp", initializer="zeros")
            self.fn = self.add_weight(name="fn", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            y_true = tf.cast(y_true, tf.float32)
            self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
            self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
            self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

        def result(self):
            precision = self.tp / (self.tp + self.fp + 1e-7)
            recall = self.tp / (self.tp + self.fn + 1e-7)
            return 2 * precision * recall / (precision + recall + 1e-7)

        def reset_state(self):
            for v in self.variables:
                v.assign(0.0)
    ```

    **Key point:** metrics are **stateful and averaged over an epoch**, losses are computed per batch to drive gradients.

    !!! tip "Interviewer's Insight"
        - Knows metrics need **`update_state`/`result`/`reset_state`** for correct epoch aggregation
        - Understands a loss must be **differentiable**; a metric like F1 need not be
        - Real-world: **Microsoft tracks F1 as a metric while optimizing a smooth surrogate loss**

---

### How Does Transfer Learning and Fine-Tuning Work in Keras? - Google, Apple Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Transfer Learning`, `Keras` | **Asked by:** Google, Apple, Meta

??? success "View Answer"

    **Transfer learning** reuses a network pretrained on a large dataset as a feature extractor for a new, usually smaller, task. The standard recipe has two phases.

    **Phase 1: feature extraction.** Freeze the pretrained backbone and train only a fresh head.

    ```python
    import tensorflow as tf
    from tensorflow import keras

    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(224, 224, 3), pooling="avg",
    )
    base.trainable = False  # freeze all backbone weights

    inputs = keras.Input((224, 224, 3))
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)  # keep BatchNorm in inference mode
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, epochs=10)
    ```

    **Phase 2: fine-tuning.** Unfreeze part or all of the backbone and continue training with a **much lower learning rate** so you do not destroy the pretrained weights.

    ```python
    base.trainable = True
    # optionally keep early layers frozen
    for layer in base.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 10-100x smaller
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, epochs=10)
    ```

    **Critical details:**

    - Pass **`training=False`** when calling the base so frozen **BatchNorm layers use running statistics** rather than updating them.
    - You must **recompile** after changing `trainable` for the change to take effect in the training step.
    - Fine-tune with a **small learning rate**; large steps wipe out useful pretrained features.

    !!! tip "Interviewer's Insight"
        - Knows **BatchNorm must stay in inference mode** while the backbone is frozen
        - Understands you **recompile after toggling `trainable`**
        - Real-world: **Apple fine-tunes vision backbones on-device for personalization**

---

### .keras Format vs SavedModel: How Do You Save and Load Models? - Amazon, NVIDIA Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Serialization`, `Keras` | **Asked by:** Amazon, NVIDIA, Netflix

??? success "View Answer"

    Keras 3 has two main serialization targets, plus a weights-only path.

    | Format | Call | Contains | Use case |
    |--------|------|----------|----------|
    | **`.keras`** | `model.save("m.keras")` | Architecture, weights, optimizer, config | Default; resume training in Keras |
    | **SavedModel** | `model.export("dir")` | Serving graph (signatures) | TF Serving, TFLite, cross-language deploy |
    | **Weights only** | `model.save_weights("w.weights.h5")` | Just weights | Reload into identical architecture |

    **The `.keras` file** is a single zip archive holding the full model. It is the recommended format for saving and reloading inside Keras.

    ```python
    model.save("model.keras")
    reloaded = keras.models.load_model("model.keras")
    ```

    For **custom layers or losses**, either register them or pass `custom_objects`:

    ```python
    @keras.saving.register_keras_serializable()
    class DenseWithScale(keras.layers.Layer):
        ...

    reloaded = keras.models.load_model(
        "model.keras",
        custom_objects={"HuberLoss": HuberLoss},
    )
    ```

    **SavedModel via `export`** produces a language-neutral inference graph for deployment. It drops the Python training state and keeps only serving signatures.

    ```python
    model.export("saved_model_dir")          # for TF Serving / TFLite
    infer = tf.saved_model.load("saved_model_dir")
    ```

    **When to use which:**

    - Iterating and resuming training in Python: **`.keras`**.
    - Deploying to TF Serving, mobile, or another runtime: **`export` to SavedModel**.
    - Sharing weights across code that rebuilds the architecture: **`save_weights`**.

    !!! tip "Interviewer's Insight"
        - Knows **`.keras` keeps optimizer state** so training resumes exactly
        - Understands **`export` is for serving**, not for resuming training
        - Real-world: **NVIDIA exports SavedModels then converts to TensorRT for inference**

---

### How Do You Write a Custom Training Loop with GradientTape? - OpenAI, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Custom Training`, `GradientTape` | **Asked by:** OpenAI, Uber, Google

??? success "View Answer"

    When `fit` is too rigid (multiple optimizers, custom gradient manipulation, GANs, RL), write the loop yourself with **`tf.GradientTape`**.

    ```python
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10),
    ])
    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc = keras.metrics.SparseCategoricalAccuracy()

    @tf.function  # compile to a graph for speed
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)   # training=True for dropout/BN
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc.update_state(y, logits)
        return loss

    for epoch in range(epochs):
        train_acc.reset_state()
        for x_batch, y_batch in train_ds:
            loss = train_step(x_batch, y_batch)
        print(f"epoch {epoch}: acc={train_acc.result():.3f}")
    ```

    **What each piece does:**

    - **`GradientTape`** records operations on watched tensors (trainable variables by default) so you can compute gradients of the loss.
    - **`tape.gradient(loss, vars)`** does reverse-mode autodiff.
    - **`optimizer.apply_gradients`** applies the update rule to the variables.
    - **`@tf.function`** traces the Python function into a static graph, giving large speedups; keep it side-effect free.
    - **`training=True`** in the forward call activates dropout and BatchNorm updates.

    **Gradient tricks:** clip with `tf.clip_by_global_norm(grads, 1.0)` before applying, or use `tape.watch(tensor)` to differentiate w.r.t. a non-variable. Use `persistent=True` if you need multiple `gradient` calls from one tape.

    !!! tip "Interviewer's Insight"
        - Knows **`@tf.function` graph-compiles** the step and why side effects are dangerous inside it
        - Understands **`training=True` vs `False`** changes layer behavior in the forward pass
        - Real-world: **OpenAI-style RLHF loops need custom steps for policy and value gradients**

---

### Compare Dropout, Weight Decay, and Other Regularization in Keras - Meta, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Regularization`, `Keras` | **Asked by:** Meta, Microsoft, Amazon

??? success "View Answer"

    Regularization fights **overfitting** by constraining model capacity or adding noise. Keras exposes several complementary tools.

    | Technique | Mechanism | Where applied |
    |-----------|-----------|---------------|
    | **Dropout** | Randomly zeros activations during training | Between layers |
    | **L2 / weight decay** | Penalizes large weights | Per-layer `kernel_regularizer` |
    | **L1** | Encourages sparse weights | Per-layer `kernel_regularizer` |
    | **BatchNorm** | Normalizes activations, mild regularizing effect | Between layers |
    | **Early stopping** | Stops before overfitting | Callback |

    ```python
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential([
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    ```

    **Key nuances:**

    - **Dropout is active only in training.** In inference Keras scales activations so expected values match; you never manually toggle it.
    - **L2 penalty vs decoupled weight decay are not identical for adaptive optimizers.** With Adam, adding an L2 `kernel_regularizer` is not the same as true weight decay. Use **`keras.optimizers.AdamW`**, which decouples decay from the gradient update, for the correct behavior.

    ```python
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4,
    )
    ```

    - **Do not stack aggressive dropout on top of BatchNorm carelessly.** They can interact poorly; a common pattern is BatchNorm in conv blocks and dropout in the dense head.
    - Regularization strength is a **hyperparameter**: too much underfits, too little overfits. Tune with validation curves.

    **Rule of thumb:** start with **L2/AdamW weight decay plus modest dropout (0.2 to 0.5)**, add early stopping, and only then reach for heavier schemes.

    !!! tip "Interviewer's Insight"
        - Knows **L2 regularizer with Adam is not true weight decay**; AdamW fixes this
        - Understands **dropout auto-scales at inference**, no manual switch needed
        - Real-world: **Meta uses AdamW with tuned weight decay as the default for transformers**

---

### What is tf.data and Why Use It Over a Plain Python Generator? - Google, Netflix Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `tf.data`, `Input Pipeline` | **Asked by:** Google, Netflix, Amazon

??? success "View Answer"

    **`tf.data.Dataset`** is TensorFlow's API for building **input pipelines** that stream data into training loops efficiently. Instead of loading everything into memory, you describe a pipeline of transformations that execute lazily and in **C++**, overlapping with GPU compute.

    **Why prefer it over a Python generator:**

    - **Performance:** transformations run in the TensorFlow runtime, not the Python interpreter, so they escape the Global Interpreter Lock and parallelize across cores.
    - **Prefetching:** data for step N+1 is prepared while step N trains, keeping the accelerator busy.
    - **Portability:** the same pipeline plugs into `model.fit`, custom loops, and distribution strategies.
    - **Composability:** `map`, `batch`, `shuffle`, `repeat`, `filter` chain cleanly.

    ```python
    import tensorflow as tf

    dataset = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(buffer_size=10000)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(dataset, epochs=10)
    ```

    For large corpora on disk, `tf.data.TFRecordDataset` reads sharded **TFRecord** files, and `Dataset.list_files` plus `interleave` reads many shards concurrently.

    !!! tip "Interviewer's Insight"
        - Knows tf.data runs transformations in **C++ outside the GIL**
        - Mentions **lazy evaluation** and streaming rather than loading into memory
        - Real-world: **Netflix builds petabyte-scale training pipelines on tf.data with TFRecords**

---

### How Do prefetch, cache, and AUTOTUNE Improve Pipeline Throughput? - Google, NVIDIA Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `tf.data`, `Performance` | **Asked by:** Google, NVIDIA, Meta

??? success "View Answer"

    A slow input pipeline starves the GPU, so these three tools remove that bottleneck.

    **`prefetch(buffer_size)`** overlaps preprocessing on the CPU with model execution on the GPU. While the accelerator trains on batch N, the CPU prepares batch N+1. Always place it **last** in the pipeline.

    **`cache()`** stores the results of expensive upstream transformations after the first epoch, either in memory or in a file. Everything **before** the cache runs once; everything **after** runs every epoch. Place cache after costly deterministic ops (decode, resize) but **before** random augmentation so you still get fresh randomness each epoch.

    **`tf.data.AUTOTUNE`** lets the runtime dynamically tune buffer sizes and parallelism at runtime based on available CPU and memory, instead of you hardcoding magic numbers.

    ```python
    dataset = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(load_and_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()                       # cache decoded images (deterministic)
        .map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)    # overlap CPU prep with GPU compute
    )
    ```

    | Op | What it fixes | Placement |
    |----|---------------|-----------|
    | `prefetch` | GPU idle between steps | Last |
    | `cache` | Repeated expensive decode | After decode, before augment |
    | `AUTOTUNE` | Manual buffer tuning | In `map`/`prefetch` args |

    **Order matters:** a common mistake is caching after random augmentation, which freezes the augmentations to the first epoch's values.

    !!! tip "Interviewer's Insight"
        - Explains why **prefetch goes last** and why **cache goes before augmentation**
        - Uses **AUTOTUNE** rather than hardcoded buffer sizes
        - Real-world: **NVIDIA benchmarks show input pipeline tuning can double GPU utilization**

---

### What Does interleave Do and When Would You Use It? - Amazon, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `tf.data`, `Performance` | **Asked by:** Amazon, Uber, Google

??? success "View Answer"

    **`Dataset.interleave`** maps a function that produces a dataset over each element, then **interleaves** the results from multiple source datasets concurrently. Its main use is **parallel reading of many files** so I/O latency from one file overlaps with reading another.

    Compared to `map`, which produces one output element per input, `interleave` produces a **whole sub-dataset per input** and pulls from several of them at once.

    ```python
    files = tf.data.Dataset.list_files("gs://bucket/train-*.tfrecord")

    dataset = files.interleave(
        lambda path: tf.data.TFRecordDataset(path),
        cycle_length=8,                       # read 8 files concurrently
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,                  # allow out-of-order for speed
    )
    ```

    Key knobs:

    - **`cycle_length`** controls how many input elements (files) are processed concurrently. Higher values raise parallelism and memory use.
    - **`block_length`** sets how many consecutive elements are drawn from each open sub-dataset before rotating.
    - **`deterministic=False`** lets the runtime emit elements as soon as they are ready, trading reproducible ordering for throughput.

    **When to reach for it:** reading from sharded TFRecords, remote object stores where per-file latency is high, or any case where a single reader cannot saturate bandwidth. For pure element-wise transforms, plain `map` with `num_parallel_calls` is the right tool instead.

    !!! tip "Interviewer's Insight"
        - Distinguishes **interleave** (dataset per element, parallel files) from **map** (element per element)
        - Knows **cycle_length** and **deterministic=False** trade ordering for throughput
        - Real-world: **Uber reads sharded training data from object storage with interleave**

---

### How Does Mixed Precision Training Work in TensorFlow? - NVIDIA, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Mixed Precision`, `Performance` | **Asked by:** NVIDIA, OpenAI, Google

??? success "View Answer"

    **Mixed precision** runs most operations in **16-bit** floats (`float16` on GPU, `bfloat16` on TPU) while keeping numerically sensitive parts in **float32**. On modern GPUs with Tensor Cores this cuts memory roughly in half and can speed training 2 to 3 times.

    **The policy:** set a global policy so layers compute in 16-bit but keep their **variables (weights) in float32** for stable updates.

    ```python
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_model()
    # IMPORTANT: final layer outputs float32 for numerical stability
    outputs = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32")(outputs)
    ```

    **Loss scaling** is the critical piece. `float16` has a small dynamic range, so tiny gradients underflow to zero. A loss scaler multiplies the loss by a large factor before backprop and unscales the gradients afterward. Keras applies this automatically when you compile with the default optimizer wrapper:

    ```python
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # Keras wraps it with LossScaleOptimizer under mixed_float16 policy
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    ```

    **Gotchas:**

    - Keep the **softmax/output layer in float32** to avoid unstable probabilities.
    - Mixed precision helps most on GPUs with **Tensor Cores** (Volta and newer); older cards see little gain.
    - On TPU use **`mixed_bfloat16`**, which has float32's range and needs no loss scaling.

    !!! tip "Interviewer's Insight"
        - Explains **loss scaling** to prevent float16 gradient underflow
        - Knows variables stay **float32** while compute is float16
        - Real-world: **NVIDIA and OpenAI train large models in mixed precision to fit bigger batches**

---

### Explain MirroredStrategy and MultiWorkerMirroredStrategy - Meta, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Distributed`, `Strategy` | **Asked by:** Meta, Microsoft, Amazon

??? success "View Answer"

    **`tf.distribute.Strategy`** is TensorFlow's API for distributing training across multiple devices with minimal code change. Two common ones:

    **`MirroredStrategy`** does **synchronous data parallelism across multiple GPUs on a single machine**. Each GPU holds a full replica of the model, processes a different slice of the batch, and gradients are combined with **all-reduce** before each weight update, so replicas stay identical.

    ```python
    strategy = tf.distribute.MirroredStrategy()
    print("Replicas:", strategy.num_replicas_in_sync)

    with strategy.scope():
        model = build_model()
        model.compile(optimizer="adam", loss="mse")

    # Scale the global batch size by replica count
    model.fit(dataset, epochs=10)
    ```

    **`MultiWorkerMirroredStrategy`** extends the same synchronous all-reduce approach **across multiple machines**. Each worker is configured through the **`TF_CONFIG`** environment variable describing the cluster and this worker's index.

    ```python
    # TF_CONFIG example (set per worker before starting):
    # {"cluster": {"worker": ["host1:12345", "host2:12345"]},
    #  "task": {"type": "worker", "index": 0}}
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    ```

    **Key points to raise:**

    - **Everything that creates variables** (model, optimizer, metrics) must be built **inside `strategy.scope()`**.
    - The **batch size you pass is the global batch size**; it is split across replicas, so scale it up and often the learning rate too.
    - Multi-worker requires each worker to **save checkpoints** to avoid conflicts and to handle fault tolerance; only the chief writes the final model.
    - For huge embedding tables, **`ParameterServerStrategy`** (async) may suit better than mirrored all-reduce.

    !!! tip "Interviewer's Insight"
        - Knows model and optimizer must be created **inside strategy.scope()**
        - Explains **global vs per-replica batch size** and all-reduce
        - Real-world: **Meta and Microsoft scale training across GPU clusters with multi-worker strategies**

---

### How Do You Serve a TensorFlow Model in Production? - Google, Airbnb Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Deployment`, `Serving` | **Asked by:** Google, Airbnb, Amazon

??? success "View Answer"

    The standard path is the **SavedModel** format served by **TensorFlow Serving**, a high-performance C++ server with gRPC and REST endpoints.

    **Step 1: export a SavedModel.**

    ```python
    model.export("saved_model/my_model/1")   # note the version subdirectory
    ```

    The trailing `1` is a **version number**; TF Serving watches the directory and can hot-swap versions without downtime.

    **Step 2: run TensorFlow Serving** (typically via Docker):

    ```bash
    docker run -p 8501:8501 \
      --mount type=bind,source=$(pwd)/saved_model/my_model,target=/models/my_model \
      -e MODEL_NAME=my_model tensorflow/serving
    ```

    **Step 3: send a REST request:**

    ```python
    import requests, json
    resp = requests.post(
        "http://localhost:8501/v1/models/my_model:predict",
        data=json.dumps({"instances": inputs.tolist()}),
    )
    predictions = resp.json()["predictions"]
    ```

    **What TF Serving gives you:**

    - **Dynamic batching** of concurrent requests for higher throughput.
    - **Version management** with canary and A/B rollout across versions.
    - **gRPC** for low-latency, high-volume traffic; **REST** for convenience.

    For **on-device** or edge deployment, convert to **TFLite / LiteRT** instead (covered separately). For full MLOps pipelines, **TFX** wraps training, validation, and serving.

    !!! tip "Interviewer's Insight"
        - Uses **SavedModel with a version subdirectory** for hot-swapping
        - Mentions **dynamic batching** and gRPC vs REST trade-offs
        - Real-world: **Airbnb serves ranking models through TensorFlow Serving with versioned rollouts**

---

### How Would You Deploy a Model to a Mobile Device? - Apple, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `TFLite`, `Edge` | **Asked by:** Apple, Google, Meta

??? success "View Answer"

    For phones, wearables, and microcontrollers you convert to **TFLite** (now branded **LiteRT**), a compact runtime optimized for low latency and small binary size.

    **Convert the SavedModel:**

    ```python
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/my_model/1")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # enables quantization
    tflite_model = converter.convert()

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    ```

    **Quantization** is the main lever for shrinking and accelerating models:

    - **Dynamic range quantization:** weights to int8, activations computed in float. Roughly 4x smaller, easy default.
    - **Full integer quantization:** weights and activations to int8 using a **representative dataset**; needed for integer-only accelerators and Edge TPUs.
    - **Float16 quantization:** halves size, good for GPU delegates.

    ```python
    # Full integer quantization with a representative dataset
    def rep_data():
        for sample in calibration_samples.take(100):
            yield [tf.cast(sample, tf.float32)]

    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    ```

    **Running inference** uses the `Interpreter`, and on device you attach **delegates** (GPU, NNAPI on Android, Core ML on Apple) to offload to hardware accelerators.

    | Concern | TF Serving | TFLite / LiteRT |
    |---------|-----------|-----------------|
    | Target | Servers, cloud | Phones, edge, MCUs |
    | Size | Large | Tiny, quantized |
    | Latency | Network round trip | On-device, offline |

    !!! tip "Interviewer's Insight"
        - Knows **quantization types** and when a representative dataset is required
        - Mentions hardware **delegates** (GPU, NNAPI, Core ML)
        - Real-world: **Apple runs on-device vision models converted to compact int8 formats**

---

### How Do You Debug Shape Errors and Inspect Tensors During Training? - Microsoft, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Debugging`, `Best Practices` | **Asked by:** Microsoft, Uber, Google

??? success "View Answer"

    TensorFlow 2.x runs **eagerly by default**, so you can debug with normal Python tools until you wrap code in `@tf.function`, where graph tracing changes the rules.

    **1. Shape errors.** Most bugs are dimension mismatches. Inspect and assert shapes explicitly:

    ```python
    print(x.shape, x.dtype)             # eager: works like NumPy
    tf.debugging.assert_shapes([(x, ("batch", 128)), (w, (128, 10))])
    ```

    A frequent culprit is a missing or extra **batch dimension**, or broadcasting that silently succeeds then fails downstream.

    **2. Printing inside graphs.** A plain Python `print` runs **once at trace time**, not every step. To print actual runtime values inside a `@tf.function`, use **`tf.print`**, which becomes a graph op that executes each call:

    ```python
    @tf.function
    def step(x):
        tf.print("x mean:", tf.reduce_mean(x))   # prints every execution
        print("traced once")                     # prints only during tracing
        return model(x)
    ```

    **3. Force eager execution to debug.** Temporarily run functions eagerly so breakpoints and `print` behave normally:

    ```python
    tf.config.run_functions_eagerly(True)   # disable graph mode globally
    # ... reproduce the bug, step through with pdb ...
    tf.config.run_functions_eagerly(False)  # re-enable for speed
    ```

    You can also pass `model.compile(..., run_eagerly=True)` to debug a `fit` loop.

    **4. NaNs and Infs.** Enable the numeric checker to pinpoint where they first appear:

    ```python
    tf.debugging.enable_check_numerics()
    ```

    **5. Retracing warnings.** Repeated retracing (from passing Python ints or varying shapes) silently slows training; use `input_signature` on `tf.function` to lock shapes.

    !!! tip "Interviewer's Insight"
        - Knows **print traces once** while **tf.print runs every step**
        - Uses **run_functions_eagerly** to drop back into normal Python debugging
        - Real-world: **Uber teams gate graph mode off in dev to catch shape bugs before scaling**

---

### TensorFlow vs PyTorch and How Do You Handle GPU Out-of-Memory Errors? - OpenAI, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Frameworks`, `Memory` | **Asked by:** OpenAI, Meta, NVIDIA

??? success "View Answer"

    **Framework trade-offs.** The two have converged, but differences remain:

    | Dimension | TensorFlow | PyTorch |
    |-----------|-----------|---------|
    | Default execution | Eager, graphs via `@tf.function` | Eager, graphs via `torch.compile` |
    | Deployment | Mature: TF Serving, TFLite, TFX | TorchServe, ExecuTorch, growing |
    | Mobile / edge | Strong (LiteRT, Edge TPU) | Improving (ExecuTorch) |
    | Research adoption | Solid | Dominant in papers |
    | TPU support | First-class | Via XLA / PyTorch-XLA |

    Rule of thumb: **PyTorch** for research flexibility and rapid iteration; **TensorFlow** when you need a battle-tested serving and edge story. Both support XLA compilation and mixed precision.

    **Handling GPU OOM.** When training crashes with an out-of-memory error, work through these levers:

    - **Reduce batch size**, then recover the effective batch with **gradient accumulation** (accumulate grads over several micro-batches before one update).
    - **Mixed precision** (`mixed_float16`) roughly halves activation memory.
    - **Gradient checkpointing** trades compute for memory by recomputing activations during the backward pass instead of storing them.
    - **Limit or grow GPU memory** so TensorFlow does not grab all of it up front:

    ```python
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    ```

    - **Shrink the model** (fewer layers, smaller hidden size) or shard it with **model parallelism** across GPUs.
    - **Trim the input pipeline:** smaller image resolution, or lower `cache`/`shuffle` buffers that themselves consume host memory.
    - **Free references:** avoid accumulating tensors in Python lists across steps, which keeps them alive and leaks memory.

    **Diagnosing:** watch `nvidia-smi`, and note that memory that spikes then crashes usually points to a too-large batch or an unbounded buffer, while a slow climb points to a **leak** from retained tensors.

    !!! tip "Interviewer's Insight"
        - Pairs **batch reduction with gradient accumulation** to preserve effective batch size
        - Knows **gradient checkpointing** trades compute for memory
        - Real-world: **OpenAI and Meta combine mixed precision, checkpointing, and sharding to fit large models**

---

## Quick Reference: 100 TensorFlow Questions

| # | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|---|----------------|----------------|------------------|------------|--------|
| 1 | What is TensorFlow and what are its core use cases? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Amazon | Easy | Fundamentals |
| 2 | What is a tensor and how does it differ from a NumPy array? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Microsoft | Easy | Fundamentals, Tensors |
| 3 | Explain the rank, shape, and dtype of a tensor. | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Meta | Easy | Tensors |
| 4 | What is the difference between tf.constant and tf.Variable? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Netflix | Easy | Tensors, Variables |
| 5 | How do you create tensors filled with zeros, ones, or random values? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Apple | Easy | Tensors |
| 6 | What is eager execution and why was it made the default in TF 2.x? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Medium | Fundamentals, Execution |
| 7 | Explain the difference between TF 1.x graph mode and TF 2.x eager mode. | [TF Guide](https://www.tensorflow.org/guide) | Meta, Amazon | Medium | Fundamentals, Execution |
| 8 | What does the @tf.function decorator do? | [TF Guide](https://www.tensorflow.org/guide) | Google, Uber | Medium | Graphs, Performance |
| 9 | How does AutoGraph convert Python control flow into graph ops? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Hard | Graphs, AutoGraph |
| 10 | What is a computational graph and why is it useful? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Meta | Medium | Graphs |
| 11 | How do you perform basic tensor arithmetic and broadcasting? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Apple | Easy | Tensors, Broadcasting |
| 12 | Explain broadcasting rules in TensorFlow. | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Netflix | Medium | Broadcasting |
| 13 | How do you reshape, transpose, and squeeze tensors? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Microsoft | Easy | Tensors |
| 14 | What is the difference between tf.reshape and tf.transpose? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Meta, Amazon | Easy | Tensors |
| 15 | How do you concatenate and stack tensors? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Uber | Easy | Tensors |
| 16 | How do you slice and index into tensors? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Apple | Easy | Tensors, Indexing |
| 17 | What is tf.gather and when would you use it? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Meta | Medium | Tensors, Indexing |
| 18 | Explain tf.reduce_sum, tf.reduce_mean, and axis arguments. | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Microsoft | Easy | Tensors, Reductions |
| 19 | How do you cast a tensor from one dtype to another? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Netflix | Easy | Tensors, dtypes |
| 20 | What are ragged tensors and when are they needed? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Amazon | Hard | Tensors, Ragged |
| 21 | What are sparse tensors and how do you work with them? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Meta, Google | Hard | Tensors, Sparse |
| 22 | What is a string tensor and how do you manipulate text tensors? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Amazon, Apple | Medium | Tensors, Text |
| 23 | How does automatic differentiation work in TensorFlow? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Medium | Autodiff, Gradients |
| 24 | What is tf.GradientTape and how do you use it? | [TF Guide](https://www.tensorflow.org/guide) | Google, Meta | Medium | Autodiff, Gradients |
| 25 | How do you compute higher-order derivatives with nested GradientTapes? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Hard | Autodiff, Gradients |
| 26 | What does persistent=True do in tf.GradientTape? | [TF Guide](https://www.tensorflow.org/guide) | Meta, Uber | Medium | Autodiff |
| 27 | How do you stop gradients from flowing with tf.stop_gradient? | [TensorFlow Docs](https://www.tensorflow.org/api_docs/python/tf) | Google, Amazon | Medium | Autodiff, Gradients |
| 28 | What is the difference between watched and unwatched variables in GradientTape? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Medium | Autodiff |
| 29 | What is Keras and how does it relate to TensorFlow? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Easy | Keras, Fundamentals |
| 30 | Compare the Sequential, Functional, and Subclassing model APIs. | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Meta | Medium | Keras, Models |
| 31 | How do you build a model with the Keras Sequential API? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Apple | Easy | Keras, Models |
| 32 | How do you build a model with the Keras Functional API? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Netflix | Medium | Keras, Models |
| 33 | How do you create a custom model by subclassing tf.keras.Model? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Meta, Amazon | Medium | Keras, Models |
| 34 | How do you write a custom Keras layer? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Microsoft | Medium | Keras, Layers |
| 35 | What is the difference between a layer's build and call methods? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Medium | Keras, Layers |
| 36 | What common layer types does Keras provide? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Uber | Easy | Keras, Layers |
| 37 | Explain Dense, Conv2D, and LSTM layers. | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Meta | Medium | Keras, Layers |
| 38 | What is the purpose of an activation function and which are available? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Apple | Easy | Keras, Activations |
| 39 | Compare ReLU, sigmoid, tanh, and softmax activations. | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Microsoft | Medium | Activations |
| 40 | How do you compile a Keras model? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Netflix | Easy | Keras, Training |
| 41 | What arguments does model.fit accept? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Meta, Amazon | Easy | Keras, Training |
| 42 | How do you evaluate and predict with a Keras model? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Apple | Easy | Keras, Training |
| 43 | What is a loss function and how do you choose one? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Medium | Losses |
| 44 | Compare MSE, categorical crossentropy, and binary crossentropy. | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Meta | Medium | Losses |
| 45 | How do you write a custom loss function in Keras? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Microsoft | Medium | Losses, Keras |
| 46 | What is an optimizer and how does gradient descent work? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Uber | Medium | Optimizers |
| 47 | Compare SGD, Adam, RMSprop, and Adagrad optimizers. | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Meta | Medium | Optimizers |
| 48 | What is a learning rate and how do you schedule it? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Apple | Medium | Optimizers, Training |
| 49 | How do learning rate schedules and warmup work in Keras? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Hard | Optimizers, Training |
| 50 | What are Keras metrics and how do you track them? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Netflix | Easy | Metrics |
| 51 | How do you write a custom training loop with GradientTape? | [TF Guide](https://www.tensorflow.org/guide) | Meta, Google | Hard | Training, Autodiff |
| 52 | What are Keras callbacks and give examples? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Medium | Callbacks, Training |
| 53 | How does the EarlyStopping callback work? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Microsoft | Medium | Callbacks |
| 54 | How do you use ModelCheckpoint to save the best model? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Apple | Medium | Callbacks, Saving |
| 55 | What is the ReduceLROnPlateau callback? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Uber | Medium | Callbacks |
| 56 | How do you write a custom Keras callback? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Meta, Amazon | Medium | Callbacks |
| 57 | What is the tf.data API and why use it? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Google, Amazon | Medium | Data, Input Pipeline |
| 58 | How do you create a Dataset from tensors or NumPy arrays? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Google, Apple | Easy | Data |
| 59 | Explain Dataset map, batch, shuffle, and repeat. | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Amazon, Meta | Medium | Data, Input Pipeline |
| 60 | What does dataset.prefetch do and why is it important? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Google, Microsoft | Medium | Data, Performance |
| 61 | What is tf.data.AUTOTUNE? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Amazon, Google | Medium | Data, Performance |
| 62 | How do you build an efficient input pipeline for large datasets? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Google, Netflix | Hard | Data, Performance |
| 63 | What is the TFRecord format and when should you use it? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Meta, Google | Hard | Data, TFRecord |
| 64 | How do you parse tf.train.Example protos from TFRecords? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Amazon, Google | Hard | Data, TFRecord |
| 65 | How do you cache a dataset and when does it help? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Google, Uber | Medium | Data, Performance |
| 66 | How do you interleave reads from multiple files? | [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) | Amazon, Meta | Hard | Data, Performance |
| 67 | How do you load image data with image_dataset_from_directory? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Apple | Easy | Data, Images |
| 68 | How do you perform data augmentation in TensorFlow? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Medium | Data, Augmentation |
| 69 | What preprocessing layers does Keras offer (Normalization, Rescaling)? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Microsoft | Medium | Preprocessing |
| 70 | How do you handle text tokenization with TextVectorization? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Meta, Amazon | Medium | Preprocessing, Text |
| 71 | What is feature engineering with tf.feature_column? | [TF Guide](https://www.tensorflow.org/guide) | Google, Amazon | Hard | Preprocessing, Features |
| 72 | How do you save and load an entire Keras model? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Apple | Easy | Saving, Deployment |
| 73 | What is the difference between SavedModel and HDF5 formats? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Medium | Saving, Deployment |
| 74 | How do you save and restore only model weights? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Netflix | Medium | Saving |
| 75 | What is a tf.train.Checkpoint and how does it differ from SavedModel? | [TF Guide](https://www.tensorflow.org/guide) | Meta, Google | Hard | Saving, Checkpoints |
| 76 | How do you use transfer learning with a pretrained model? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Amazon, Google | Medium | Transfer Learning |
| 77 | How do you freeze and unfreeze layers for fine-tuning? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Google, Apple | Medium | Transfer Learning |
| 78 | What is TensorBoard and what can you visualize with it? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Medium | TensorBoard, Debugging |
| 79 | How do you log scalars, histograms, and images to TensorBoard? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Medium | TensorBoard |
| 80 | How do you debug NaN or exploding gradients during training? | [SO: tensorflow](https://stackoverflow.com/questions/tagged/tensorflow) | Google, Meta | Hard | Debugging, Training |
| 81 | What causes shape mismatch errors and how do you fix them? | [SO: tensorflow](https://stackoverflow.com/questions/tagged/tensorflow) | Amazon, Google | Medium | Debugging |
| 82 | How do you profile a TensorFlow model for bottlenecks? | [TF Guide](https://www.tensorflow.org/guide) | Google, Uber | Hard | Performance, Profiling |
| 83 | What is mixed precision training and how do you enable it? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Hard | Performance, Mixed Precision |
| 84 | What is XLA and how does it accelerate TensorFlow? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Hard | Performance, XLA |
| 85 | How do you use GPUs with TensorFlow and control memory growth? | [TF Guide](https://www.tensorflow.org/guide) | Google, Amazon | Medium | GPU, Performance |
| 86 | What is a tf.distribute.Strategy? | [TF Guide](https://www.tensorflow.org/guide) | Meta, Google | Hard | Distributed |
| 87 | Explain MirroredStrategy for multi-GPU training. | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Hard | Distributed |
| 88 | What is MultiWorkerMirroredStrategy and how does it work? | [TF Guide](https://www.tensorflow.org/guide) | Google, Microsoft | Hard | Distributed |
| 89 | How does TPUStrategy enable training on TPUs? | [TF Guide](https://www.tensorflow.org/guide) | Google, Apple | Hard | Distributed, TPU |
| 90 | What is data parallelism versus model parallelism? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Meta | Hard | Distributed |
| 91 | What is TensorFlow Serving and how do you deploy a model with it? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Google, Amazon | Hard | Deployment, Serving |
| 92 | What is TensorFlow Lite and when do you use it? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Google, Apple | Medium | Deployment, TFLite |
| 93 | How do you quantize a model for edge deployment? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Amazon, Google | Hard | Deployment, Quantization |
| 94 | What is TensorFlow.js and what can it do? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Google, Meta | Medium | Deployment, TFjs |
| 95 | What is TFX and what problem does it solve? | [TF Guide](https://www.tensorflow.org/guide) | Amazon, Google | Hard | Ecosystem, TFX |
| 96 | What is TensorFlow Hub? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Google, Microsoft | Easy | Ecosystem, Hub |
| 97 | What is TensorFlow Datasets (TFDS)? | [TF Tutorials](https://www.tensorflow.org/tutorials) | Amazon, Google | Easy | Ecosystem, Data |
| 98 | How do you handle overfitting with dropout and regularization? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Google, Meta | Medium | Regularization |
| 99 | How does batch normalization work and where do you place it? | [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) | Amazon, Google | Medium | Normalization |
| 100 | How do you set random seeds for reproducibility in TensorFlow? | [SO: tensorflow](https://stackoverflow.com/questions/tagged/tensorflow) | Google, Apple | Medium | Reproducibility |

## Code Examples

### 1. Minimal End-to-End Training Script

**Difficulty:** 🟢 Easy | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Load and normalize MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build a simple classifier with the Sequential API
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=5, batch_size=64,
        callbacks=[keras.callbacks.EarlyStopping(patience=2,
                                                 restore_best_weights=True)],
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"test accuracy: {test_acc:.4f}")
    model.save("mnist.keras")
    ```

### 2. Custom Layer and Model with GradientTape

**Difficulty:** 🟡 Medium | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import tensorflow as tf
    from tensorflow import keras

    # A reusable custom layer with lazy weight creation and serialization
    @keras.saving.register_keras_serializable()
    class Linear(keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform", trainable=True, name="w",
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros",
                trainable=True, name="b",
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

        def get_config(self):
            return {**super().get_config(), "units": self.units}

    model = keras.Sequential([Linear(64), keras.layers.ReLU(), Linear(10)])
    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # for x_batch, y_batch in train_ds:
    #     loss = train_step(x_batch, y_batch)
    ```

### 3. High-Performance tf.data Pipeline

**Difficulty:** 🟡 Medium | **Tags:** `Code Example` | **Asked by:** Code Pattern

??? success "View Code Example"

    ```python
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE
    IMG_SIZE = 224

    def load_and_decode(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((file_paths, labels))
        .shuffle(buffer_size=10000, seed=42)
        .map(load_and_decode, num_parallel_calls=AUTOTUNE)
        .cache()                    # cache decoded images before random augment
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(64)
        .prefetch(AUTOTUNE)         # overlap CPU prep with GPU compute
    )

    # model.fit(train_ds, epochs=10)
    ```

---

## Questions asked in Google interview
- Explain the difference between eager execution and tf.function graphs
- How would you diagnose and fix excessive tf.function retracing?
- Design an efficient tf.data pipeline for a petabyte-scale image dataset
- How do you make a TensorFlow training run bit-for-bit reproducible?
- Implement a custom training loop with GradientTape and gradient clipping
- How does AutoGraph translate Python control flow into graph ops?
- Explain XLA compilation and when it speeds up a model
- How would you serve a versioned SavedModel with TensorFlow Serving?
- Compare data parallelism and model parallelism for large models
- How do you profile and remove input-pipeline bottlenecks?

## Questions asked in Amazon interview
- What is the difference between tf.constant and tf.Variable?
- Walk through compile, fit, evaluate, and predict in Keras
- How do you write and serialize a custom Keras layer?
- Explain how prefetch, cache, and AUTOTUNE improve throughput
- How would you parse tf.train.Example protos from TFRecords?
- Design a transfer-learning pipeline for a small labeled dataset
- How do you save and restore only model weights across architectures?
- Explain the trade-offs between SavedModel and the .keras format
- How would you handle a GPU out-of-memory error during training?
- Implement a custom stateful metric like F1 in Keras

## Questions asked in Meta interview
- Explain broadcasting rules and a silent broadcasting bug you have seen
- When do you use ragged versus sparse tensors?
- Compare Sequential, Functional, and subclassed model APIs
- How does MirroredStrategy combine gradients across GPUs?
- Explain L2 regularization versus decoupled weight decay with AdamW
- How do you build a custom model by subclassing tf.keras.Model?
- Explain how batch normalization behaves in training versus inference
- How would you scale synchronous training across many workers?
- Implement semantic-style feature hashing for high-cardinality inputs
- How do you debug exploding gradients in a deep network?

## Questions asked in Microsoft interview
- How does tf.GradientTape compute gradients under the hood?
- What is the difference between a layer's build and call methods?
- How do you enable and reason about mixed precision training?
- Explain MultiWorkerMirroredStrategy and the role of TF_CONFIG
- How do you debug shape mismatch errors in a graph-mode function?
- What does persistent=True do in a GradientTape?
- How would you log scalars and histograms to TensorBoard?
- Explain learning rate schedules and warmup in Keras
- How do you control GPU memory growth to avoid grabbing all memory?
- Compare SavedModel and HDF5 serialization formats

## Questions asked in OpenAI interview
- Why do Python side effects run only during tf.function tracing?
- How do you implement gradient accumulation for large effective batches?
- Explain loss scaling and why float16 gradients underflow
- How would you write an RLHF-style custom training step?
- Compare TensorFlow and PyTorch for research and deployment
- How does gradient checkpointing trade compute for memory?
- How do you compute higher-order derivatives with nested tapes?
- Explain how stop_gradient is used in practice
- How do you keep a training step fully on the GPU with no host copies?
- How would you shard a model that does not fit on one GPU?

---

## Additional Resources

- [Official TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Developer Guides](https://keras.io/guides/)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurelien Geron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
- [DeepLearning.AI TensorFlow Developer Professional Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)
