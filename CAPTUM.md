# Captum Semantic Pullbacks API Design

## Background

Semantic Pullbacks (SP) are an alternative to standard gradients for neural networks: an operator that transports a target output back through the network by treating each layer as an input-dependent affine transformation `f(x) = W(x)x + b(x)`. Because the network is a composition of such functions, the whole model can be represented by such transformation as well. Standard autograd is one particular choice of backward operator; pullbacks provide another, and any gradient-based attribution method can be run with either.

For a network composed of locally dynamic operators, let `u` be a scalar target selector (normally one output logit). The standard pullback is

```text
ν_u(x) = W(x)^T u
```

where `W(x)` is the local forward operator evaluated at `x`. Optional
backward-only softening is particularly useful for hard-gated layers (e.g.
ReLU or MaxPool), whose routing decisions can be sensitive to the input.
The forward pass remains unchanged; only the backward transport is modified.

Components:

* Model (forward function)
* X = Input
* u = Output-space target selector (e.g. one-hot vector for a class logit)
* A = Registry of adjoint modules: pullback backward operators selectable per
  module class, with optional softening for hard-gated layers
* T = Optional temperatures for backward-only softening

### Pseudocode:

Pullback (one-step backward pass):

```
Temporarily install adjoint modules in the model
Activate pullback mode (optionally with temperatures T)
Compute s_u(X), the selected model output
Backpropagate from s_u(X) to X using the adjoints
Restore original modules, including on exceptions
Return the pullback of s_u at X
```

Pullback Ascent (iterative refinement): gradient-ascent updates using pullbacks.

```
Temporarily install adjoint modules in the model
Activate pullback mode (optionally with temperatures T)
Run the ascent loop:
    X^(0) = X
    Repeat K times:
        Compute the pullback of the selected output at X^(t)
        Optionally normalize the direction per example
        X^(t+1) = X^(t) + α · ν_u(X^(t))
        Project or clip X^(t+1), if configured
Restore original modules, including on exceptions
Return X^(K) - X
```

The method is based on the [Semantic Pullbacks paper](https://arxiv.org/abs/2507.22832).
Replacing gradients with pullbacks in saliency, smoothing, and ascent improves
matched explanations quantitatively and qualitatively. Pullback-based variants
also remain competitive with more elaborate methods without specialized
regularization or post-processing. Pullback Ascent can produce sharp,
target-specific perturbations in pixel space in just a few steps.

## Design Considerations:

* Pullbacks are a backward operator, not a standalone attribution method.
  They are an alternative to ordinary gradients: existing gradient-based
  Captum methods (e.g. Saliency, Integrated Gradients, NoiseTunnel) become
  pullback methods by running their existing `attribute` calls inside pullback
  mode. No new attribution classes are required for this.
* Forward invariance is a hard requirement: the model's forward values and
  predictions must be those of the unmodified model. Where backward-only
  softening is used, a straight-through construction preserves the original
  forward value via a detached correction while the differentiable soft value
  supplies the backward Jacobian (`soft + (hard - soft).detach()`).
* The backward rule must be selected at runtime via a scoped context, not via
  permanent global state. Nested contexts must work, and the previous mode
  must be restored on normal exit and on exceptions. A normal model call
  outside the context must continue to produce ordinary gradients, so that
  pullbacks are opt-in.
* Model adaptation (installing adjoint modules) is an in-place mutation of the
  module tree. It should be explicit, idempotent, and reversible so the model
  stays exactly the same before and after the context.
* The adjoint registry is declarative: each adjoint declares which module class
  it substitutes via `@adjoint_params`. Layers whose gradients already
  coincide with pullbacks (e.g. Linear) need no replacement.
* Like gradients, adjoints can be implemented locally and generically, including
  for custom modules: through registered module replacements or custom autograd
  backward rules.
* The implementation must not assume anything about the input modality
  (pixels, embeddings, etc.) or the model architecture: inputs are generic tensors.

## Proposed Captum API Design:

The core contribution is a context manager that selects the backward rule,
together with an adjoint registry and a coverage check for potentially
unsupported modules. Gradient-based attributions require no API changes:
they are used as-is inside the context.

## backward_mode

`backward_mode` is the public mechanism for selecting the backward rule. It
switches the model's backward computation between standard autograd and
pullback adjoints for the duration of the context.

```
class BackwardMode(str, Enum):
    GRADIENT = "gradient"
    PULLBACK = "pullback"

backward_mode(
    mode: BackwardMode,
    temperatures: Optional[Mapping[Type[nn.Module], float]] = None,
    modules: Optional[Union[nn.Module, Iterable[nn.Module]]] = None,
)
```

**Argument Descriptions:**

* `mode` - The backward rule to use inside the context. `GRADIENT` is the
  default and reproduces standard autograd; `PULLBACK` activates the adjoints.
* `temperatures` - Optional mapping of module classes to temperatures for
  backward-only softening (mainly hard-gated layers such as ReLU or MaxPool),
  overriding registered defaults. Supplying custom temperatures in gradient
  mode is an error.
* `modules` - Optional module or list of modules to adapt with adjoints on
  context entry; original modules are restored on exit, including on exceptions.
  Passing the model here avoids manual installation and restoration.

Nested contexts are supported: the previous mode and temperature configuration
are restored on normal exit and on exceptions.

### Composing with existing attributions

Compatible gradient-based Captum attributions run in pullback mode by wrapping
their calls in the context:

```
with backward_mode(BackwardMode.PULLBACK, modules=model):
    attribution = Saliency(model).attribute(inputs, target=target, ...)
```

Methods that perform multiple backward passes (e.g. Integrated Gradients or
NoiseTunnel) inherit the pullback operator for every pass.

## adjoint_params

`adjoint_params` is the registration decorator for adjoint modules. Each
adjoint declares the module class it substitutes and, optionally, a default
temperature. An adjoint module must preserve the original forward behavior
while supplying the pullback backward rule.

```
adjoint_params(original: Type[nn.Module], default_temperature: Optional[float] = None)
```

**Argument Descriptions:**

* `original` - The module class whose backward is replaced, e.g. `nn.GELU`.
* `default_temperature` - Optional temperature for backward-only softening,
  primarily for hard-gated layers (e.g. ReLU or MaxPool); `None` means no
  temperature-controlled softening.

Example:

```
@adjoint_params(original=nn.GELU)
class AdjointGELU(nn.Module):
    def forward(self, x):
        # Original forward value, pullback backward rule
        ...
```

Adjoints for common building blocks such as activations, normalization layers,
pooling, and attention are provided; users can register adjoints for
architecture-specific modules with the same decorator. Modules for which the
ordinary gradient equals the pullback need no replacement.

## check_pullback_coverage

```
check_pullback_coverage(model: nn.Module) -> List[str]
```

This best-effort check flags potentially unsupported modules (including
custom gating or routing implementations) that may need an adjoint but were not
adapted. After installing adjoints, `backward_mode` runs the check when entering
`PULLBACK` mode and emits one concise warning, grouping suspicious module types
and counts, with a link to [custom adjoint registration](#adjoint_params). Known
modules whose gradients coincide with pullbacks are excluded. The check cannot
guarantee full coverage, especially for functional operations hidden inside
`forward`; users can address flagged modules by registering an adjoint or
providing a custom autograd backward rule.
