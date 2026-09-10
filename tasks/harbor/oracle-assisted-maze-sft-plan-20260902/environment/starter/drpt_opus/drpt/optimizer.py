"""
General memory-efficient optimizer for compressed gradient training.

This optimizer works with any compression method that provides a transpose operation,
including GraSS, LoGra, or custom projectors.
"""

import torch
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Callable, Iterable, Tuple
import logging

from .hook import GradientHook
from .compressor import Compressor
from .compression_mode import CompressionMode
from .utils import apply_hadamard_matvec, stochastic_diagonal_estimation
logger = logging.getLogger(__name__)


def zeropower_via_newton_schulz(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    *,
    torch_numerics: bool = False,
) -> torch.Tensor:
    """Muon zeroth-power normalization for scoring or exact fallback updates.

    Optimizer-aware soft ranking needs a smooth, differentiable map, so the
    default keeps the historical float32 computation. The bundled optimizer
    fallback passes torch_numerics=True to reproduce official Muon's deliberate
    bfloat16 Newton--Schulz boundary exactly.
    """
    if matrix.ndim != 2:
        raise ValueError("Muon Newton--Schulz requires a 2D matrix")
    if steps < 0 or steps >= 100:
        raise ValueError("Muon Newton--Schulz steps must be in [0, 100)")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise ValueError("Muon Newton--Schulz eps must be positive and finite")

    if not torch_numerics and steps == 0:
        return matrix

    original_dtype = matrix.dtype
    x = matrix.bfloat16() if torch_numerics else matrix.float()

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        gram_update = torch.addmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
        x = torch.addmm(x, gram_update, x, beta=a)

    if transposed:
        x = x.T
    return x if torch_numerics else x.to(original_dtype)


def _muon_adjust_lr_scale(
    shape: torch.Size,
    adjust_lr_fn: Optional[str] = "original",
    enabled: bool = True,
) -> float:
    """PyTorch Muon-compatible matrix-shape learning-rate multiplier."""
    if not enabled or len(shape) != 2 or shape[1] == 0:
        return 1.0
    mode = "original" if adjust_lr_fn is None else str(adjust_lr_fn).lower()
    if mode in ("", "none", "false", "off", "disabled"):
        return 1.0
    if mode == "original":
        return math.sqrt(max(1.0, shape[0] / shape[1]))
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(max(shape[0], shape[1]))
    raise ValueError(
        "muon_adjust_lr_fn must be one of: original, match_rms_adamw, none"
    )


def _muon_shape_lr_scale(param: torch.Tensor) -> float:
    """Backward-compatible OPUS/Muon matrix-shape learning-rate multiplier."""
    return _muon_adjust_lr_scale(param.shape, "original", True)


class MuonWithAuxAdamW(Optimizer):
    """
    Thin single-Optimizer facade around official Muon plus auxiliary AdamW.

    PyTorch Muon intentionally accepts only eligible 2D hidden-layer matrices
    and its documentation prescribes a separate optimizer for embeddings,
    output heads, biases, norms, and other parameters. Hugging Face Trainer
    expects one optimizer object, so this class exposes those two official
    optimizers through one facade. It does not reimplement Muon when
    torch.optim.Muon can be constructed. The bundled update is used only when
    the official class is unavailable/incompatible; a private test hook exists
    solely for backend-conformance tests.
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
        model: nn.Module,
        lr: float = 1e-3,
        muon_lr: Optional[float] = None,
        aux_adamw_lr: Optional[float] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_eps: float = 1e-7,
        muon_lr_shape_scale: bool = True,
        muon_adjust_lr_fn: Optional[str] = "original",
        muon_backend: str = "auto",
        lora_optimizer: str = "adamw",
        _force_local_for_test: bool = False,
    ):
        named_params = [(n, p) for n, p in named_params if p.requires_grad]
        if not named_params:
            raise ValueError("MuonWithAuxAdamW received no trainable parameters")

        self.model = model
        self._param_to_name: Dict[int, str] = {}
        self._param_to_kind: Dict[int, str] = {}
        self._param_to_state_owner: Dict[int, str] = {}
        self._inner_optimizers: Dict[str, Optimizer] = {}
        self.lora_optimizer = str(lora_optimizer).lower()
        if self.lora_optimizer not in ("adamw", "muon"):
            raise ValueError("lora_optimizer must be either 'adamw' or 'muon'")
        requested_muon_backend = str(muon_backend).lower()
        if requested_muon_backend == "pytorch":
            requested_muon_backend = "torch"
        if requested_muon_backend not in ("auto", "torch", "local"):
            raise ValueError("muon_backend must be one of: auto, torch, local")

        muon_lr = lr if muon_lr is None else muon_lr
        aux_adamw_lr = lr if aux_adamw_lr is None else aux_adamw_lr
        for name, value in (
            ("lr", lr),
            ("muon_lr", muon_lr),
            ("aux_adamw_lr", aux_adamw_lr),
            ("weight_decay", weight_decay),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(muon_momentum)) or float(muon_momentum) < 0:
            raise ValueError("muon_momentum must be finite and non-negative")
        if muon_ns_steps < 0 or muon_ns_steps >= 100:
            raise ValueError("muon_ns_steps must be in [0, 100)")
        if not math.isfinite(float(muon_eps)) or float(muon_eps) <= 0:
            raise ValueError("muon_eps must be positive and finite")

        self.requested_muon_backend = requested_muon_backend
        self.muon_backend_fallback_reason: Optional[str] = None
        self.muon_lr = float(muon_lr)
        self.aux_adamw_lr = float(aux_adamw_lr)
        if muon_adjust_lr_fn is not None:
            muon_adjust_lr_fn = str(muon_adjust_lr_fn).lower()
            if muon_adjust_lr_fn in ("", "none", "false", "off", "disabled"):
                muon_adjust_lr_fn = "none"
            if muon_adjust_lr_fn not in ("original", "match_rms_adamw", "none"):
                raise ValueError(
                    "muon_adjust_lr_fn must be one of: original, match_rms_adamw, none"
                )

        muon_params = []
        adamw_params = []
        for name, param in named_params:
            self._param_to_name[id(param)] = name
            if self._uses_muon(name, param):
                muon_params.append(param)
                self._param_to_kind[id(param)] = "muon"
            else:
                adamw_params.append(param)
                self._param_to_kind[id(param)] = "adamw"

        if not muon_params:
            raise ValueError(
                "MuonWithAuxAdamW found no Muon-eligible trainable 2D hidden-layer "
                "parameters. A run labeled Muon must not silently become AdamW-only. "
                "For LoRA-only training, set optimizer_aware_lora_optimizer='muon'."
            )
        self.muon_param_count = len(muon_params)
        self.aux_adamw_param_count = len(adamw_params)

        torch_muon_cls = getattr(torch.optim, "Muon", None)
        torch_muon_optimizer: Optional[Optimizer] = None
        self.muon_backend = "local"
        if requested_muon_backend == "local" and not _force_local_for_test:
            logger.warning(
                "muon_backend='local' is deprecated for experiment runs and now "
                "still attempts torch.optim.Muon first; bundled Muon is fallback-only."
            )
        if not _force_local_for_test:
            incompatibility = None
            if torch_muon_cls is None:
                incompatibility = "torch.optim.Muon is not available"
            elif not muon_lr_shape_scale or muon_adjust_lr_fn == "none":
                incompatibility = (
                    "torch.optim.Muon does not support disabling its matrix-shape "
                    "learning-rate adjustment"
                )

            if incompatibility is None:
                try:
                    torch_muon_optimizer = torch_muon_cls(
                        muon_params,
                        lr=muon_lr,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        nesterov=muon_nesterov,
                        ns_coefficients=(3.4445, -4.7750, 2.0315),
                        eps=muon_eps,
                        ns_steps=muon_ns_steps,
                        adjust_lr_fn=(
                            None if muon_adjust_lr_fn == "original"
                            else muon_adjust_lr_fn
                        ),
                    )
                except (TypeError, ValueError, RuntimeError, NotImplementedError) as exc:
                    incompatibility = (
                        f"torch.optim.Muon constructor is incompatible: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    self.muon_backend = "torch"

            if incompatibility is not None:
                self.muon_backend_fallback_reason = incompatibility
                logger.warning(
                    "Falling back to the bundled Muon implementation because %s.",
                    incompatibility,
                )
        self.resolved_muon_backend = self.muon_backend

        param_groups = []
        if muon_params:
            param_groups.append({
                "params": muon_params,
                "optimizer_kind": "muon",
                "muon_backend": self.muon_backend,
                "muon_update_rule": self.muon_backend,
                "lr": muon_lr,
                "weight_decay": weight_decay,
                "momentum": muon_momentum,
                "nesterov": muon_nesterov,
                "muon_ns_steps": muon_ns_steps,
                "muon_eps": muon_eps,
                "muon_lr_shape_scale": muon_lr_shape_scale,
                "ns_coefficients": (3.4445, -4.7750, 2.0315),
                "adjust_lr_fn": None if muon_adjust_lr_fn == "original" else muon_adjust_lr_fn,
            })
        if adamw_params:
            param_groups.append({
                "params": adamw_params,
                "optimizer_kind": "adamw",
                "lr": aux_adamw_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            })

        super().__init__(param_groups, defaults={})

        if torch_muon_optimizer is not None:
            self._inner_optimizers["muon"] = torch_muon_optimizer
            for param in muon_params:
                self._param_to_state_owner[id(param)] = "muon"

        if adamw_params:
            self._inner_optimizers["adamw"] = torch.optim.AdamW(
                adamw_params,
                lr=aux_adamw_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            for param in adamw_params:
                self._param_to_state_owner[id(param)] = "adamw"

        logger.info("Initialized MuonWithAuxAdamW optimizer")
        logger.info(
            "  Muon backend: requested=%s, resolved=%s",
            self.requested_muon_backend,
            self.muon_backend,
        )
        if self.muon_backend_fallback_reason:
            logger.info("  Muon fallback reason: %s", self.muon_backend_fallback_reason)
        logger.info(f"  Muon params: {len(muon_params)}")
        logger.info(f"  AdamW params: {len(adamw_params)}")
        logger.info(f"  Muon LR: {muon_lr}; auxiliary AdamW LR: {aux_adamw_lr}")
        logger.info(
            f"  Muon momentum: {muon_momentum}, nesterov: {muon_nesterov}, "
            f"NS steps: {muon_ns_steps}"
        )
        logger.info(
            f"  Muon shape LR scaling: {muon_lr_shape_scale}, "
            f"adjust_lr_fn: {muon_adjust_lr_fn}"
        )
        logger.info(f"  LoRA optimizer policy: {self.lora_optimizer}")

    def _module_for_param(self, name: str) -> Optional[nn.Module]:
        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            return None
        module_name = parts[0]
        module = self.model
        for attr in module_name.split("."):
            if not hasattr(module, attr):
                return None
            module = getattr(module, attr)
        return module

    def _uses_muon(self, name: str, param: torch.nn.Parameter) -> bool:
        if param.ndim != 2:
            return False
        if name.endswith(".bias"):
            return False
        lowered = name.lower()
        if "norm" in lowered or "layernorm" in lowered or "ln_" in lowered:
            return False
        module_leaf = lowered.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        output_modules = {
            "lm_head", "embed_out", "unembed", "output", "output_layer",
            "output_projection", "classifier", "score",
        }
        if (
            "embed" in lowered
            or "embedding" in lowered
            or module_leaf in output_modules
        ):
            return False
        if "lora_" in lowered or ".lora" in lowered:
            return self.lora_optimizer == "muon"
        module = self._module_for_param(name)
        if isinstance(module, nn.Embedding):
            return False
        return self._is_transformer_muon_matrix(name, module)

    @staticmethod
    def _is_transformer_muon_matrix(name: str, module: Optional[nn.Module] = None) -> bool:
        """Return whether an already-filtered parameter is a hidden weight matrix.

        The historical helper name is retained for compatibility. Restricting
        this to a hard-coded list of Transformer path names caused otherwise
        valid custom/generic Linear layers to be silently sent to AdamW.
        """
        lowered = name.lower()
        if not lowered.endswith(".weight"):
            return False
        return module is None or isinstance(module, nn.Linear) or hasattr(module, "weight")

    def get_param_name(self, param: torch.nn.Parameter) -> Optional[str]:
        return self._param_to_name.get(id(param))

    def get_param_optimizer_kind(self, param: torch.nn.Parameter) -> str:
        return self._param_to_kind.get(id(param), "adamw")

    def get_param_group(self, param: torch.nn.Parameter) -> Optional[dict]:
        """Return the exact facade group that controls the parameter."""
        for group in self.param_groups:
            if any(candidate is param for candidate in group.get("params", ())):
                return group
        return None

    def get_param_state(self, param: torch.nn.Parameter) -> Dict[str, Any]:
        owner = self._param_to_state_owner.get(id(param))
        if owner in self._inner_optimizers:
            return self._inner_optimizers[owner].state.get(param, {})
        return self.state.get(param, {})

    def get_muon_update_rule(self) -> str:
        return self.muon_backend

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable backend/group details for run metadata."""
        muon_optimizer = self._inner_optimizers.get("muon")
        adamw_optimizer = self._inner_optimizers.get("adamw")
        muon_group = self._wrapper_group("muon") or {}
        adamw_group = self._wrapper_group("adamw") or {}
        if muon_optimizer is None:
            muon_impl = "drpt.optimizer.MuonWithAuxAdamW.local_fallback"
        else:
            cls = type(muon_optimizer)
            muon_impl = f"{cls.__module__}.{cls.__qualname__}"
        if adamw_optimizer is None:
            adamw_impl = None
        else:
            cls = type(adamw_optimizer)
            adamw_impl = f"{cls.__module__}.{cls.__qualname__}"
        return {
            "optimizer_runtime_class": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "muon_backend_requested": self.requested_muon_backend,
            "muon_backend_resolved": self.resolved_muon_backend,
            "muon_backend_fallback_reason": self.muon_backend_fallback_reason,
            "muon_optimizer_implementation": muon_impl,
            "aux_adamw_optimizer_implementation": adamw_impl,
            "muon_learning_rate_resolved": float(muon_group["lr"]),
            "aux_adamw_learning_rate_resolved": (
                float(adamw_group["lr"]) if adamw_group else None
            ),
            "muon_parameter_tensor_count": self.muon_param_count,
            "aux_adamw_parameter_tensor_count": self.aux_adamw_param_count,
            "lora_optimizer_resolved": self.lora_optimizer,
        }

    def _wrapper_group(self, kind: str) -> Optional[dict]:
        for group in self.param_groups:
            if group.get("optimizer_kind") == kind:
                return group
        return None

    def _sync_inner_groups(self) -> None:
        muon_group = self._wrapper_group("muon")
        muon_optimizer = self._inner_optimizers.get("muon")
        if muon_group is not None and muon_optimizer is not None:
            inner_group = muon_optimizer.param_groups[0]
            for key in (
                "lr", "weight_decay", "momentum", "nesterov",
                "ns_coefficients", "eps", "ns_steps", "adjust_lr_fn",
            ):
                source_key = "muon_eps" if key == "eps" and "muon_eps" in muon_group else key
                source_key = "muon_ns_steps" if key == "ns_steps" and "muon_ns_steps" in muon_group else source_key
                if source_key in muon_group:
                    inner_group[key] = muon_group[source_key]

        adamw_group = self._wrapper_group("adamw")
        adamw_optimizer = self._inner_optimizers.get("adamw")
        if adamw_group is not None and adamw_optimizer is not None:
            inner_group = adamw_optimizer.param_groups[0]
            for key in ("lr", "betas", "eps", "weight_decay"):
                if key in adamw_group:
                    inner_group[key] = adamw_group[key]

    def _sync_state_views_from_inner(self) -> None:
        for optimizer in self._inner_optimizers.values():
            for param, state in optimizer.state.items():
                self.state[param] = state

    def _sync_inner_state_from_wrapper(self) -> None:
        for kind, optimizer in self._inner_optimizers.items():
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    if param in self.state:
                        optimizer.state[param] = self.state[param]

    def state_dict(self):
        self._sync_state_views_from_inner()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        muon_group = self._wrapper_group("muon")
        if muon_group is not None:
            muon_group["muon_backend"] = self.muon_backend
            muon_group["muon_update_rule"] = self.muon_backend
        self._sync_inner_state_from_wrapper()
        self._sync_inner_groups()
        return result

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._sync_inner_groups()

        for group in self.param_groups:
            kind = group.get("optimizer_kind", "adamw")
            if kind == "muon" and self.muon_backend == "torch":
                continue
            if kind == "adamw" and "adamw" in self._inner_optimizers:
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                if kind == "muon":
                    self._step_muon(param, group)
                else:
                    self._step_adamw(param, group)

        muon_optimizer = self._inner_optimizers.get("muon")
        if muon_optimizer is not None:
            muon_optimizer.step()

        adamw_optimizer = self._inner_optimizers.get("adamw")
        if adamw_optimizer is not None:
            adamw_optimizer.step()

        self._sync_state_views_from_inner()
        return loss

    def _step_muon(self, param: torch.nn.Parameter, group: dict) -> None:
        grad = param.grad
        if grad is None:
            return
        if grad.ndim != 2:
            raise RuntimeError("Muon group received a non-2D gradient")

        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)

        momentum = group["momentum"]
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1.0 - momentum)

        if group.get("nesterov", True):
            q = grad.lerp(buf, momentum)
        else:
            q = buf
        update = zeropower_via_newton_schulz(
            q,
            steps=group["muon_ns_steps"],
            eps=group["muon_eps"],
            torch_numerics=True,
        )

        lr_scale = _muon_adjust_lr_scale(
            param.shape,
            group.get("adjust_lr_fn", "original"),
            group.get("muon_lr_shape_scale", True),
        )

        if group["weight_decay"] != 0:
            param.mul_(1.0 - group["lr"] * group["weight_decay"])
        param.add_(update, alpha=-(group["lr"] * lr_scale))

    def _step_adamw(self, param: torch.nn.Parameter, group: dict) -> None:
        grad = param.grad
        if grad is None:
            return

        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)

        beta1, beta2 = group["betas"]
        state["step"] += 1
        state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = group["lr"] / bias_correction1
        denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

        if group["weight_decay"] != 0:
            param.mul_(1.0 - group["lr"] * group["weight_decay"])
        param.addcdiv_(state["exp_avg"], denom, value=-step_size)


# Backward-compatible import name. The runtime is a facade over official Muon
# and auxiliary AdamW, not a custom hybrid Muon update when PyTorch supports it.
HybridMuonAdamW = MuonWithAuxAdamW


class MeSOAdamW(Optimizer):
    """
    AdamW optimizer that maintains states in compressed gradient space.

    This is a general implementation that works with any compression method
    that provides a transpose operation (GraSS, LoGra, etc.).

    The optimizer:
    1. Receives compressed gradients from hooks
    2. Maintains first and second moments in compressed space
    3. Applies optimizer updates in compressed space
    4. Uses transpose to project updates back to parameter space

    Args:
        params: Model parameters to optimize (only Linear layer params will be optimized via compression)
        grad_hook: GradientHook instance for gradient compression and transpose
        lr: Learning rate
        betas: Coefficients for computing running averages of gradient and its square
        eps: Term added to denominator to improve numerical stability
        weight_decay: Weight decay coefficient (L2 penalty)
        compressed_layer_names: Optional list of layer names to apply compression.
                                If None, will attempt compression on all layers.
        stochastic_num_samples: Number of samples for stochastic second moment transform.
                                If > 0: use stochastic estimation (efficient, approximate).
                                If None or 0: use exact (M ⊙ M) @ v transformation (expensive, exact).
                                Default: 100 (stochastic)
    """

    def __init__(
        self,
        params,
        grad_hook: GradientHook,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        compressed_layer_names: Optional[List[str]] = None,
        stochastic_num_samples: Optional[int] = 100
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.grad_hook = grad_hook
        self.compressed_layer_names = compressed_layer_names or grad_hook.layer_names
        self.stochastic_num_samples = stochastic_num_samples

        # Create mapping from parameter to layer name
        self._param_to_layer_name = {}
        self._setup_param_mapping()

        # Determine transformation mode from stochastic_num_samples
        use_stochastic = stochastic_num_samples is not None and stochastic_num_samples > 0

        logger.info(f"Initialized MeSOAdamW optimizer")
        logger.info(f"  Compressed layers: {len(self.compressed_layer_names)}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Betas: {betas}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Second moment transform: {'stochastic' if use_stochastic else 'exact'}")
        if use_stochastic:
            logger.info(f"  Stochastic samples: {stochastic_num_samples}")

    def _setup_param_mapping(self):
        """
        Create mapping from parameters to layer names.

        This is crucial for knowing which compressed gradient corresponds to which parameter.
        """
        model = self.grad_hook.model

        for layer_name in self.compressed_layer_names:
            # Navigate to the module using layer_name
            module = model
            for attr in layer_name.split('.'):
                module = getattr(module, attr)

            # Map weight and bias parameters
            if hasattr(module, 'weight') and module.weight is not None:
                self._param_to_layer_name[id(module.weight)] = (layer_name, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                self._param_to_layer_name[id(module.bias)] = (layer_name, 'bias')


    def _get_layer_info(self, param):
        """
        Get layer name and parameter type for a given parameter.

        Returns:
            tuple: (layer_name, param_type) or (None, None) if not found
        """
        param_id = id(param)
        return self._param_to_layer_name.get(param_id, (None, None))

    def get_current_step(self) -> int:
        """
        Get the current training step from optimizer state.

        Returns:
            Current step number (0 if no state exists yet)
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'step' in state:
                    return state['step']
        return 0

    def _is_sparsifier_random_mask(self, old_compressor: Compressor, new_compressor: Compressor) -> bool:
        """
        Check if random_mask sparsifier is used in both old and new compressors.

        This generally allows us to work in intermediate space k' instead of full space D,
        providing significant speedup even for optimizer states transformation.
        """
        from .projection import ProjectionType

        old_sparsifier = old_compressor.sparsifier
        new_sparsifier = new_compressor.sparsifier

        old_s1, old_s2 = old_sparsifier.sparsifier_comp
        new_s1, new_s2 = new_sparsifier.sparsifier_comp

        # Check if all sparsifier components use random_mask
        sparsifiers_use_random_mask = (
            old_s1.proj_type == ProjectionType.random_mask and
            old_s2.proj_type == ProjectionType.random_mask and
            new_s1.proj_type == ProjectionType.random_mask and
            new_s2.proj_type == ProjectionType.random_mask
        )

        return sparsifiers_use_random_mask

    def _transform_first_moment(
        self,
        state_old: torch.Tensor,
        layer_idx: int,
        old_compressor: Compressor
    ) -> torch.Tensor:
        """
        Transform first moment (momentum) from old subspace to new subspace during compressor refresh:
            m_new = M @ m_old where M = P_new @ P_old^T

        When random_mask sparsifiers is used, we provide an optimized method that avoids full
        materialization in the M's D-dimensional space by working in intermediate space.

        Args:
            state_old: Old first moment in compressed space [k]
            layer_idx: Index of the layer
            old_compressor: Old Compressor (before refresh) for transformation

        Returns:
            Transformed first moment in new compressed space [k]
        """
        new_compressor = self.grad_hook.update_compressors[layer_idx]

        norm_old = state_old.norm().item()

        # Add batch dimension if needed [k] -> [1, k]
        if state_old.dim() == 1:
            state_old_batch = state_old.unsqueeze(0)
        else:
            state_old_batch = state_old

        if self._is_sparsifier_random_mask(old_compressor, new_compressor):
            # Optimized transform for random_mask: works in intermediate space k' to avoid
            # full D-dimensional materialization
            # Mathematical flow:
            # 1. Map from compressed to old intermediate: m_old_inter = Proj_old^T @ m_old
            # 2. Apply sparse index mapping: m_new_inter = Spars_new @ Spars_old^T @ m_old_inter
            # 3. Map from new intermediate to new compressed: m_new = Proj_new @ m_new_inter

            # Extract components
            old_sparsifier = old_compressor.sparsifier
            new_sparsifier = new_compressor.sparsifier
            old_projector = old_compressor.projector
            new_projector = new_compressor.projector

            # Get sparsifier indices
            old_s1, old_s2 = old_sparsifier.sparsifier_comp
            new_s1, new_s2 = new_sparsifier.sparsifier_comp

            old_idx1, old_idx2 = old_s1.active_indices, old_s2.active_indices
            new_idx1, new_idx2 = new_s1.active_indices, new_s2.active_indices

            k1, k2 = len(old_idx1), len(old_idx2)

            # Step 1: Transform state_old from compressed to old intermediate space
            # [1, k] → Proj_old^T → [1, k']
            m_old_inter = old_projector.transpose(state_old_batch)  # [1, k']

            # Step 2: Reshape to matrix form for index operations
            m_old_matrix = m_old_inter.squeeze(0).reshape(k1, k2)

            # Step 3: Apply sparse index mapping: Spars_new @ Spars_old^T
            # Build index mappings for O(1) lookup
            old_to_compressed_1 = {idx.item(): i for i, idx in enumerate(old_idx1)}
            old_to_compressed_2 = {idx.item(): i for i, idx in enumerate(old_idx2)}
            new_to_compressed_1 = {idx.item(): i for i, idx in enumerate(new_idx1)}
            new_to_compressed_2 = {idx.item(): i for i, idx in enumerate(new_idx2)}

            # Find common indices and build mapping lists
            common_idx1 = set(old_to_compressed_1.keys()) & set(new_to_compressed_1.keys())
            common_idx2 = set(old_to_compressed_2.keys()) & set(new_to_compressed_2.keys())
            mapping_list_1 = [(new_to_compressed_1[idx], old_to_compressed_1[idx]) for idx in common_idx1]
            mapping_list_2 = [(new_to_compressed_2[idx], old_to_compressed_2[idx]) for idx in common_idx2]

            # Pre-compute index tensors for vectorized operations
            new_pos1_tensor = torch.tensor([new_pos1 for new_pos1, _ in mapping_list_1], device=state_old_batch.device)
            old_pos1_tensor = torch.tensor([old_pos1 for _, old_pos1 in mapping_list_1], device=state_old_batch.device)
            new_pos2_tensor = torch.tensor([new_pos2 for new_pos2, _ in mapping_list_2], device=state_old_batch.device)
            old_pos2_tensor = torch.tensor([old_pos2 for _, old_pos2 in mapping_list_2], device=state_old_batch.device)

            # Pre-compute meshgrid indices for 2D indexing
            new_i, new_j = torch.meshgrid(new_pos1_tensor, new_pos2_tensor, indexing='ij')
            old_i, old_j = torch.meshgrid(old_pos1_tensor, old_pos2_tensor, indexing='ij')

            # Initialize result matrix in new intermediate space
            m_new_matrix = torch.zeros(k1, k2, device=state_old_batch.device, dtype=state_old_batch.dtype)

            # Vectorized mapping through index intersection
            m_new_matrix[new_i, new_j] = m_old_matrix[old_i, old_j]

            # Step 4: Map from new intermediate to new compressed space
            m_new_inter = m_new_matrix.reshape(1, -1)  # [1, k']
            m_new = new_projector.forward(m_new_inter)  # [1, k]
            state_new = m_new.squeeze(0)
        else:
            # Transform: m_new = M @ m_old where M = P_new @ P_old^T
            full = old_compressor.transpose(state_old_batch)
            state_new_batch = new_compressor.forward(full)
            state_new = state_new_batch.squeeze(0)

        # Rescale to match original norm
        norm_new = state_new.norm().item()
        if norm_new > 1e-10 and norm_old > 1e-10:
            scale_factor = norm_old / norm_new
            state_new.mul_(scale_factor)

        return state_new

    def _second_moment_postprocessing(
        self,
        v_new: torch.Tensor,
        state_old: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply common post-processing to second moment transformation result.

        This preserves the effective denominator magnitude (RMS of sqrt(second_moment))
        to maintain similar learning rate scale after transformation.

        Args:
            v_new: Transformed second moment
            state_old: Original second moment (for RMS reference)

        Returns:
            Post-processed second moment
        """
        # Sanity check: clamp negative values to zero (from numerical errors in stochastic estimation)
        v_new.clamp_(min=0)

        # Preserve effective denominator magnitude
        # In AdamW: denom = sqrt(exp_avg_sq) + eps
        # We want: RMS(sqrt(v_new)) ≈ RMS(sqrt(state_old))
        # This maintains the typical learning rate scale

        # Compute RMS of sqrt on the ACTUAL transformed values (including zeros)
        # Don't clamp to floor first - that would artificially inflate the RMS
        old_rms_sqrt = torch.sqrt(state_old.clamp(min=0)).pow(2).mean().sqrt()
        new_rms_sqrt = torch.sqrt(v_new.clamp(min=0)).pow(2).mean().sqrt()

        # Scale to preserve RMS if old state is meaningful
        # We allow new_rms_sqrt to be very small (happens when most transformed values are zeros)
        if old_rms_sqrt > 1e-10 and new_rms_sqrt > 1e-15:
            # Scale to preserve RMS(sqrt(second_moment))
            # We want: sqrt(v_new_scaled) = sqrt(v_new) * scale
            # So: v_new_scaled = v_new * scale^2
            scale = old_rms_sqrt / new_rms_sqrt
            v_new.mul_(scale.pow(2))

        # Apply floor clamp (after preserving the true RMS)
        # This prevents division by zero in AdamW without affecting the RMS calculation
        v_new.clamp_(min=1e-10)

        return v_new

    def _transform_second_moment_exact(
        self,
        state_old: torch.Tensor,
        layer_idx: int,
        old_compressor: Compressor,
        chunk_size: int = 256
    ) -> torch.Tensor:
        """
        Transform second moment from old subspace to new subspace during compressor refresh:
            v_new = (M ⊙ M) @ v_old,
        where M = P_new @ P_old^T and ⊙ is element-wise product.

        Given compressed gradients ĝ_old = P_old @ g and ĝ_new = P_new @ g,
        and the relationship ĝ_new ≈ M @ ĝ_old, the second moment transforms as:
            E[ĝ_new²] = E[(M @ ĝ_old)²] = (M ⊙ M) @ E[ĝ_old²]

        Args:
            state_old: Old second moment in compressed space [k]
            layer_idx: Index of the layer
            old_compressor: Old Compressor (before refresh)
            chunk_size: Number of basis vectors to process per batch (for memory efficiency).
                       If None, automatically determined based on layer size.

        Returns:
            Transformed second moment in new compressed space [k]
        """
        new_compressor = self.grad_hook.update_compressors[layer_idx]

        if self._is_sparsifier_random_mask(old_compressor, new_compressor):
            # Get sparsifier indices
            old_s1, old_s2 = old_compressor.sparsifier.sparsifier_comp
            new_s1, new_s2 = new_compressor.sparsifier.sparsifier_comp

            old_idx1, old_idx2 = old_s1.active_indices, old_s2.active_indices
            new_idx1, new_idx2 = new_s1.active_indices, new_s2.active_indices

            k1, k2 = len(old_idx1), len(old_idx2)
            k_prime = k1 * k2

            # Step 1: Transform state_old from compressed to intermediate space
            state_old_inter = old_compressor.projector.transpose(state_old.unsqueeze(0)).squeeze(0) # [k']

            # Step 2: Reshape to matrix form
            V_old_matrix = state_old_inter.reshape(k1, k2)  # [k_1', k_2']

            # Step 3: Apply sparse index transformation (T ⊙ T) @ V_old where T = Spars_new @ Spars_old^T
            # Build index mappings
            old_to_compressed_1 = {idx.item(): i for i, idx in enumerate(old_idx1)}
            old_to_compressed_2 = {idx.item(): i for i, idx in enumerate(old_idx2)}

            # For each position in new intermediate space, compute contribution from old space
            V_inter_new_matrix = torch.zeros(k1, k2, device=state_old.device, dtype=state_old.dtype)

            # Iterate over new intermediate positions
            for i_new in range(k1):
                idx1_new = new_idx1[i_new].item()
                for j_new in range(k2):
                    idx2_new = new_idx2[j_new].item()

                    # Check if this index exists in old subspace
                    if idx1_new in old_to_compressed_1 and idx2_new in old_to_compressed_2:
                        i_old = old_to_compressed_1[idx1_new]
                        j_old = old_to_compressed_2[idx2_new]
                        # Direct copy for overlapping indices (T[i,j] = 1)
                        V_inter_new_matrix[i_new, j_new] = V_old_matrix[i_old, j_old]

            # Flatten to vector form
            V_inter_new = V_inter_new_matrix.reshape(-1)  # [k']

            # Step 4: Compute (M_proj ⊙ M_proj) @ V_inter_new using basis vectors
            # where M_proj = Proj_new @ Proj_old^T
            chunk_size = min(chunk_size, k_prime)
            v_new = apply_hadamard_matvec(
                M_proj_forward=new_compressor.projector.forward,
                source_vector=V_inter_new,
                result_size=state_old.shape[0],
                chunk_size=chunk_size,
                source_size=k_prime,
                device=state_old.device,
                dtype=state_old.dtype
            )

        else:
            # General case: compute (M ⊙ M) @ state_old where M = P_new @ P_old^T
            # We use a composition function for M_proj_forward
            def compose_forward(I_chunk):
                # P_old^T @ I_chunk^T = P_old^T @ [e_i, e_{i+1}, ...]
                full_chunk = old_compressor.transpose(I_chunk)  # [chunk_size, d]
                # P_new @ full_chunk
                return new_compressor.forward(full_chunk)  # [chunk_size, k]

            v_new = apply_hadamard_matvec(
                M_proj_forward=compose_forward,
                source_vector=state_old,
                result_size=state_old.shape[0],
                chunk_size=chunk_size,
                source_size=state_old.shape[0],
                device=state_old.device,
                dtype=state_old.dtype
            )

        # Apply common post-processing (clamping and norm adjustment)
        v_new = self._second_moment_postprocessing(v_new, state_old)

        return v_new

    def _transform_second_moment_stochastic(
        self,
        state_old: torch.Tensor,
        layer_idx: int,
        old_compressor: Compressor,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Transform second moment from old subspace to new subspace during compressor refresh using a probabilistic diagonal estimator.
            - For A = M V_old M^T, we want v_new = diag(A)
            - Probabilistic diagonal estimator: diag(A) ≈ (1/N) Σ z_i ⊙ (A z_i)
            - Compute A z_i = M V_old M^T z_i as: M (V_old (M^T z_i))

        Args:
            state_old: Old second moment in compressed space [k]
            layer_idx: Index of the layer
            old_compressor: Old Compressor (before refresh)
            num_samples: Number of random samples for approximation (default: 100)

        Returns:
            Approximated second moment in new compressed space [k]
        """
        new_compressor = self.grad_hook.update_compressors[layer_idx]

        if self._is_sparsifier_random_mask(old_compressor, new_compressor):
            # Optimized path for random_mask: work in intermediate space

            # Prepare intermediate space setup
            old_s1, old_s2 = old_compressor.sparsifier.sparsifier_comp
            new_s1, new_s2 = new_compressor.sparsifier.sparsifier_comp
            old_idx1, old_idx2 = old_s1.active_indices, old_s2.active_indices
            new_idx1, new_idx2 = new_s1.active_indices, new_s2.active_indices
            k1, k2 = len(old_idx1), len(old_idx2)

            # Transform state_old to intermediate space and reshape to matrix
            state_old_inter = old_compressor.projector.transpose(state_old.unsqueeze(0)).squeeze(0)
            V_old_matrix = state_old_inter.reshape(k1, k2)

            # Build index mappings for sparse operations
            old_to_compressed_1 = {idx.item(): i for i, idx in enumerate(old_idx1)}
            old_to_compressed_2 = {idx.item(): i for i, idx in enumerate(old_idx2)}
            new_to_compressed_1 = {idx.item(): i for i, idx in enumerate(new_idx1)}
            new_to_compressed_2 = {idx.item(): i for i, idx in enumerate(new_idx2)}

            # Build mapping list for efficient indexing
            common_idx1 = set(old_to_compressed_1.keys()) & set(new_to_compressed_1.keys())
            common_idx2 = set(old_to_compressed_2.keys()) & set(new_to_compressed_2.keys())
            mapping_list_1 = [(new_to_compressed_1[idx], old_to_compressed_1[idx]) for idx in common_idx1]
            mapping_list_2 = [(new_to_compressed_2[idx], old_to_compressed_2[idx]) for idx in common_idx2]

            # Pre-compute index tensors for vectorized operations
            new_pos1_tensor = torch.tensor([new_pos1 for new_pos1, _ in mapping_list_1], device=state_old.device)
            old_pos1_tensor = torch.tensor([old_pos1 for _, old_pos1 in mapping_list_1], device=state_old.device)
            new_pos2_tensor = torch.tensor([new_pos2 for new_pos2, _ in mapping_list_2], device=state_old.device)
            old_pos2_tensor = torch.tensor([old_pos2 for _, old_pos2 in mapping_list_2], device=state_old.device)

            # Pre-compute meshgrid indices for 2D indexing
            new_i, new_j = torch.meshgrid(new_pos1_tensor, new_pos2_tensor, indexing='ij')
            old_i, old_j = torch.meshgrid(old_pos1_tensor, old_pos2_tensor, indexing='ij')

            # Define computation function: compute M V_old M^T z
            def compute_Mz(z: torch.Tensor) -> torch.Tensor:
                """Compute M V_old M^T z in intermediate space (vectorized)."""
                # Map z to new intermediate space
                z_inter = new_compressor.projector.transpose(z)  # [1, k']
                z_inter_matrix = z_inter.squeeze(0).reshape(k1, k2)

                # Apply sparse mapping: Spars_old @ (Spars_new^T @ z_inter)
                z_mapped_to_old = torch.zeros(k1, k2, device=state_old.device, dtype=state_old.dtype)
                z_mapped_to_old[old_i, old_j] = z_inter_matrix[new_i, new_j]

                # Element-wise multiply with V_old_matrix
                B_i_matrix = V_old_matrix * z_mapped_to_old

                # Apply sparse mapping back: Spars_new @ (Spars_old^T @ B_i)
                B_i_mapped_to_new = torch.zeros(k1, k2, device=state_old.device, dtype=state_old.dtype)
                B_i_mapped_to_new[new_i, new_j] = B_i_matrix[old_i, old_j]

                # Project to new compressed space
                B_i_inter = B_i_mapped_to_new.reshape(1, -1)
                y_i = new_compressor.projector.forward(B_i_inter)
                return y_i

            v_new = stochastic_diagonal_estimation(
                compute_Mz_func=compute_Mz,
                result_size=state_old.shape[0],
                num_samples=num_samples,
                device=state_old.device,
                dtype=state_old.dtype,
                seed=42
            )

        else:
            # General case: compute M V_old M^T directly
            # V_old is diagonal, so we use state_old as the diagonal entries
            v_old_diag = state_old  # [k]

            # Define computation function: compute M V_old M^T z
            def compute_Mz(z: torch.Tensor) -> torch.Tensor:
                """Compute M V_old M^T z = M (V_old (M^T z))."""
                # Step 1: M^T z = P_old @ (P_new^T @ z)
                step_a1 = new_compressor.transpose(z)  # [1, d]
                a_i = old_compressor.forward(step_a1)  # [1, k]

                # Step 2: V_old * (M^T z) - element-wise product
                b_i = v_old_diag[None, :] * a_i  # [1, k]

                # Step 3: M b = P_new @ (P_old^T @ b)
                step_c1 = old_compressor.transpose(b_i)  # [1, d]
                y_i = new_compressor.forward(step_c1)  # [1, k]

                return y_i

            v_new = stochastic_diagonal_estimation(
                compute_Mz_func=compute_Mz,
                result_size=state_old.shape[0],
                num_samples=num_samples,
                device=state_old.device,
                dtype=state_old.dtype,
                seed=42 + layer_idx
            )

        # Apply common post-processing (clamping and norm adjustment)
        v_new = self._second_moment_postprocessing(v_new, state_old)

        return v_new

    def _transform_second_moment(
        self,
        state_old: torch.Tensor,
        layer_idx: int,
        old_compressor: Compressor
    ) -> torch.Tensor:
        """
        Transform second moment (variance) from old subspace to new subspace during compressor refresh.

        Two transformation methods controlled by stochastic_num_samples:

        1. Stochastic estimation (stochastic_num_samples > 0, default):
           v_new ≈ E[z ⊙ (M V_old M^T z)] using random sampling

        2. Exact (stochastic_num_samples = None or 0):
           v_new = (M ⊙ M) @ v_old

        Args:
            state_old: Old second moment in compressed space [k]
            layer_idx: Index of the layer
            old_compressor: Old Compressor (before refresh) for transformation

        Returns:
            Transformed second moment in new compressed space [k]
        """
        if self.stochastic_num_samples is not None and self.stochastic_num_samples > 0:
            return self._transform_second_moment_stochastic(
                state_old, layer_idx, old_compressor,
                num_samples=self.stochastic_num_samples
            )
        else:
            return self._transform_second_moment_exact(
                state_old, layer_idx, old_compressor
            )

    def refresh_compressors_if_needed(self) -> int:
        """
        Refresh compressors if needed based on current step.

        This should be called before forward/backward passes to ensure
        gradients are computed with the correct (refreshed) projectors.

        Returns:
            Number of compressors refreshed
        """
        current_step = self.get_current_step()

        # Add 1 because we want to refresh for the NEXT step
        next_step = current_step + 1

        logger.info(f"Checking compressor refresh: current_step={current_step}, next_step={next_step}")

        num_refreshed, old_compressors = self.grad_hook.refresh_compressors(next_step)

        if num_refreshed > 0:
            # Only transform optimizer states when using FULL compression mode (compressed states)
            # In SCORE_ONLY mode, optimizer states are in full space and don't need transformation
            if self.grad_hook.compression_mode.uses_compressed_updates:
                logger.info(f"Refreshed {num_refreshed} compressors, now transforming optimizer states...")
                # Transform optimizer states after refresh
                self._transform_optimizer_states(old_compressors)
            else:
                logger.info(f"Refreshed {num_refreshed} compressors (SCORE_ONLY mode: skipping state transformation)")
        else:
            logger.debug(f"No compressor refresh needed at step {next_step}")

        return num_refreshed

    def _transform_optimizer_states(self, old_compressors: Compressor) -> int:
        """
        Transform all optimizer states after compressor refresh.

        This is called automatically after refresh to stabilize training by
        mapping first and second moments from the old subspace to the new one.

        Args:
            old_compressors: List of old Compressor (before refresh)

        Returns:
            Number of layers with transformed states
        """
        num_transformed = 0
        num_no_state = 0
        num_no_old_compressor = 0

        for group in self.param_groups:
            for p in group['params']:
                # Get layer information
                layer_name, param_type = self._get_layer_info(p)

                # Only transform compressed layers with existing states
                if layer_name is None or layer_name not in self.compressed_layer_names:
                    continue

                state = self.state.get(p, {})
                if len(state) == 0 or 'exp_avg' not in state:
                    num_no_state += 1
                    continue

                # Get layer index
                layer_idx = self.grad_hook.layer_name_to_idx[layer_name]

                # Check if old compressor is available
                old_compressor = old_compressors[layer_idx] if layer_idx < len(old_compressors) else None

                if old_compressor is None:
                    num_no_old_compressor += 1
                    logger.debug(f"Skipping {layer_name}/{param_type}: no old compressor at index {layer_idx}")
                    continue

                try:
                    # Transform first moment
                    state['exp_avg'] = self._transform_first_moment(state['exp_avg'], layer_idx, old_compressor)

                    # Transform second moment
                    state['exp_avg_sq'] = self._transform_second_moment(state['exp_avg_sq'], layer_idx, old_compressor)

                    # Check for numerical issues
                    has_nan_first = torch.isnan(state['exp_avg']).any() or torch.isinf(state['exp_avg']).any()
                    has_nan_second = torch.isnan(state['exp_avg_sq']).any() or torch.isinf(state['exp_avg_sq']).any()
                    has_negative_second = (state['exp_avg_sq'] < 0).any()

                    if has_nan_first or has_nan_second or has_negative_second:
                        logger.warning(f"Numerical issues in {layer_name}/{param_type}: nan_first={has_nan_first}, nan_second={has_nan_second}, negative_second={has_negative_second}")

                    num_transformed += 1

                except Exception as e:
                    logger.error(f"Failed to transform state for layer {layer_name}/{param_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with other layers even if one fails
                    continue

        logger.info(f"State transformation complete: transformed={num_transformed}, no_state={num_no_state}, no_old_compressor={num_no_old_compressor}")

        return num_transformed

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure to reevaluate the model and return the loss

        Returns:
            Optional loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update each parameter group
        for group in self.param_groups:
            for p in group['params']:
                # Get layer information
                layer_name, param_type = self._get_layer_info(p)

                # Check if this parameter should use compressed optimization
                if layer_name is not None:
                    layer_idx = self.grad_hook.layer_name_to_idx.get(layer_name)
                    if layer_idx is not None:
                        # Get compressed gradient from the weight parameter's _compressed_grad attribute
                        # Note: Compressed grad is always stored on weight (even for bias params)
                        # because it contains both weight and bias gradients combined
                        if param_type == 'weight':
                            compressed_grad = getattr(p, '_compressed_grad', None)
                        else:
                            # For bias, get compressed grad from the weight of the same layer
                            module = self.grad_hook.layer_name_to_module.get(layer_name)
                            compressed_grad = getattr(module.weight, '_compressed_grad', None) if module else None

                        if compressed_grad is not None:
                            # Use compressed gradient pathway
                            # Note: p.grad will be None for hooked layers (intentional)
                            self._step_compressed(p, compressed_grad, group, layer_name)
                            continue

                # Use standard gradient pathway (for non-compressed layers)
                # Skip if gradient is None
                if p.grad is None:
                    continue
                self._step_standard(p, group)

        # Clear all compressed gradients after step to free memory
        # Done at end of step() because both weight and bias params need access
        # to the same _compressed_grad during processing
        self.grad_hook.clear_all_compressed_grads()

        return loss

    def _step_compressed(self, param, compressed_grad, group, layer_name):
        """
        Update parameter using compressed gradient optimization.

        This maintains optimizer states in compressed space and uses transpose
        to project updates back to parameter space.

        Args:
            param: Parameter tensor to update
            compressed_grad: Compressed gradient [1, k_l]
            group: Optimizer parameter group
            layer_name: Name of the layer
        """
        state = self.state[param]

        # Get compressor for this layer
        layer_idx = self.grad_hook.layer_name_to_idx[layer_name]
        compressor = self.grad_hook.update_compressors[layer_idx] if layer_idx < len(self.grad_hook.update_compressors) else None

        if compressor is None:
            # Fallback to standard update if compression not available
            self._step_standard(param, group)
            return

        # Initialize state if needed
        is_first_step = len(state) == 0
        if is_first_step:
            state['step'] = 0

            # Initialize compressed states (squeeze to remove batch dim)
            state['exp_avg'] = torch.zeros_like(compressed_grad.squeeze(0))  # First moment in compressed space [k_l]
            state['exp_avg_sq'] = torch.zeros_like(compressed_grad.squeeze(0))  # Second moment in compressed space [k_l]

        # Get hyperparameters
        beta1, beta2 = group['betas']
        state['step'] += 1

        # Update biased first and second moment estimates in compressed space
        # compressed_grad is already in compressed space [1, k_l], squeeze to [k_l]
        compressed_grad_vec = compressed_grad.squeeze(0)

        # Dimension assertion: ensure gradient matches state dimensions
        assert compressed_grad_vec.shape == state['exp_avg'].shape, \
            f"Compressed gradient dimension mismatch for {layer_name}: " \
            f"grad {compressed_grad_vec.shape} vs state {state['exp_avg'].shape}"

        state['exp_avg'].mul_(beta1).add_(compressed_grad_vec, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(
            compressed_grad_vec, compressed_grad_vec, value=1 - beta2
        )

        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Compute step in compressed space
        step_size = group['lr'] / bias_correction1
        denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        compressed_update = state['exp_avg'] / denom

        # Decompress update back to parameter space using transpose composition
        # Apply transpose in REVERSE order of compression:
        # Forward compression: g → (sparsify) → g' → (project) → ĝ
        # Backward transpose:  ĝ → (project^T) → g' → (sparsify^T) → ḡ

        # compressed_update is [k_l], need to add batch dim to make [1, k_l]
        if compressed_update.dim() == 1:
            compressed_update_batch = compressed_update.unsqueeze(0)
        else:
            compressed_update_batch = compressed_update

        # Apply decompression
        full_update = compressor.transpose(compressed_update_batch)  # ĝ → ḡ [1, p_l]

        # Extract the correct portion for this parameter.
        # Decompressed gradient has shape [1, out_features * (in_features + 1)]
        # when bias exists, because input was augmented with ones column in backward().
        # The gradient matrix [out_features, in_features + 1] is flattened in row-major order,
        # so weight and bias elements are interleaved. We must reshape to extract correctly.
        layer_name_for_param, param_type = self._get_layer_info(param)

        # Get the module to check if it has bias
        module = self.grad_hook.model
        for attr in layer_name.split('.'):
            module = getattr(module, attr)

        has_bias = hasattr(module, 'bias') and module.bias is not None

        if has_bias and param_type == 'bias':
            # Extract bias portion from augmented gradient matrix
            # full_update is [1, out_features * (in_features + 1)] flattened in row-major order
            # Reshape to [out_features, in_features + 1], bias is the last column
            out_features = param.numel()
            in_features_plus_1 = full_update.shape[1] // out_features
            grad_matrix = full_update.reshape(out_features, in_features_plus_1)
            param_update = grad_matrix[:, -1]  # Last column is bias [out_features]
        elif has_bias and param_type == 'weight':
            # Extract weight portion from augmented gradient matrix
            # Reshape to [out_features, in_features + 1], weight is all but last column
            out_features, in_features = param.shape
            in_features_plus_1 = in_features + 1
            grad_matrix = full_update.reshape(out_features, in_features_plus_1)
            param_update = grad_matrix[:, :-1]  # All but last column [out_features, in_features]
        else:
            # No bias, so full_update is just the weight
            param_update = full_update.reshape(param.shape)

        # Apply weight decay (in parameter space)
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.add_(param_update, alpha=-step_size)

    def _step_standard(self, param, group):
        """
        Standard AdamW update for parameters without compression.

        Args:
            param: Parameter tensor to update
            group: Optimizer parameter group
        """
        grad = param.grad
        if grad is None:
            return

        state = self.state[param]

        # Initialize state if needed
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)

        # Get hyperparameters
        beta1, beta2 = group['betas']
        state['step'] += 1

        # Update biased first and second moment estimates
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        step_size = group['lr'] / bias_correction1

        # Compute update
        denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        # Apply weight decay
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.addcdiv_(state['exp_avg'], denom, value=-step_size)


class MeSOSGD(Optimizer):
    """
    SGD optimizer with momentum that maintains states in compressed gradient space.

    Simpler variant of MeSOAdamW for comparison and debugging.
    """

    def __init__(
        self,
        params,
        grad_hook: GradientHook,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        compressed_layer_names: Optional[List[str]] = None
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.grad_hook = grad_hook
        self.compressed_layer_names = compressed_layer_names or grad_hook.layer_names

        # Create mapping from parameter to layer name
        self._param_to_layer_name = {}
        self._setup_param_mapping()

        logger.info(f"Initialized MeSOSGD optimizer")
        logger.info(f"  Compressed layers: {len(self.compressed_layer_names)}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Momentum: {momentum}")
        logger.info(f"  Weight decay: {weight_decay}")

    def _setup_param_mapping(self):
        """Create mapping from parameters to layer names."""
        model = self.grad_hook.model

        for layer_name in self.compressed_layer_names:
            module = model
            for attr in layer_name.split('.'):
                module = getattr(module, attr)

            if hasattr(module, 'weight') and module.weight is not None:
                self._param_to_layer_name[id(module.weight)] = (layer_name, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                self._param_to_layer_name[id(module.bias)] = (layer_name, 'bias')

    def _get_layer_info(self, param):
        """Get layer name and parameter type for a given parameter."""
        param_id = id(param)
        return self._param_to_layer_name.get(param_id, (None, None))


    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        This implementation processes gradients layer-by-layer to minimize memory usage.
        Instead of decompressing all gradients at once, we decompress each layer's
        gradient/momentum only when needed.

        Args:
            closure: Optional closure to reevaluate the model and return the loss

        Returns:
            Optional loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update each parameter group
        for group in self.param_groups:
            for p in group['params']:
                # Get layer information
                layer_name, param_type = self._get_layer_info(p)

                # Check if this parameter should use compressed optimization
                if layer_name is not None:
                    layer_idx = self.grad_hook.layer_name_to_idx.get(layer_name)
                    if layer_idx is not None:
                        # Get compressed gradient from the weight parameter's _compressed_grad attribute
                        # Note: Compressed grad is always stored on weight (even for bias params)
                        # because it contains both weight and bias gradients combined
                        if param_type == 'weight':
                            compressed_grad = getattr(p, '_compressed_grad', None)
                        else:
                            # For bias, get compressed grad from the weight of the same layer
                            module = self.grad_hook.layer_name_to_module.get(layer_name)
                            compressed_grad = getattr(module.weight, '_compressed_grad', None) if module else None

                        if compressed_grad is not None:
                            # Use compressed gradient pathway (layer-by-layer decompression)
                            self._step_compressed(p, compressed_grad, group, layer_name)
                            continue

                # Use standard gradient pathway (for non-compressed layers)
                if p.grad is None:
                    continue
                self._step_standard(p, group)

        # Clear all compressed gradients after step to free memory
        # Done at end of step() because both weight and bias params need access
        # to the same _compressed_grad during processing
        self.grad_hook.clear_all_compressed_grads()

        return loss

    def _step_compressed(self, param, compressed_grad, group, layer_name):
        """
        Update parameter using compressed gradient optimization.

        This maintains momentum in compressed space and uses transpose
        to project updates back to parameter space layer-by-layer.

        Args:
            param: Parameter tensor to update
            compressed_grad: Compressed gradient [1, k_l]
            group: Optimizer parameter group
            layer_name: Name of the layer
        """
        state = self.state[param]

        # Get compressor for this layer
        layer_idx = self.grad_hook.layer_name_to_idx[layer_name]
        compressor = self.grad_hook.update_compressors[layer_idx] if layer_idx < len(self.grad_hook.update_compressors) else None

        if compressor is None:
            # Fallback to standard update if compression not available
            self._step_standard(param, group)
            return

        # Initialize state if needed
        if len(state) == 0:
            # Initialize momentum buffer in compressed space (squeeze to remove batch dim)
            state['momentum_buffer'] = torch.zeros_like(compressed_grad.squeeze(0))  # [k_l]

        # Get momentum coefficient
        momentum = group['momentum']

        # Update momentum in compressed space
        # compressed_grad is [1, k_l], squeeze to [k_l]
        compressed_grad_vec = compressed_grad.squeeze(0)

        # Dimension assertion
        assert compressed_grad_vec.shape == state['momentum_buffer'].shape, \
            f"Compressed gradient dimension mismatch for {layer_name}: " \
            f"grad {compressed_grad_vec.shape} vs state {state['momentum_buffer'].shape}"

        if momentum != 0:
            # v_t = momentum * v_{t-1} + g_t (in compressed space)
            state['momentum_buffer'].mul_(momentum).add_(compressed_grad_vec)
            compressed_update = state['momentum_buffer']
        else:
            # No momentum: use gradient directly
            compressed_update = compressed_grad_vec

        # Decompress update back to parameter space using transpose
        # compressed_update is [k_l], need to add batch dim to make [1, k_l]
        if compressed_update.dim() == 1:
            compressed_update_batch = compressed_update.unsqueeze(0)
        else:
            compressed_update_batch = compressed_update

        # Apply decompression
        full_update = compressor.transpose(compressed_update_batch)  # [1, p_l]

        # Extract the correct portion for this parameter.
        # Decompressed gradient has shape [1, out_features * (in_features + 1)]
        # when bias exists, because input was augmented with ones column in backward().
        # The gradient matrix [out_features, in_features + 1] is flattened in row-major order,
        # so weight and bias elements are interleaved. We must reshape to extract correctly.
        layer_name_for_param, param_type = self._get_layer_info(param)

        # Get the module to check if it has bias
        module = self.grad_hook.model
        for attr in layer_name.split('.'):
            module = getattr(module, attr)

        has_bias = hasattr(module, 'bias') and module.bias is not None

        if has_bias and param_type == 'bias':
            # Extract bias portion from augmented gradient matrix
            # full_update is [1, out_features * (in_features + 1)] flattened in row-major order
            # Reshape to [out_features, in_features + 1], bias is the last column
            out_features = param.numel()
            in_features_plus_1 = full_update.shape[1] // out_features
            grad_matrix = full_update.reshape(out_features, in_features_plus_1)
            param_update = grad_matrix[:, -1]  # Last column is bias [out_features]
        elif has_bias and param_type == 'weight':
            # Extract weight portion from augmented gradient matrix
            # Reshape to [out_features, in_features + 1], weight is all but last column
            out_features, in_features = param.shape
            in_features_plus_1 = in_features + 1
            grad_matrix = full_update.reshape(out_features, in_features_plus_1)
            param_update = grad_matrix[:, :-1]  # All but last column [out_features, in_features]
        else:
            # No bias, so full_update is just the weight
            param_update = full_update.reshape(param.shape)

        # Apply weight decay (in parameter space)
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.add_(param_update, alpha=-group['lr'])

    def _step_standard(self, param, group):
        """
        Standard SGD update for parameters without compression.

        Args:
            param: Parameter tensor to update
            group: Optimizer parameter group
        """
        grad = param.grad
        if grad is None:
            return

        state = self.state[param]

        # Get momentum coefficient
        momentum = group['momentum']

        if momentum != 0:
            # Initialize momentum buffer if needed
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(param)

            # Update momentum
            state['momentum_buffer'].mul_(momentum).add_(grad)
            update = state['momentum_buffer']
        else:
            # No momentum: use gradient directly
            update = grad

        # Apply weight decay
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.add_(update, alpha=-group['lr'])
