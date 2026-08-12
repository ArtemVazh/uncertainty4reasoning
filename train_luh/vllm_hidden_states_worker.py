"""vLLM worker extension for capturing hidden states from hybrid models."""

from __future__ import annotations

import logging

from speculators.data_generation.custom_worker import HiddenStatesWorkerExtension as _BaseExtension


log = logging.getLogger(__name__)


class HiddenStatesWorkerExtension(_BaseExtension):
    """Capture layer outputs with hooks instead of replacing the model forward.

    The upstream speculators extension replaces the decoder's ``forward`` method.
    That replacement is bypassed by vLLM 0.19 for Qwen3.5/3.6 hybrid models after
    model-runner initialization. Forward hooks remain attached to the actual layer
    modules and work for both full-attention and linear-attention decoder blocks.
    """

    def _resolve_decoder(self):
        model = self.model_runner.model
        if hasattr(model, "get_language_model"):
            language_model = model.get_language_model()
            decoder = getattr(language_model, "model", language_model)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            decoder = model.model
        else:
            decoder = model

        if not hasattr(decoder, "layers"):
            raise AttributeError(
                "Could not find decoder layers for hidden-state capture: "
                f"model={type(model).__name__}, decoder={type(decoder).__name__}"
            )
        return decoder

    def _setup_hidden_states_capture(self, layer_ids: list[int]):
        self._layer_ids = frozenset(layer_ids)
        self._captured_states = None
        self._request_metadata = []
        self._current_request_metadata = None
        self._pending_layer_states = {}

        for handle in getattr(self, "_hidden_state_hook_handles", []):
            handle.remove()
        self._hidden_state_hook_handles = []

        decoder = self._resolve_decoder()
        start_layer = int(getattr(decoder, "start_layer", 0))
        end_layer = int(getattr(decoder, "end_layer", len(decoder.layers)))
        local_targets = [
            layer_id for layer_id in sorted(self._layer_ids)
            if start_layer <= layer_id < end_layer
        ]

        for layer_id in local_targets:
            layer = decoder.layers[layer_id - start_layer]

            def capture(_module, _args, output, *, captured_layer_id=layer_id):
                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError(
                        "Expected decoder layer output (hidden_states, residual), "
                        f"got {type(output).__name__}"
                    )
                hidden_states, residual = output[:2]
                state = hidden_states if residual is None else hidden_states + residual
                self._pending_layer_states[captured_layer_id] = state.clone()

                if all(target in self._pending_layer_states for target in local_targets):
                    ordered = [self._pending_layer_states.pop(target) for target in local_targets]
                    self._store_captured_states(ordered)

            self._hidden_state_hook_handles.append(layer.register_forward_hook(capture))

        log.info(
            "Hidden-state hook capture configured for layers %s on %s (local range %d:%d)",
            local_targets,
            type(decoder).__name__,
            start_layer,
            end_layer,
        )

    def _reset_capture(self):
        self._pending_layer_states.clear()
        super()._reset_capture()
