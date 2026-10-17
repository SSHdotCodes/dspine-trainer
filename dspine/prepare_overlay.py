"""Generate a small, fail-closed patch against the *installed* SGLang source.

Does not copy the historical Qwen4 implementation: that older overlay has an
obsolete capture contract and importing it wholesale would replace current
qwen4_exp fixes. Its post-layer HC-mean capture placement is reused here.
"""

import argparse
import ast
from pathlib import Path


def patch(source):
    original = source
    source = (
        "from dspine.capture import capture_layer as _dspine_capture_layer, capture_batch as _dspine_capture_batch, capture_layer_ids as _dspine_layer_ids\n"
        + source
        if "from __future__" not in source
        else source
    )
    # Preserve future-import ordering, if present.
    if source == original:
        point = source.index("\n", source.index("from __future__ import")) + 1
        source = (
            source[:point]
            + "from dspine.capture import capture_layer as _dspine_capture_layer, capture_batch as _dspine_capture_batch, capture_layer_ids as _dspine_layer_ids\n"
            + source[point:]
        )
    start = source.index("class Qwen4ExpModel(")
    end = source.index("\nclass Qwen4ExpVLModel", start)
    section = source[start:end]
    needle = "        aux_hidden_states = AuxHiddenStateList()"
    if section.count(needle) != 1:
        raise ValueError("unsupported capture canonicalization setup")
    section = section.replace(needle, """        for capture_id in _dspine_layer_ids():
            self.layers[capture_id]._is_layer_to_capture = True
""" + needle)
    # qwenfast observes the flag above to materialize the combined HC residual.
    # Avoid collecting the inherited auxiliary tensor list as well.
    section = section.replace(
        'if getattr(layer, "_is_layer_to_capture", False)',
        'if False and getattr(layer, "_is_layer_to_capture", False)',
    )
    needle = "        _commit_ple_batch(ple_batch, forward_batch)"
    if section.count(needle) != 1:
        raise ValueError("unsupported Qwen4ExpModel layout")
    section = section.replace(
        needle,
        "            _dspine_capture_layer(self, i, hidden_states, forward_batch)\n\n"
        + needle,
    )
    needle = "        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)"
    if section.count(needle) != 1:
        raise ValueError("unsupported final hyper-connection mixer")
    section = section.replace(
        needle, needle + "\n        self._dspine_final_hidden = hidden_states"
    )
    source = source[:start] + section + source[end:]
    start = source.index("class Qwen4ExpForConditionalGeneration(")
    needle = "        hc_hidden_states = self.model.last_hc_hidden_states"
    if source[start:].count(needle) != 1:
        raise ValueError("unsupported causal LM wrapper")
    injection = """        if getattr(self.model, "_dspine_final_hidden", None) is not None:
            _dspine_capture_batch(self.model, forward_batch.input_ids,
                                  self.model._dspine_final_hidden, forward_batch,
                                  lambda h: self.logits_processor._get_logits(
                                      h, self.lm_head, None, use_logits_buffer=False),
                                  head_module=self.lm_head)
"""
    source = source[:start] + source[start:].replace(needle, injection + needle)
    ast.parse(source)
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    Path(args.output).write_text(patch(Path(args.source).read_text()))


if __name__ == "__main__":
    main()
