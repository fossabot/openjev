"""Give DiffusionGemma's image tokens the bidirectional attention it was trained with.

DiffusionGemma's checkpoint config sets ``use_bidirectional_attention: "vision"``, so
tokens inside one image are meant to see each other in both directions. vLLM already
does this for Gemma4: the generic model state calls ``compute_mm_prefix_ranges`` and
hands the ranges to ``build_attn_metadata``, which the Triton backend applies.
DiffusionGemma has its own ``prepare_attn`` and never passes them, so image tokens are
prefilled causally and each patch sees only the patches before it.

This edits the installed vLLM source in the image. It is deliberately narrow: two
anchors, both of which must match exactly once, or the build fails rather than guesses.
Upstream fix pending; drop this once DiffusionGemma passes the ranges itself.
"""
import pathlib
import sys

IMPORT_ANCHOR = "from vllm.v1.worker.gpu.attn_utils import build_attn_metadata\n"
IMPORT_NEW = "from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, compute_mm_prefix_ranges\n"

CALL_ANCHOR = """        return build_attn_metadata(
            attn_groups=attn_groups,"""
CALL_NEW = """        # Image tokens attend bidirectionally within their own span, as the
        # checkpoint's use_bidirectional_attention="vision" asks. Same helper
        # and the same conditions the generic model state uses for Gemma4.
        _mm_ranges = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            _mm_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        return build_attn_metadata(
            mm_req_doc_ranges=_mm_ranges,
            attn_groups=attn_groups,"""


def patch(text: str) -> str:
    for anchor, new in ((IMPORT_ANCHOR, IMPORT_NEW), (CALL_ANCHOR, CALL_NEW)):
        if text.count(anchor) != 1:
            raise SystemExit(f"vision patch: expected exactly one match for {anchor.splitlines()[0]!r}, "
                             f"found {text.count(anchor)}. The pinned vLLM source moved.")
        text = text.replace(anchor, new)
    return text


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    patched = patch(path.read_text())
    compile(patched, str(path), "exec")  # never leave a file that cannot import
    path.write_text(patched)
    print(f"vision patch applied to {path}")


if __name__ == "__main__":
    main()
