"""The kale-tts-1 text front end: the release's own JavaScript front end (sentence split, number / date / money / term
reading, CMUdict lexicon + checked overrides, a small g2p model for unknown words, British rules for British voices)
run in-process with QuickJS (`pip install quickjs`, MIT). The script and word tables ship with the model files
(`frontend/`), so this module only loads them and maps IPA to the model's ids."""

import base64
import json
import threading
from pathlib import Path
from typing import Dict, List

# Kokoro-native symbols (voices "phonemes": "kokoro-native"): the release IPA rewritten before the id lookup
# (= the release's python/kale_tts.py kn_ipa and js/kale-tts.mjs knIPA)
_KN_PAIRS = {"eɪ": "A", "aɪ": "I", "aʊ": "W", "ɔɪ": "Y", "tʃ": "ʧ", "dʒ": "ʤ"}
_KN_VOW = set("aeiouɑɐɒæəɘɚɛɜɝɞɪɨʊʉʌɔøœɤɯʏyAIWOYᵊᵻ")
_KN_END = set(" .,!?;:")


def kn_ipa(chars: List[str], gb: bool) -> List[str]:
    pairs = {**_KN_PAIRS, **({"əʊ": "Q"} if gb else {"oʊ": "O"})}
    out, i = [], 0
    while i < len(chars):
        two = chars[i] + (chars[i + 1] if i + 1 < len(chars) else "")
        if two in pairs:
            out.append(pairs[two])
            i += 2
            continue
        c = chars[i]
        i += 1
        if gb:
            out.append(c)
            continue
        if c == "ː":
            continue
        out += (
            ["ə", "ɹ"]
            if c == "ɚ"
            else ["ɜ", "ɹ"] if c == "ɝ" else ["T"] if c == "ɾ" else [c]
        )
    for k in range(1, len(out) - 1):
        if (
            out[k] == "ə"
            and out[k + 1] in "lnm"
            and (k + 2 == len(out) or out[k + 2] in _KN_END)
            and out[k - 1] not in _KN_VOW
            and out[k - 1] not in "ˈˌ"
            and not out[k - 1].isspace()
        ):
            out[k] = "ᵊ"
    return out


class KaleFrontEnd:
    def __init__(
        self,
        model_dir: Path,
        vocab: Dict[str, int],
        phonemes: str = "kokoro-native",
        ettin: bool = False,
    ):
        try:
            import quickjs
        except ImportError as e:
            raise ImportError(
                "kale-tts-1 reads text with its own front end: pip install quickjs"
            ) from e
        fd = Path(model_dir) / "frontend"
        self.vocab, self.kn = vocab, phonemes == "kokoro-native"
        self.ctx = quickjs.Context()
        self.ctx.set_memory_limit(1 << 30)
        self.ctx.eval((fd / "kale_frontend.js").read_text())
        self.ctx.eval(
            "globalThis.kaleInit = (...a) => kaleFrontEnd.init(...a); globalThis.kaleInitGb = (...a) => kaleFrontEnd.initGb(...a);"
            "globalThis.kaleRun = (t, a) => kaleFrontEnd.run(t, a); globalThis.kaleIpaWords = (t, a) => kaleFrontEnd.ipaWords(t, a);"
            "globalThis.kaleInitEttin = (j) => kaleFrontEnd.initEttin(j); globalThis.kaleEttin = (t, p, s, w) => kaleFrontEnd.ettin(t, p, s, w);"
        )

        def tables(d: Path):
            texts = [
                (d / f).read_text()
                for f in ("lexicon.tsv", "override.public.tsv", "terms.tsv")
            ]
            return texts, base64.b64encode((d / "g2p.bin").read_bytes()).decode()

        words = fd / "words"
        t, g = tables(words)
        self.ctx.get("kaleInit")(*t, g)
        self.accents = {"en-us"}
        if (words / "en-gb" / "g2p.bin").exists():
            t, g = tables(words / "en-gb")
            self.ctx.get("kaleInitGb")(*t, g)
            self.accents.add("en-gb")
        if ettin:
            self.ctx.get("kaleInitEttin")(
                (Path(model_dir) / "ettin-tokenizer.json").read_text()
            )
        self.lock = threading.Lock()

    def _accent(self, accent: str) -> str:
        return accent if accent in self.accents else "en-us"

    def sentences(self, text: str, accent: str = "en-us") -> List[dict]:
        """text -> [{text, ipa}], one per sentence"""
        with self.lock:
            return json.loads(self.ctx.get("kaleRun")(text, self._accent(accent)))

    def ids(self, ipa: str, accent: str = "en-us") -> List[int]:
        chars = [c for c in ipa if c in self.vocab]
        if self.kn:
            chars = kn_ipa(chars, accent == "en-gb")
        return [self.vocab[c] for c in chars if c in self.vocab]

    def ettin_inputs(
        self, text: str, prev: str, sent_ids: List[int], accent: str = "en-us"
    ) -> dict:
        """the Ettin word reader's inputs for one sentence (previous sentence as left context)"""
        with self.lock:
            words = json.loads(self.ctx.get("kaleIpaWords")(text, self._accent(accent)))
        per_word = [self.ids(w, accent) for w in words]
        with self.lock:
            return json.loads(
                self.ctx.get("kaleEttin")(
                    text, prev, json.dumps(sent_ids), json.dumps(per_word)
                )
            )
