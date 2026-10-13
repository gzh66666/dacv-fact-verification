import os
from pathlib import Path


class PromptHandler:
    def __init__(self):
        # Prompt search order: DACV_PROMPT_DIR, <code>/prompts, <package>/prompts.
        # The last entry keeps the default layout working from a source checkout
        # and from the released package, where the four prompt files sit at the
        # package root.
        self.dacv_root = Path(__file__).resolve().parent
        override = os.environ.get("DACV_PROMPT_DIR", "")
        self.dacv_prompt_dir = (Path(override) if override
                                else self.dacv_root / "prompts")

        delphi_root = os.environ.get("DELPHIAGENT_ROOT", "")
        self.delphi_prompt_dir = Path(delphi_root) / "prompts" if delphi_root else None

    def handle_prompt(self, name: str) -> str:
        candidates = [self.dacv_prompt_dir / f"{name}.txt",
                      self.dacv_root / "prompts" / f"{name}.txt",
                      self.dacv_root.parent / "prompts" / f"{name}.txt"]
        if self.delphi_prompt_dir is not None:
            candidates.append(self.delphi_prompt_dir / f"{name}.txt")

        seen = set()
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()

        searched = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Prompt '{name}' not found. Searched:\n  - {searched}\n"
            f"If this is a DelphiAgent prompt, set DELPHIAGENT_ROOT env var."
        )
