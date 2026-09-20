import argparse

from agent.llm_factory import PROVIDERS, build_chat_model, report_env_config
from engine.savegame import load_game, write_save_payload
from engine.simulation_loop import run


def _resolve_report_model(args):
    """Return (llm, label, enabled) for the CLI, honouring .env and flags."""
    enabled = not args.no_report
    conf = report_env_config()
    if args.report_provider is not None:
        provider = args.report_provider.strip().lower()
        conf = (
            {"provider": provider, "model": args.report_model, "api_key": None}
            if provider
            else None
        )
    elif conf and args.report_model:
        conf = dict(conf, model=args.report_model)

    if conf and conf.get("provider") == "none":
        return None, None, False
    if not enabled or not conf or conf.get("provider") not in PROVIDERS:
        return None, None, enabled

    provider = conf["provider"]
    try:
        llm = build_chat_model(
            provider=provider, model=conf.get("model"), api_key=conf.get("api_key")
        )
    except Exception as exc:
        print(
            f"[report] dedicated model unavailable ({exc}); using the winner's model.",
            flush=True,
        )
        return None, None, enabled
    label = f"{provider}/{conf.get('model') or PROVIDERS[provider]['default_model']}"
    return llm, label, enabled


def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM Arena simulation (optionally resume a save)."
    )
    parser.add_argument(
        "--load",
        metavar="ID_OR_PATH",
        help="Resume a saved game by id (e.g. 'autosave') or path to a .json file.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Turn cap. Defaults to the save's value (when resuming) or 20.",
    )
    parser.add_argument(
        "--lang", default="en", choices=["en", "zh"], help="Output language."
    )
    parser.add_argument(
        "--save-dir", default=None, help="Directory holding save files."
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="Write a checkpoint to '<save-dir>/autosave.json' every turn.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable the AI-written end-of-game report (keep the template).",
    )
    parser.add_argument(
        "--report-provider",
        default=None,
        help="Override REPORT_PROVIDER for this run (or 'none' to disable).",
    )
    parser.add_argument(
        "--report-model",
        default=None,
        help="Override REPORT_MODEL for this run.",
    )
    args = parser.parse_args()

    resume = None
    if args.load:
        resume = load_game(args.load, save_dir=args.save_dir)
        print(
            f"Resuming '{resume.get('name') or resume.get('id')}' from turn "
            f"{int(resume.get('next_turn', 0)) + 1} "
            f"(max {resume.get('max_turns')})...",
            flush=True,
        )

    on_checkpoint = None
    if args.autosave:

        def on_checkpoint(payload):
            meta = write_save_payload(
                payload, save_id="autosave", save_dir=args.save_dir
            )
            print(f"  [autosave] turn {int(payload.get('next_turn', 0)) + 1}", flush=True)
            return meta

    report_llm, report_label, report_enabled = _resolve_report_model(args)
    if report_llm is not None:
        print(f"[report] using dedicated model: {report_label}", flush=True)

    run(
        max_turns=args.max_turns,
        verbose=True,
        model_config=None,
        should_stop=None,
        lang=args.lang,
        resume=resume,
        on_checkpoint=on_checkpoint,
        report_llm=report_llm,
        report_label=report_label,
        report_enabled=report_enabled,
    )


if __name__ == "__main__":
    main()
