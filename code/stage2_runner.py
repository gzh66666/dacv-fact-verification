import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.debater import setup_debater, run_argument  # noqa: E402
from agents.dim_verifier import setup_verifier, run_verifier  # noqa: E402
from search.bm25_retrieval import BM25Retriever  # noqa: E402


def _load_decomposition(decomp_dir: Path, dataset: str, task: str, event_id: str) -> dict:
    p = decomp_dir / dataset / task / f"{event_id}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_evidence_pool(pool_dir: Path, dataset: str, task: str, event_id: str) -> List[str]:
    p = pool_dir / dataset / task / f"{event_id}.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("evidence_pool", []) or []


def _list_event_ids(decomp_dir: Path, dataset: str, task: str) -> List[str]:
    target = decomp_dir / dataset / task
    if not target.exists():
        raise FileNotFoundError(f"No Stage 1 outputs at {target}")
    ids = []
    for fn in sorted(os.listdir(target)):
        if fn.endswith(".json"):
            ids.append(fn[:-5])
    return ids


def _save_stage2(record: dict, output_file: Path) -> None:
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _write_status_file(path: Optional[Path], status: dict) -> None:
    if path is None:
        return
    payload = {"updated_epoch": time.time(), **status}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_subclaim_iter(decomp: dict) -> List[dict]:
    out = []
    for e in decomp.get("E_list", []):
        if e.get("verifiable") and e.get("focus_question"):
            out.append({
                "key": e.get("key", "E"),
                "dimension": "E",
                "content": e.get("content", ""),
                "focus_question": e.get("focus_question", ""),
            })
    for letter in ("A", "Q"):
        d = decomp.get(letter)
        if d and d.get("verifiable") and d.get("focus_question"):
            out.append({
                "key": letter,
                "dimension": letter,
                "content": d.get("content", ""),
                "focus_question": d.get("focus_question", ""),
            })
    return out


def _debate_one_subclaim(
    *,
    pro_agent, pro_proxy,
    con_agent, con_proxy,
    verifier_agent, verifier_proxy,
    dataset: str,
    rounds: int,
    use_search: bool,
    serper_client,
    original_claim: str,
    sub_claim: dict,
    retriever: BM25Retriever,
    topk_evidence: int,
    n_search_results: int,
    event_id: str = "",
    logger=None,
) -> dict:
    fq = sub_claim["focus_question"]

    initial_evidence = retriever.top_k_sentences(fq, k=topk_evidence)
    evidence_pro = list(initial_evidence)
    evidence_con = list(initial_evidence)

    pro_args: List[dict] = []
    con_args: List[dict] = []
    external_searches: List[dict] = []

    pro_r1 = run_argument(
        pro_agent, pro_proxy,
        side="Pro", dataset=dataset,
        original_claim=original_claim,
        sub_claim_text=sub_claim["content"],
        sub_claim_dimension=sub_claim["key"],
        focus_question=fq,
        evidence_sentences=evidence_pro,
        opponent_round1=None,
        logger=logger,
        call_context={"event_id": event_id, "stage": "debate", "agent": "Pro",
                      "dimension": sub_claim["key"], "round": 1},
    )
    pro_args.append(pro_r1)

    con_r1 = run_argument(
        con_agent, con_proxy,
        side="Con", dataset=dataset,
        original_claim=original_claim,
        sub_claim_text=sub_claim["content"],
        sub_claim_dimension=sub_claim["key"],
        focus_question=fq,
        evidence_sentences=evidence_con,
        opponent_round1=None,
        logger=logger,
        call_context={"event_id": event_id, "stage": "debate", "agent": "Con",
                      "dimension": sub_claim["key"], "round": 1},
    )
    con_args.append(con_r1)

    if use_search and serper_client is not None:
        for arg, ev_list, side in (
            (pro_r1, evidence_pro, "Pro"),
            (con_r1, evidence_con, "Con"),
        ):
            if arg.get("needs_external_search") and arg.get("search_query"):
                gl = "us" if dataset == "RAWFC" else "cn"
                hl = "en" if dataset == "RAWFC" else "zh-cn"
                snippets = serper_client.search(
                    arg["search_query"],
                    num_results=n_search_results,
                    gl=gl, hl=hl,
                    log_context={"event_id": event_id, "stage": "external_search",
                                 "dimension": sub_claim["key"], "side": side, "round": 1},
                )
                external_searches.append({
                    "side": side,
                    "round": 1,
                    "query": arg["search_query"],
                    "n_snippets": len(snippets),
                    "snippets": snippets,
                })
                ev_list.extend([f"[external] {s}" for s in snippets])

    if rounds >= 2:
        pro_r2 = run_argument(
            pro_agent, pro_proxy,
            side="Pro", dataset=dataset,
            original_claim=original_claim,
            sub_claim_text=sub_claim["content"],
            sub_claim_dimension=sub_claim["key"],
            focus_question=fq,
            evidence_sentences=evidence_pro,
            opponent_round1=con_r1,
            logger=logger,
            call_context={"event_id": event_id, "stage": "debate", "agent": "Pro",
                          "dimension": sub_claim["key"], "round": 2},
        )
        pro_args.append(pro_r2)

        con_r2 = run_argument(
            con_agent, con_proxy,
            side="Con", dataset=dataset,
            original_claim=original_claim,
            sub_claim_text=sub_claim["content"],
            sub_claim_dimension=sub_claim["key"],
            focus_question=fq,
            evidence_sentences=evidence_con,
            opponent_round1=pro_r1,
            logger=logger,
            call_context={"event_id": event_id, "stage": "debate", "agent": "Con",
                          "dimension": sub_claim["key"], "round": 2},
        )
        con_args.append(con_r2)

    verifier_evidence = list(initial_evidence)
    seen = set(verifier_evidence)
    for s in evidence_pro + evidence_con:
        if s not in seen:
            verifier_evidence.append(s)
            seen.add(s)

    verdict = run_verifier(
        verifier_agent, verifier_proxy,
        dataset=dataset,
        sub_claim_text=sub_claim["content"],
        sub_claim_dimension=sub_claim["key"],
        focus_question=fq,
        evidence_sentences=verifier_evidence,
        pro_arguments=pro_args,
        con_arguments=con_args,
        logger=logger,
        call_context={"event_id": event_id, "stage": "verifier", "agent": "Dim_Verifier",
                      "dimension": sub_claim["key"], "round": rounds},
    )

    return {
        "key": sub_claim["key"],
        "dimension": sub_claim["dimension"],
        "content": sub_claim["content"],
        "focus_question": fq,
        "initial_evidence_topk": initial_evidence,
        "external_searches": external_searches,
        "pro_arguments": pro_args,
        "con_arguments": con_args,
        "verdict": verdict,
    }


def run_stage2_for_item(
    *,
    pro_agent, pro_proxy,
    con_agent, con_proxy,
    verifier_agent, verifier_proxy,
    dataset: str,
    rounds: int,
    use_search: bool,
    serper_client,
    decomp_record: dict,
    evidence_pool: List[str],
    topk_evidence: int,
    n_search_results: int,
    early_stop: bool,
    logger=None,
) -> dict:
    claim = decomp_record["claim"]
    decomp = decomp_record["decomposition"]
    subclaims = _build_subclaim_iter(decomp)

    retriever = BM25Retriever(evidence_pool)

    debate_results: List[dict] = []
    early_stopped = False
    early_stop_reason = None

    for sub in subclaims:
        if early_stopped:
            break
        result = _debate_one_subclaim(
            pro_agent=pro_agent, pro_proxy=pro_proxy,
            con_agent=con_agent, con_proxy=con_proxy,
            verifier_agent=verifier_agent, verifier_proxy=verifier_proxy,
            dataset=dataset,
            rounds=rounds,
            use_search=use_search,
            serper_client=serper_client,
            original_claim=claim,
            sub_claim=sub,
            retriever=retriever,
            topk_evidence=topk_evidence,
            n_search_results=n_search_results,
            event_id=decomp_record.get("event_id", ""),
            logger=logger,
        )
        debate_results.append(result)

        if early_stop and sub["dimension"] == "E" and sub["key"] in ("E", "E1"):
            v = result["verdict"]
            if v.get("verdict") == "F" and v.get("confidence") == "high":
                a_ver = bool((decomp.get("A") or {}).get("verifiable"))
                q_ver = bool((decomp.get("Q") or {}).get("verifiable"))
                if not a_ver and not q_ver:
                    early_stopped = True
                    early_stop_reason = (
                        f"E={sub['key']} verdict=F (high confidence) and A/Q are both unverifiable"
                    )

    verdicts_summary = {
        "E_verdicts": [
            {"key": r["key"], "verdict": r["verdict"]["verdict"], "confidence": r["verdict"]["confidence"]}
            for r in debate_results if r["dimension"] == "E"
        ],
        "A_verdict": next(
            ({"verdict": r["verdict"]["verdict"], "confidence": r["verdict"]["confidence"]}
             for r in debate_results if r["dimension"] == "A"),
            None,
        ),
        "Q_verdict": next(
            ({"verdict": r["verdict"]["verdict"], "confidence": r["verdict"]["confidence"]}
             for r in debate_results if r["dimension"] == "Q"),
            None,
        ),
    }

    return {
        "event_id": decomp_record.get("event_id"),
        "claim": claim,
        "label": decomp_record.get("label"),
        "stage2_config": {
            "rounds": rounds,
            "use_search": use_search,
            "topk_evidence": topk_evidence,
            "n_search_results": n_search_results,
            "early_stop": early_stop,
        },
        "evidence_pool_size": len(evidence_pool),
        "n_subclaims_debated": len(debate_results),
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "debates": debate_results,
        "verdicts_summary": verdicts_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="DACV Stage 2: Adversarial Debate Orchestrator")
    parser.add_argument("--model_name", type=str, default="deepseek-flash")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["RAWFC", "LIAR-RAW"])
    parser.add_argument("--task", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--decomp_dir", type=str, default=str(ROOT / "results" / "decomposition"))
    parser.add_argument("--pool_dir", type=str, default=str(ROOT / "results" / "evidence_pool"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "stage2"))
    parser.add_argument("--rounds", type=int, default=1, choices=[1, 2],
                        help="Number of debate rounds (default: 1)")
    parser.add_argument("--use_search", action="store_true",
                        help="Enable on-demand Serper search (requires SERPER_API_KEY)")
    parser.add_argument("--topk_evidence", type=int, default=5,
                        help="BM25 top-k retrieval from R' (default: 5)")
    parser.add_argument("--n_search_results", type=int, default=5,
                        help="Max snippets per Serper call (default: 5)")
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early-stop after E=F(high) when A and Q are unverifiable. "
                             "Default OFF: in 19-claim eval, early-stop misfired on borderline half cases. "
                             "Enable only for cost-saving experiments where false-class precision matters more "
                             "than half-class recall.")
    parser.add_argument("--limit", type=int, default=-1, help="-1 = all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Resume: skip event_ids already in output_dir")
    parser.add_argument("--event_ids", type=str, default="",
                        help="Comma-separated subset of event_ids to run (overrides --limit ordering)")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Run audit directory for configuration and API logs")
    parser.add_argument("--run_id", type=str, default="",
                        help="Stable identifier written to audit records")
    args = parser.parse_args()

    decomp_dir = Path(args.decomp_dir)
    pool_dir = Path(args.pool_dir)
    output_dir = Path(args.output_dir) / args.dataset / args.task
    os.makedirs(output_dir, exist_ok=True)

    logger = None
    transcript_path = None
    status_path = None
    if args.run_dir:
        from run_logger import RunLogger
        logger = RunLogger(args.run_dir, run_id=args.run_id or None)
        transcript_path = Path(args.run_dir) / "06_模型完整输出.log"
        transcript_path.touch(exist_ok=True)
        status_path = Path(args.run_dir) / "06_运行状态.json"
        logger.write_preflight({
            "purpose": "Stage 2 debate and dimension verification",
            "dataset": args.dataset, "split": args.task,
            "model_id_requested": os.environ.get("DACV_MODEL_ID", "deepseek-flash"),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "temperature": os.environ.get("DACV_TEMPERATURE", "0.1"),
            "thinking": "disabled", "api_seed": os.environ.get("DACV_API_SEED"),
            "cache_seed": 42, "max_tokens": os.environ.get("DACV_MAX_TOKENS"),
            "rounds": args.rounds,
            "use_search": args.use_search, "topk_evidence": args.topk_evidence,
            "n_search_results": args.n_search_results, "early_stop": args.early_stop,
            "stage": "stage2",
        })

    if args.event_ids.strip():
        event_ids = [s.strip() for s in args.event_ids.split(",") if s.strip()]
    else:
        event_ids = _list_event_ids(decomp_dir, args.dataset, args.task)
    if args.limit > 0:
        event_ids = event_ids[: args.limit]

    started_epoch = time.time()
    _write_status_file(status_path, {
        "run_id": args.run_id or (Path(args.run_dir).name if args.run_dir else None),
        "stage": "stage2",
        "status": "running",
        "total": len(event_ids),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_event_id": None,
        "started_epoch": started_epoch,
        "output_dir": str(output_dir),
    })

    print(f"[setup] Building Pro/Con/Verifier agents on model={args.model_name}, dataset={args.dataset}")
    pro_agent, pro_proxy = setup_debater(args.model_name, args.dataset, side="Pro")
    con_agent, con_proxy = setup_debater(args.model_name, args.dataset, side="Con")
    verifier_agent, verifier_proxy = setup_verifier(args.model_name, args.dataset)

    serper_client = None
    if args.use_search:
        from search.serper_client import SerperClient
        serper_client = SerperClient(logger=logger)

    n_ok, n_fail, n_skip = 0, 0, 0
    t_start = time.time()
    for idx, eid in enumerate(event_ids):
        out_file = output_dir / f"{eid}.json"
        if args.skip_existing and out_file.exists():
            n_skip += 1
            print(f"[{idx + 1}/{len(event_ids)}] {eid} exists, skip")
            _write_status_file(status_path, {
                "run_id": args.run_id or (Path(args.run_dir).name if args.run_dir else None),
                "stage": "stage2", "status": "running", "total": len(event_ids),
                "completed": n_ok, "failed": n_fail, "skipped": n_skip,
                "current_event_id": eid, "started_epoch": started_epoch,
                "output_dir": str(output_dir),
            })
            continue

        try:
            decomp_record = _load_decomposition(decomp_dir, args.dataset, args.task, eid)
            pool = _load_evidence_pool(pool_dir, args.dataset, args.task, eid)
            if not pool:
                print(f"[{idx + 1}/{len(event_ids)}] {eid} WARN empty evidence pool")

            if transcript_path is not None:
                with transcript_path.open("a", encoding="utf-8") as transcript:
                    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
                        record = run_stage2_for_item(
                            pro_agent=pro_agent, pro_proxy=pro_proxy,
                            con_agent=con_agent, con_proxy=con_proxy,
                            verifier_agent=verifier_agent, verifier_proxy=verifier_proxy,
                            dataset=args.dataset,
                            rounds=args.rounds,
                            use_search=args.use_search,
                            serper_client=serper_client,
                            decomp_record=decomp_record,
                            evidence_pool=pool,
                            topk_evidence=args.topk_evidence,
                            n_search_results=args.n_search_results,
                            early_stop=args.early_stop,
                            logger=logger,
                        )
            else:
                record = run_stage2_for_item(
                    pro_agent=pro_agent, pro_proxy=pro_proxy,
                    con_agent=con_agent, con_proxy=con_proxy,
                    verifier_agent=verifier_agent, verifier_proxy=verifier_proxy,
                    dataset=args.dataset,
                    rounds=args.rounds,
                    use_search=args.use_search,
                    serper_client=serper_client,
                    decomp_record=decomp_record,
                    evidence_pool=pool,
                    topk_evidence=args.topk_evidence,
                    n_search_results=args.n_search_results,
                    early_stop=args.early_stop,
                    logger=logger,
                )
            _save_stage2(record, out_file)
            n_ok += 1

            if logger is not None:
                logger.log_sample({
                    "event_id": eid, "stage": "stage2",
                    "output_path": str(out_file),
                    "n_subclaims_debated": record.get("n_subclaims_debated"),
                    "early_stopped": record.get("early_stopped"),
                    "verdicts_summary": record.get("verdicts_summary"),
                    "n_external_searches": sum(
                        len(d.get("external_searches", [])) for d in record.get("debates", [])
                    ),
                })

            vs = record["verdicts_summary"]
            ev = ", ".join(f"{x['key']}={x['verdict']}({x['confidence'][0]})" for x in vs["E_verdicts"])
            av = f" A={vs['A_verdict']['verdict']}({vs['A_verdict']['confidence'][0]})" if vs["A_verdict"] else ""
            qv = f" Q={vs['Q_verdict']['verdict']}({vs['Q_verdict']['confidence'][0]})" if vs["Q_verdict"] else ""
            es = " [early-stop]" if record["early_stopped"] else ""
            print(f"[{idx + 1}/{len(event_ids)}] {eid} label={record['label']} -> {ev}{av}{qv}{es}")
            _write_status_file(status_path, {
                "run_id": args.run_id or (Path(args.run_dir).name if args.run_dir else None),
                "stage": "stage2", "status": "running", "total": len(event_ids),
                "completed": n_ok, "failed": n_fail, "skipped": n_skip,
                "current_event_id": eid, "started_epoch": started_epoch,
                "output_dir": str(output_dir), "last_label": record.get("label"),
            })

        except Exception as e:
            n_fail += 1
            print(f"[{idx + 1}/{len(event_ids)}] {eid} ERROR: {type(e).__name__}: {e}")
            if logger is not None:
                logger.log_text(f"ERROR event_id={eid} type={type(e).__name__} message={e}")
            _write_status_file(status_path, {
                "run_id": args.run_id or (Path(args.run_dir).name if args.run_dir else None),
                "stage": "stage2", "status": "running", "total": len(event_ids),
                "completed": n_ok, "failed": n_fail, "skipped": n_skip,
                "current_event_id": eid, "started_epoch": started_epoch,
                "output_dir": str(output_dir), "last_error": f"{type(e).__name__}: {e}",
            })

    elapsed = time.time() - t_start
    print("\n=== Stage 2 summary ===")
    print(f"total      : {len(event_ids)}")
    print(f"ok         : {n_ok}")
    print(f"fail       : {n_fail}")
    print(f"elapsed    : {elapsed:.1f}s ({elapsed / max(1, n_ok):.1f}s/claim)")
    if serper_client:
        print(f"serper calls: {serper_client.n_calls}")
    print(f"output     : {output_dir}")
    _write_status_file(status_path, {
        "run_id": args.run_id or (Path(args.run_dir).name if args.run_dir else None),
        "stage": "stage2", "status": "completed", "total": len(event_ids),
        "completed": n_ok, "failed": n_fail, "skipped": n_skip,
        "current_event_id": None, "started_epoch": started_epoch,
        "elapsed_sec": elapsed, "output_dir": str(output_dir),
    })


if __name__ == "__main__":
    main()
