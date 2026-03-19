import argparse
import json
import os
import time
from typing import List

from app.agents._0_contracts import OCROptions
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.utils.image_io import load_image


def build_parser():
    parser = argparse.ArgumentParser(
        description="특정 메뉴판 이미지에 대해 OCR 언어 자동 감지가 잘 되는지 진단합니다."
    )
    parser.add_argument("--image", default="menu_image/menu_korean.png", help="입력 이미지 경로")
    parser.add_argument(
        "--with-preprocess",
        action="store_true",
        help="전처리 후 언어 감지/OCR 실행",
    )
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_ocr_lang_detect.png",
        help="전처리 이미지 저장 경로(--with-preprocess일 때 사용)",
    )
    parser.add_argument(
        "--json-out",
        default="debug/ocr_lang_detect_result.json",
        help="진단 결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--langs",
        nargs="*",
        default=list(OCRAgent.DEFAULT_AUTO_LANGS),
        help="자동 감지 후보 언어 목록. 예: --langs korean ch en",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument("--top", type=int, default=12, help="언어별 샘플 OCR 라인 최대 출력 수")
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def average_confidence(lines) -> float:
    if not lines:
        return 0.0
    return sum(line.confidence for line in lines) / len(lines)


def sample_texts(lines, limit: int) -> List[str]:
    top_n = max(0, int(limit))
    return [line.text for line in lines[:top_n]]


def summarize_output(lang: str, out, top: int) -> dict:
    texts = [line.text for line in out.lines]
    return {
        "lang": lang,
        "resolved_lang": out.resolved_lang or lang,
        "lang_source": out.lang_detection_source,
        "line_count": len(out.lines),
        "avg_confidence": round(average_confidence(out.lines), 6),
        "script_ratio": round(OCRAgent._expected_script_ratio(texts, lang), 6),
        "sample_lines": sample_texts(out.lines, top),
        "result": out.model_dump() if hasattr(out, "model_dump") else out.dict(),
    }


def print_candidate_table(candidates):
    print("--- AUTO CANDIDATES ---")
    if not candidates:
        print("- none")
        print("")
        return

    for idx, cand in enumerate(candidates, start=1):
        print(
            f"[{idx:02d}] lang={cand.lang} score={cand.score:.3f} "
            f"lines={cand.line_count} conf={cand.avg_confidence:.3f} "
            f"script={cand.script_ratio:.3f}"
        )
    print("")


def print_manual_runs(runs, top: int):
    print("--- FULL OCR BY LANGUAGE ---")
    if not runs:
        print("- none")
        print("")
        return

    for idx, run in enumerate(runs, start=1):
        print(
            f"[{idx:02d}] lang={run['lang']} | lines={run['line_count']} | "
            f"conf={run['avg_confidence']:.3f} | script={run['script_ratio']:.3f}"
        )
        samples = run.get("sample_lines") or []
        for sample_idx, text in enumerate(samples[:top], start=1):
            print(f"      [{sample_idx:02d}] {text}")
    print("")


def main():
    args = build_parser().parse_args()
    opts = OCROptions(min_confidence=args.min_confidence)

    t0 = time.perf_counter()
    raw_data, mime = load_image(args.image)
    t_load = int((time.perf_counter() - t0) * 1000)

    data_for_ocr = raw_data
    t_pre = 0
    if args.with_preprocess:
        t_pre_s = time.perf_counter()
        pre = ImagePreprocessAgent()
        data_for_ocr, _ = pre.run(raw_data, mime, save_path=args.preprocessed_out)
        t_pre = int((time.perf_counter() - t_pre_s) * 1000)

    t_auto_s = time.perf_counter()
    auto_ocr = OCRAgent(menu_country_code="AUTO", probe_languages=args.langs)
    try:
        auto_out = auto_ocr.run(data_for_ocr, options=opts)
    except Exception as exc:
        print("[ERROR] auto OCR language detection failed:", str(exc))
        return 1
    t_auto = int((time.perf_counter() - t_auto_s) * 1000)

    manual_runs = []
    manual_errors = []
    for lang in args.langs:
        t_lang_s = time.perf_counter()
        agent = OCRAgent(ocr_lang_override=lang)
        try:
            out = agent.run(data_for_ocr, options=opts)
            summary = summarize_output(lang=lang, out=out, top=args.top)
            summary["elapsed_ms"] = int((time.perf_counter() - t_lang_s) * 1000)
            manual_runs.append(summary)
        except Exception as exc:
            manual_errors.append({"lang": lang, "error": str(exc)})

    top_n = max(0, int(args.top))
    print("=== OCR Language Detection Result ===")
    print(f"image              : {args.image}")
    print(f"with preprocess    : {args.with_preprocess}")
    if args.with_preprocess:
        print(f"preprocessed       : {args.preprocessed_out}")
    print(f"probe languages    : {', '.join(args.langs)}")
    print(f"resolved ocr lang  : {auto_out.resolved_lang or '-'}")
    print(f"lang source        : {auto_out.lang_detection_source or '-'}")
    print(f"auto ocr lines     : {len(auto_out.lines)}")
    print(f"timings(ms)        : load={t_load}, preprocess={t_pre}, auto={t_auto}")
    print("")

    print_candidate_table(auto_out.lang_detection_candidates)
    print_manual_runs(manual_runs, top=top_n)

    if manual_errors:
        print("--- ERRORS ---")
        for item in manual_errors:
            print(f"[ERR] lang={item['lang']} | error={item['error']}")
        print("")

    ensure_parent(args.json_out)
    payload = {
        "input_image": args.image,
        "with_preprocess": bool(args.with_preprocess),
        "preprocessed_image": args.preprocessed_out if args.with_preprocess else "",
        "probe_languages": list(args.langs),
        "timings_ms": {
            "load": t_load,
            "preprocess": t_pre,
            "auto": t_auto,
        },
        "options": opts.model_dump() if hasattr(opts, "model_dump") else opts.dict(),
        "auto_result": auto_out.model_dump() if hasattr(auto_out, "model_dump") else auto_out.dict(),
        "manual_runs": manual_runs,
        "manual_errors": manual_errors,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"json saved         : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
