from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.agents._0_contracts import OCRLanguageCandidate, OCRLine, OCROptions, OCROutput


class OCRAgent:
    """PaddleOCR text extractor with optional OCR-based language auto detection."""

    SUPPORTED_OCR_LANGS: Tuple[str, ...] = ("korean", "en", "es")
    DEFAULT_AUTO_LANGS: Tuple[str, ...] = SUPPORTED_OCR_LANGS
    CJK_LANGS = {"korean"}
    SPANISH_HINT_CHARS = set("áéíóúüñÁÉÍÓÚÜÑ¿¡")
    _PADDLE_OCR_CLASS = None
    _PADDLE_OCR_CLASS_LOCK = threading.Lock()
    _SHARED_OCR_ENGINE_CACHE: Dict[Tuple[str, int, int, int], Any] = {}
    _SHARED_OCR_ENGINE_CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        menu_country_code: Optional[str] = "AUTO",
        ocr_lang_override: Optional[str] = None,
        text_det_limit_side_len: int = 1216,
        text_recognition_batch_size: int = 4,
        cpu_threads: int = 6,
        det_model_name: str = "PP-OCRv5_mobile_det",
        probe_languages: Optional[Sequence[str]] = None,
        probe_text_det_limit_side_len: int = 512,
        probe_text_recognition_batch_size: int = 8,
        probe_cpu_threads: int = 4,
        probe_image_max_side: int = 960,
        probe_center_crop_ratio: float = 0.72,
        probe_large_image_threshold: int = 1800,
        probe_refine_top_k: int = 2,
        probe_early_exit_score_gap: float = 0.12,
        probe_early_exit_min_score: float = 0.72,
        probe_early_exit_min_script_ratio: float = 0.6,
    ):
        resolved_lang = self._resolve_lang(menu_country_code=menu_country_code, ocr_lang_override=ocr_lang_override)
        self.requested_lang = resolved_lang
        self.lang = resolved_lang or "auto"
        self.lang_source = "manual" if resolved_lang else "auto"
        self.text_det_limit_side_len = max(256, int(text_det_limit_side_len))
        self.text_recognition_batch_size = max(1, int(text_recognition_batch_size))
        self.cpu_threads = max(1, int(cpu_threads))
        self.det_model_name = (det_model_name or "PP-OCRv5_mobile_det").strip()
        self.probe_languages = self._normalize_probe_languages(probe_languages)
        self.probe_text_det_limit_side_len = max(256, int(probe_text_det_limit_side_len))
        self.probe_text_recognition_batch_size = max(1, int(probe_text_recognition_batch_size))
        self.probe_cpu_threads = max(1, int(probe_cpu_threads))
        self.probe_image_max_side = max(320, int(probe_image_max_side))
        self.probe_center_crop_ratio = min(0.95, max(0.4, float(probe_center_crop_ratio)))
        self.probe_large_image_threshold = max(960, int(probe_large_image_threshold))
        self.probe_refine_top_k = max(1, int(probe_refine_top_k))
        self.probe_early_exit_score_gap = max(0.0, float(probe_early_exit_score_gap))
        self.probe_early_exit_min_score = max(0.0, min(1.0, float(probe_early_exit_min_score)))
        self.probe_early_exit_min_script_ratio = max(0.0, min(1.0, float(probe_early_exit_min_script_ratio)))
        self.last_detection_candidates: List[OCRLanguageCandidate] = []
        self.last_resolved_lang = self.lang
        self.last_lang_source = self.lang_source

    def run(self, image_bytes: bytes, options: Optional[OCROptions] = None) -> OCROutput:
        opts = options or OCROptions()
        image = self._decode_image(image_bytes)
        if image is None:
            return OCROutput(lines=[], texts=[], resolved_lang="", lang_detection_source="")

        resolved_lang, lang_source, candidates = self._resolve_run_lang(image)
        keep_bbox = bool(opts.include_bbox)
        raw = self._run_ocr_raw(
            image=image,
            lang=resolved_lang,
            text_det_limit_side_len=self.text_det_limit_side_len,
            text_recognition_batch_size=self.text_recognition_batch_size,
            cpu_threads=self.cpu_threads,
        )
        lines = self._filter_and_sort_lines(
            self._parse_lines(raw),
            min_confidence=opts.min_confidence,
            include_bbox=keep_bbox,
        )

        self.lang = resolved_lang
        self.lang_source = lang_source
        self.last_resolved_lang = resolved_lang
        self.last_lang_source = lang_source
        self.last_detection_candidates = list(candidates)

        return OCROutput(
            lines=lines,
            texts=[ln.text for ln in lines],
            resolved_lang=resolved_lang,
            lang_detection_source=lang_source,
            lang_detection_candidates=list(candidates),
        )

    def run_from_file(self, image_path: str, options: Optional[OCROptions] = None) -> OCROutput:
        with open(image_path, "rb") as f:
            return self.run(f.read(), options=options)

    def _resolve_run_lang(self, image) -> Tuple[str, str, List[OCRLanguageCandidate]]:
        if self.requested_lang:
            return self.requested_lang, "manual", []

        candidates = self._detect_lang_candidates(image)
        if candidates:
            best = candidates[0]
            return best.lang, "auto", candidates

        fallback_lang = "en"
        return fallback_lang, "fallback", []

    def _detect_lang_candidates(self, image) -> List[OCRLanguageCandidate]:
        probe_views = self._build_probe_views(image)
        if not probe_views:
            return []

        first_candidates, first_line_map = self._score_probe_view(probe_views[0], self.probe_languages)
        if len(probe_views) == 1 or self._should_early_exit(first_candidates):
            return first_candidates

        refine_langs = [cand.lang for cand in first_candidates[: self.probe_refine_top_k]]
        if not refine_langs:
            return first_candidates

        merged_line_map: Dict[str, List[OCRLine]] = {
            lang: list(first_line_map.get(lang, []))
            for lang in refine_langs
        }
        for probe_image in probe_views[1:]:
            _, extra_line_map = self._score_probe_view(probe_image, refine_langs)
            for lang, lines in extra_line_map.items():
                merged_line_map.setdefault(lang, []).extend(lines)

            refined_candidates = self._build_candidates_from_line_map(merged_line_map)
            if self._should_early_exit(refined_candidates):
                break

        refined_by_lang = {
            cand.lang: cand
            for cand in self._build_candidates_from_line_map(merged_line_map)
        }
        merged_candidates = [
            refined_by_lang.get(cand.lang, cand)
            for cand in first_candidates
        ]
        return self._sort_candidates(merged_candidates)

    def _build_probe_views(self, image) -> List[Any]:
        base_view = self._resize_for_probe(image, self.probe_image_max_side)
        views = [base_view]

        max_side = max(int(image.shape[0]), int(image.shape[1]))
        if max_side >= self.probe_large_image_threshold:
            center_crop = self._extract_center_crop(image, self.probe_center_crop_ratio)
            center_crop = self._resize_for_probe(center_crop, self.probe_image_max_side)
            if center_crop is not None and center_crop.size > 0:
                views.append(center_crop)
        return views

    @staticmethod
    def _resize_for_probe(image, max_side: int):
        if image is None or getattr(image, "size", 0) == 0:
            return image
        h, w = image.shape[:2]
        current_max = max(int(h), int(w))
        if current_max <= max_side:
            return image
        scale = float(max_side) / float(current_max)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _extract_center_crop(image, ratio: float):
        if image is None or getattr(image, "size", 0) == 0:
            return image
        h, w = image.shape[:2]
        crop_h = max(1, int(round(h * ratio)))
        crop_w = max(1, int(round(w * ratio)))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        return image[top : top + crop_h, left : left + crop_w].copy()

    @staticmethod
    def _merge_probe_lines(lines: List[OCRLine]) -> List[OCRLine]:
        best_by_text: Dict[str, OCRLine] = {}
        for line in lines:
            text = " ".join((line.text or "").split()).strip()
            if not text:
                continue
            key = text.casefold()
            current = best_by_text.get(key)
            if current is None or line.confidence > current.confidence:
                best_by_text[key] = OCRLine(text=text, confidence=line.confidence, bbox=line.bbox)
        merged = list(best_by_text.values())
        merged.sort(key=lambda x: (OCRAgent._top_y(x.bbox), OCRAgent._left_x(x.bbox), x.text))
        return merged

    def _score_probe_view(
        self,
        image,
        langs: Sequence[str],
    ) -> Tuple[List[OCRLanguageCandidate], Dict[str, List[OCRLine]]]:
        line_map: Dict[str, List[OCRLine]] = {}
        for lang in langs:
            try:
                raw = self._run_ocr_raw(
                    image=image,
                    lang=lang,
                    text_det_limit_side_len=self.probe_text_det_limit_side_len,
                    text_recognition_batch_size=self.probe_text_recognition_batch_size,
                    cpu_threads=self.probe_cpu_threads,
                )
                lines = self._filter_and_sort_lines(
                    self._parse_lines(raw),
                    min_confidence=0.25,
                    include_bbox=True,
                )
            except Exception:
                continue
            line_map[lang] = lines
        return self._build_candidates_from_line_map(line_map), line_map

    def _build_candidates_from_line_map(
        self,
        line_map: Dict[str, List[OCRLine]],
    ) -> List[OCRLanguageCandidate]:
        candidates: List[OCRLanguageCandidate] = []
        for lang, raw_lines in line_map.items():
            lines = self._merge_probe_lines(raw_lines)
            texts = [line.text for line in lines if line.text]
            avg_conf = self._average_confidence(lines)
            script_ratio = self._expected_script_ratio(texts, lang)
            score = self._candidate_score(
                lang=lang,
                texts=texts,
                line_count=len(lines),
                avg_confidence=avg_conf,
                script_ratio=script_ratio,
            )
            candidates.append(
                OCRLanguageCandidate(
                    lang=lang,
                    score=round(score, 6),
                    line_count=len(lines),
                    avg_confidence=avg_conf,
                    script_ratio=script_ratio,
                )
            )
        return self._sort_candidates(candidates)

    @staticmethod
    def _sort_candidates(candidates: List[OCRLanguageCandidate]) -> List[OCRLanguageCandidate]:
        # Tie-break rule:
        # If en/es scores are identical, prefer Spanish for bilingual menus.
        lang_priority = {"es": 2, "en": 1}
        ordered = list(candidates)
        ordered.sort(
            key=lambda item: (
                item.score,
                item.line_count,
                item.avg_confidence,
                lang_priority.get((item.lang or "").strip().lower(), 0),
            ),
            reverse=True,
        )
        return ordered

    def _should_early_exit(self, candidates: List[OCRLanguageCandidate]) -> bool:
        if not candidates:
            return False
        if len(candidates) == 1:
            return True

        best = candidates[0]
        second = candidates[1]
        score_gap = float(best.score) - float(second.score)
        if best.score >= self.probe_early_exit_min_score and score_gap >= self.probe_early_exit_score_gap:
            return True
        if (
            best.script_ratio >= self.probe_early_exit_min_script_ratio
            and best.line_count >= 2
            and score_gap >= (self.probe_early_exit_score_gap * 0.75)
        ):
            return True
        return False

    def _run_ocr_raw(
        self,
        image,
        lang: str,
        text_det_limit_side_len: int,
        text_recognition_batch_size: int,
        cpu_threads: int,
    ):
        engine = self._get_ocr_engine(
            lang=lang,
            text_det_limit_side_len=text_det_limit_side_len,
            text_recognition_batch_size=text_recognition_batch_size,
            cpu_threads=cpu_threads,
        )
        return engine.ocr(image)

    def _get_ocr_engine(
        self,
        lang: str,
        text_det_limit_side_len: int,
        text_recognition_batch_size: int,
        cpu_threads: int,
    ):
        cache_key = (lang, int(text_det_limit_side_len), int(text_recognition_batch_size), int(cpu_threads))
        with self._SHARED_OCR_ENGINE_CACHE_LOCK:
            cached = self._SHARED_OCR_ENGINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        PaddleOCR = self._get_paddleocr_class()

        kwargs_v1 = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_recognition_batch_size": text_recognition_batch_size,
            "device": "cpu",
            "enable_mkldnn": True,
            "cpu_threads": cpu_threads,
        }
        if lang in {"en", "es"} and self.det_model_name:
            kwargs_v1["text_detection_model_name"] = self.det_model_name

        kwargs_v2 = {
            "lang": lang,
            "det_limit_side_len": text_det_limit_side_len,
            "rec_batch_num": text_recognition_batch_size,
            "use_gpu": False,
            "enable_mkldnn": True,
            "cpu_threads": cpu_threads,
            "use_angle_cls": False,
        }

        try:
            engine = PaddleOCR(**kwargs_v1)
        except TypeError:
            try:
                engine = PaddleOCR(**kwargs_v2)
            except Exception as exc:
                raise RuntimeError(f"PaddleOCR 엔진 초기화 실패: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR 엔진 초기화 실패: {exc}") from exc

        with self._SHARED_OCR_ENGINE_CACHE_LOCK:
            existing = self._SHARED_OCR_ENGINE_CACHE.get(cache_key)
            if existing is not None:
                return existing
            self._SHARED_OCR_ENGINE_CACHE[cache_key] = engine
        return engine

    @classmethod
    def _get_paddleocr_class(cls):
        try:
            cls._prepare_paddle_env()
            with cls._PADDLE_OCR_CLASS_LOCK:
                if cls._PADDLE_OCR_CLASS is not None:
                    return cls._PADDLE_OCR_CLASS
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "paddleocr가 설치되지 않았습니다. `pip install paddleocr paddlepaddle` 후 다시 시도하세요."
            ) from exc

        with cls._PADDLE_OCR_CLASS_LOCK:
            if cls._PADDLE_OCR_CLASS is None:
                cls._PADDLE_OCR_CLASS = PaddleOCR
            return cls._PADDLE_OCR_CLASS

    @staticmethod
    def _prepare_paddle_env() -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    @classmethod
    def warmup_shared_engines(
        cls,
        langs: Optional[Sequence[str]] = None,
        preload_probe: bool = True,
        preload_full: bool = False,
    ) -> None:
        target_langs = cls._normalize_probe_languages(langs)
        agent = cls(menu_country_code="AUTO", probe_languages=target_langs)

        # Import PaddleOCR once up front so later engine creation avoids import latency.
        agent._get_paddleocr_class()

        for lang in target_langs:
            if preload_probe:
                agent._get_ocr_engine(
                    lang=lang,
                    text_det_limit_side_len=agent.probe_text_det_limit_side_len,
                    text_recognition_batch_size=agent.probe_text_recognition_batch_size,
                    cpu_threads=agent.probe_cpu_threads,
                )
            if preload_full:
                agent._get_ocr_engine(
                    lang=lang,
                    text_det_limit_side_len=agent.text_det_limit_side_len,
                    text_recognition_batch_size=agent.text_recognition_batch_size,
                    cpu_threads=agent.cpu_threads,
                )

    @classmethod
    def _normalize_probe_languages(cls, probe_languages: Optional[Sequence[str]]) -> List[str]:
        raw_values = list(probe_languages) if probe_languages else list(cls.DEFAULT_AUTO_LANGS)
        out: List[str] = []
        seen = set()
        for value in raw_values:
            if not isinstance(value, str):
                continue
            lang = value.strip().lower()
            if not lang or lang in seen or lang not in cls.SUPPORTED_OCR_LANGS:
                continue
            seen.add(lang)
            out.append(lang)
        return out or list(cls.DEFAULT_AUTO_LANGS)

    @staticmethod
    def _resolve_lang(menu_country_code: Optional[str], ocr_lang_override: Optional[str]) -> Optional[str]:
        lang_aliases = {
            "auto": None,
            "ko": "korean",
            "korean": "korean",
            "en": "en",
            "es": "es",
        }
        override = (ocr_lang_override or "").strip().lower()
        if override:
            return lang_aliases.get(override)

        country = (menu_country_code or "").strip().upper()
        if not country or country in {"AUTO", "UNKNOWN", "NONE"}:
            return None

        country = country.replace("_", "-").split("-", 1)[0]
        country_to_lang = {
            "KR": "korean",
            "US": "en",
            "GB": "en",
            "AU": "en",
            "CA": "en",
            "ES": "es",
            "MX": "es",
            "AR": "es",
            "CL": "es",
            "CO": "es",
            "PE": "es",
        }
        return country_to_lang.get(country)

    @staticmethod
    def _decode_image(image_bytes: bytes):
        if not image_bytes:
            return None
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def _filter_and_sort_lines(
        lines: List[OCRLine],
        min_confidence: float,
        include_bbox: bool = True,
    ) -> List[OCRLine]:
        filtered = [
            OCRLine(text=ln.text.strip(), confidence=ln.confidence, bbox=ln.bbox)
            for ln in lines
            if ln.text and ln.text.strip() and ln.confidence >= min_confidence
        ]
        filtered.sort(key=lambda x: (OCRAgent._top_y(x.bbox), OCRAgent._left_x(x.bbox)))
        if not include_bbox:
            return [
                OCRLine(text=ln.text, confidence=ln.confidence, bbox=[])
                for ln in filtered
            ]
        return filtered

    @staticmethod
    def _average_confidence(lines: List[OCRLine]) -> float:
        if not lines:
            return 0.0
        return max(0.0, min(1.0, sum(line.confidence for line in lines) / len(lines)))

    @classmethod
    def _candidate_score(
        cls,
        lang: str,
        texts: List[str],
        line_count: int,
        avg_confidence: float,
        script_ratio: float,
    ) -> float:
        visible_chars = sum(len("".join(ch for ch in text if not ch.isspace())) for text in texts)
        coverage_score = min(1.0, visible_chars / 64.0)
        line_score = min(1.0, line_count / 12.0)
        latin_penalty = 0.0
        if lang in cls.CJK_LANGS:
            latin_ratio = cls._character_profile(texts)["latin_ratio"]
            latin_penalty = max(0.0, latin_ratio - 0.55) * 0.15

        total = (
            avg_confidence * 0.48
            + coverage_score * 0.14
            + line_score * 0.10
            + script_ratio * 0.34
            - latin_penalty
        )
        return max(0.0, min(1.0, total))

    @classmethod
    def _expected_script_ratio(cls, texts: List[str], lang: str) -> float:
        profile = cls._character_profile(texts)
        hangul_ratio = profile["hangul_ratio"]
        latin_ratio = profile["latin_ratio"]
        spanish_ratio = profile["spanish_ratio"]

        if lang == "korean":
            return min(1.0, hangul_ratio)
        if lang == "es":
            return min(1.0, latin_ratio + (spanish_ratio * 0.35))
        if lang == "en":
            return max(0.0, min(1.0, latin_ratio - (spanish_ratio * 0.10)))
        return 0.0

    @classmethod
    def _character_profile(cls, texts: List[str]) -> Dict[str, float]:
        joined = "".join(texts)
        signal_chars = 0
        hangul = 0
        latin = 0
        spanish = 0

        for ch in joined:
            code = ord(ch)
            if ch in cls.SPANISH_HINT_CHARS:
                spanish += 1

            if 0xAC00 <= code <= 0xD7A3:
                hangul += 1
                signal_chars += 1
            elif OCRAgent._is_latin_char(ch):
                latin += 1
                signal_chars += 1

        denom = float(signal_chars or 1)
        return {
            "hangul_ratio": hangul / denom,
            "latin_ratio": latin / denom,
            "spanish_ratio": min(1.0, spanish / denom),
        }

    @staticmethod
    def _is_latin_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x0041 <= code <= 0x005A
            or 0x0061 <= code <= 0x007A
            or 0x00C0 <= code <= 0x00FF
            or 0x0100 <= code <= 0x017F
        )

    def _parse_lines(self, raw: Any) -> List[OCRLine]:
        if not isinstance(raw, list) or not raw:
            return []

        if isinstance(raw[0], dict):
            return self._parse_dict_result(raw)

        groups = [raw] if self._is_line_item(raw[0]) else [g for g in raw if isinstance(g, list)]
        out: List[OCRLine] = []
        for group in groups:
            for item in group:
                line = self._parse_item(item)
                if line is not None:
                    out.append(line)
        return out

    @staticmethod
    def _parse_dict_result(raw: List[Any]) -> List[OCRLine]:
        out: List[OCRLine] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            polys = OCRAgent._first_present(item, ["dt_polys", "rec_polys", "polys", "boxes"], default=[])
            texts = item.get("rec_texts") or []
            scores = item.get("rec_scores") or []

            n = min(len(polys), len(texts), len(scores))
            for i in range(n):
                bbox = OCRAgent._to_bbox_points(polys[i])
                text = str(texts[i] or "")
                try:
                    conf = float(scores[i])
                except Exception:
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                out.append(OCRLine(text=text, confidence=conf, bbox=bbox))
        return out

    @staticmethod
    def _is_line_item(item: Any) -> bool:
        return isinstance(item, list) and len(item) >= 2 and isinstance(item[0], (list, tuple))

    @staticmethod
    def _parse_item(item: Any) -> Optional[OCRLine]:
        if not isinstance(item, list) or len(item) < 2:
            return None

        bbox = OCRAgent._to_bbox_points(item[0])
        if not bbox:
            return None

        txt_raw = item[1]
        text = ""
        conf = 0.0
        if isinstance(txt_raw, (list, tuple)) and len(txt_raw) >= 1:
            text = str(txt_raw[0] or "")
            if len(txt_raw) >= 2:
                try:
                    conf = float(txt_raw[1])
                except Exception:
                    conf = 0.0
        else:
            text = str(txt_raw or "")
        conf = max(0.0, min(1.0, conf))
        return OCRLine(text=text, confidence=conf, bbox=bbox)

    @staticmethod
    def _first_present(data: dict, keys: List[str], default: Any):
        for k in keys:
            v = data.get(k)
            if v is not None:
                return v
        return default

    @staticmethod
    def _to_bbox_points(bbox_raw: Any) -> List[List[float]]:
        if bbox_raw is None:
            return []

        pts_src = bbox_raw.tolist() if hasattr(bbox_raw, "tolist") else bbox_raw
        if not isinstance(pts_src, (list, tuple)):
            return []

        pts: List[List[float]] = []
        for pt in pts_src:
            cur = pt.tolist() if hasattr(pt, "tolist") else pt
            if not isinstance(cur, (list, tuple)) or len(cur) < 2:
                continue
            try:
                pts.append([float(cur[0]), float(cur[1])])
            except Exception:
                continue
        return pts

    @staticmethod
    def _top_y(bbox: List[List[float]]) -> float:
        return min((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _left_x(bbox: List[List[float]]) -> float:
        return min((p[0] for p in bbox), default=0.0)
