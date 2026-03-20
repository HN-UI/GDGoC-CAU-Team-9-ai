import unittest

from app.agents._0_contracts import OCRLine
from app.agents._eval_3_extractor import OCRMenuJudgeAgent


class _FakeGemma:
    def __init__(self, responses):
        self.responses = list(responses)

    def image_part_from_bytes(self, data, mime_type):
        return {"mime": mime_type, "size": len(data)}

    def generate_text(self, contents, max_output_tokens=900):
        if not self.responses:
            raise RuntimeError("no fake response")
        return self.responses.pop(0)


class OCRMenuJudgeAgentTest(unittest.TestCase):
    def test_marks_menu_with_text_mapping(self):
        fake = _FakeGemma(['{"items":["Acai Bowl"]}'])
        agent = OCRMenuJudgeAgent(fake)
        out = agent.run(["Acai Bowl", "$198"])

        self.assertEqual([it.text for it in out.items], ["Acai Bowl", "$198"])
        self.assertEqual([it.label for it in out.items], ["menu_item", "other"])
        self.assertEqual(out.menu_texts, ["Acai Bowl"])

    def test_supports_image_mode_with_items_array(self):
        fake = _FakeGemma(['{"items":["Kimchi Fried Rice"]}'])
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Kimchi Fried Rice", confidence=1.0, bbox=[]),
            OCRLine(text="12000", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(lines, image_bytes=b"abc", image_mime="image/png", use_image_context=True)

        self.assertEqual([it.label for it in out.items], ["menu_item", "other"])
        self.assertEqual(out.menu_texts, ["Kimchi Fried Rice"])

    def test_missing_items_fallback_to_other(self):
        fake = _FakeGemma(['{"items":[]}'])
        agent = OCRMenuJudgeAgent(fake)
        out = agent.run(["A", "B"])
        self.assertEqual([it.label for it in out.items], ["other", "other"])
        self.assertEqual(out.menu_texts, [])

    def test_recovers_spanish_menu_titles_when_labeled_as_description(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Mussels/Mejillones 18 GF", "label": "description"},'
                '{"index": 1, "text": "Baby Pimiento Rellenos20", "label": "description"},'
                '{"index": 2, "text": "Sauteed with Marinara Sauce.", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Mussels/Mejillones 18 GF", confidence=1.0, bbox=[]),
            OCRLine(text="Baby Pimiento Rellenos20", confidence=1.0, bbox=[]),
            OCRLine(text="Sauteed with Marinara Sauce.", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Mussels/Mejillones", out.menu_texts)
        self.assertIn("Baby Pimiento Rellenos", out.menu_texts)
        self.assertNotIn("Sauteed with Marinara Sauce", out.menu_texts)

    def test_spanish_rule_discards_noisy_llm_menu_labels(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Mussels/Mejillones 18 GF", "label": "menu_item"},'
                '{"index": 1, "text": "Sauteed with Marinara Sauce.", "label": "menu_item"},'
                '{"index": 2, "text": "PLEASEINFORMSTAFFOFANYFOODALLERGIESBEFOREORDERING", "label": "menu_item"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Mussels/Mejillones 18 GF", confidence=1.0, bbox=[]),
            OCRLine(text="Sauteed with Marinara Sauce.", confidence=1.0, bbox=[]),
            OCRLine(text="PLEASEINFORMSTAFFOFANYFOODALLERGIESBEFOREORDERING", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertEqual(out.menu_texts, ["Mussels/Mejillones"])

    def test_spanish_rule_handles_tags_and_avoids_wrong_parenthesis_merge(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Patatas Bravas11 GFIVG", "label": "description"},'
                '{"index": 1, "text": "Tabla Iberica24", "label": "description"},'
                '{"index": 2, "text": "(Iberico)y Queso", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Patatas Bravas11 GFIVG", confidence=1.0, bbox=[]),
            OCRLine(text="Tabla Iberica24", confidence=1.0, bbox=[]),
            OCRLine(text="(Iberico)y Queso", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Patatas Bravas", out.menu_texts)
        self.assertIn("Tabla Iberica", out.menu_texts)
        self.assertNotIn("Tabla Iberica (Iberico)y Queso", out.menu_texts)

    def test_spanish_rule_merges_title_continuation_after_comma_price(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Combinacion de Camarones,24", "label": "description"},'
                '{"index": 1, "text": "6 Deep Fried Shrimps with Orange Ginger", "label": "description"},'
                '{"index": 2, "text": "Churrasco,yChorizo", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Combinacion de Camarones,24", confidence=1.0, bbox=[]),
            OCRLine(text="6 Deep Fried Shrimps with Orange Ginger", confidence=1.0, bbox=[]),
            OCRLine(text="Churrasco,yChorizo", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Combinacion de Camarones Churrasco,yChorizo", out.menu_texts)


if __name__ == "__main__":
    unittest.main()
