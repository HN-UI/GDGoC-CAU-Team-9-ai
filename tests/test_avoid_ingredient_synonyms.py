import unittest

from app.utils.avoid_ingredient_synonyms import (
    build_avoid_synonym_lookup,
    find_matching_avoid_canonical,
    get_display_name,
    get_menu_evidence_catalog,
    normalize_ingredient_token,
)


class AvoidIngredientSynonymsTest(unittest.TestCase):
    def test_lookup_excludes_cn_terms_and_keeps_supported_lang_terms(self):
        lookup = build_avoid_synonym_lookup()

        self.assertIn("egg", lookup)
        self.assertIn(normalize_ingredient_token("계란"), lookup)
        self.assertNotIn(normalize_ingredient_token("鸡蛋"), lookup)

    def test_display_name_fallback_for_spanish_uses_supported_languages(self):
        self.assertEqual(get_display_name("egg", lang="es"), "egg")
        self.assertEqual(get_display_name("milk", lang="es"), "milk")

    def test_menu_evidence_catalog_excludes_cn_aliases(self):
        catalog = get_menu_evidence_catalog()
        milk_direct_terms = [term.casefold() for term in catalog["milk"]["direct"]]
        self.assertIn("milk", milk_direct_terms)
        self.assertNotIn("牛奶".casefold(), milk_direct_terms)

    def test_find_matching_canonical_supports_same_family_match(self):
        self.assertEqual(find_matching_avoid_canonical("cheese", {"milk"}), "milk")
        self.assertEqual(find_matching_avoid_canonical("milk", {"cheese"}), "cheese")


if __name__ == "__main__":
    unittest.main()
