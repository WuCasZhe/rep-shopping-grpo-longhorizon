"""Observation projection preserves actionable state and bounds visible tokens."""

import json
import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.environment.projection import (
    ObservationProjectionError,
    TRUNCATION_MARKER,
    project_observation,
)
from shopping_grpo.environment.observation import render_structured_observation


def search_page(product_count=12, page=1):
    products = [
        {"rank": (page - 1) * 20 + index + 1,
         "asin": str(100000000000 + (page - 1) * 20 + index),
         "title": f"product title {index} " + "x" * 80,
         "price": index + 1}
        for index in range(product_count)
    ]
    return render_structured_observation({
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results", "search_available": False,
        "query": "useful product", "normalized_query": "useful product",
        "page": page, "total_pages": 2, "total_results": 40,
        "rank_start": (page - 1) * 20 + 1, "rank_end": page * 20,
        "actions": ["back to search", "next >", *[p["asin"] for p in products]],
        "products": products,
    })


class ObservationProjectionTest(unittest.TestCase):
    def test_structured_subpage_preserves_variant_state_when_content_is_long(self):
        raw = render_structured_observation({
            "observation_version": "shopping-observation-v2",
            "page_type": "information_subpage", "subpage": "Features",
            "search_available": False, "actions": ["back to search", "< prev"],
            "product": {"asin": "123456789012", "title": "long title " * 100,
                        "brand": "brand", "category": "category",
                        "key_attributes": ["350W", "96000mAh"], "price": 599},
            "selected_price": 699, "selected_options": {"capacity": "96000mAh"},
            "available_options": {"capacity": ["96000mAh", "48000mAh"]},
            "content": "long detail " * 500,
        })
        visible, meta = project_observation("view_features", raw, count_tokens=len,
                                           detail_token_budget=900, generic_token_budget=128)
        self.assertEqual(meta.page_type, "information_subpage")
        self.assertEqual(meta.token_budget, 900)
        self.assertLessEqual(len(visible), 900)
        for key in ("asin", "price", "key_attributes", "selected_options", "available_options", "subpage"):
            line = next(x for x in raw.splitlines() if x.startswith(key + ":"))
            self.assertIn(line, visible)
        self.assertEqual(clickable_buttons(visible), clickable_buttons(raw))

    def test_search_matching_fields_are_not_title_snippets(self):
        raw = render_structured_observation({
            "observation_version": "shopping-observation-v2", "page_type": "search_results",
            "actions": ["back to search", "123456789012"], "search_available": False,
            "products": [{"asin": "123456789012", "rank": 1, "price": "199.90",
                          "brand": "brand", "category": "charger", "key_attributes": ["350W"],
                          "title": "title " * 300}],
        })
        visible, _ = project_observation("search_products", raw, count_tokens=len, token_budget=600)
        self.assertIn("123456789012|199.90|brand|charger|350W|", visible)

    def test_rich_twenty_product_page_fits_without_corrupting_prices(self):
        products = [{"asin": str(123456789000 + i), "rank": i + 1,
                     "price": "12345.67", "brand": "long brand " * 10,
                     "category": "long category " * 10,
                     "key_attributes": ["specification " * 40], "title": "title " * 100}
                    for i in range(20)]
        raw = render_structured_observation({
            "observation_version": "shopping-observation-v2", "page_type": "search_results",
            "actions": ["back to search", *[p["asin"] for p in products]],
            "search_available": False, "products": products,
        })
        visible, _ = project_observation("search_products", raw, count_tokens=len, token_budget=1536)
        self.assertEqual(product_ids(visible), product_ids(raw))
        self.assertEqual(visible.count("|12345.67|"), 20)
        self.assertIn("product_fields_compacted=true", visible)

    def test_search_projection_preserves_every_current_page_product(self):
        raw = search_page(product_count=20)
        visible, meta = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )

        self.assertLessEqual(len(visible), 1400)
        self.assertTrue(meta.truncated)
        self.assertTrue(meta.critical_footer_preserved)
        self.assertIn(TRUNCATION_MARKER, visible)
        self.assertEqual(
            {button.casefold() for button in clickable_buttons(visible) if not button.isdigit()},
            {"back to search", "next >"},
        )
        self.assertEqual(
            set(product_ids(visible)),
            {button for button in clickable_buttons(visible) if button.isdigit()},
        )
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_guard_accepts_last_product_instead_of_creating_blind_spot(self):
        raw = search_page(product_count=20)
        visible, _ = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )
        last_asin = product_ids(raw)[-1]

        self.assertIsNone(
            action_reject_reason("open_product", {"asin": last_asin}, visible)
        )

    def test_second_environment_page_preserves_products_21_through_40(self):
        raw = search_page(product_count=20, page=2)
        visible, _ = project_observation(
            "next_page",
            raw,
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )

        self.assertIn("Page 2 of 2", visible)
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_capacity_mismatch_fails_instead_of_silently_dropping_products(self):
        raw = search_page(product_count=20)
        with self.assertRaisesRegex(ObservationProjectionError, "page capacity"):
            project_observation(
                "search_products",
                raw,
                count_tokens=len,
                token_budget=1400,
                search_top_k=10,
            )

    def test_short_product_page_is_identity_projection(self):
        raw = (
            "Product [SEP] price: 20 [SEP] Buy Now"
            "\n\n搜索功能是否可用: False"
            '\n\n可点击的按钮: ["back to search", "buy now"]'
        )
        visible, meta = project_observation(
            "open_product",
            raw,
            count_tokens=len,
            token_budget=448,
        )

        self.assertEqual(visible, raw)
        self.assertFalse(meta.truncated)

    def test_generic_projection_keeps_complete_footer(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=300,
        )

        self.assertLessEqual(len(visible), 300)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertIn("TAIL_SPECIFICATION", visible)
        self.assertTrue(meta.critical_footer_preserved)

    def test_long_page_without_action_footer_fails_closed(self):
        with self.assertRaisesRegex(ObservationProjectionError, "action footer"):
            project_observation(
                "unknown",
                "x" * 1000,
                count_tokens=len,
                generic_token_budget=128,
            )

    def test_structured_search_projection_preserves_all_twenty_products(self):
        products = [
            {
                "rank": index,
                "asin": f"{index:012d}",
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index in range(1, 21)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": [
                    "back to search",
                    "next >",
                    *[product["asin"] for product in products],
                ],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 2,
                "total_results": 40,
                "rank_start": 1,
                "rank_end": 20,
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )
        self.assertEqual(product_ids(visible), product_ids(raw))
        self.assertLessEqual(len(visible), 1400)

    def test_structured_projection_preserves_mixed_catalog_id_lengths(self):
        asins = ["12345678", "123456789", "1234567890", "35842622441", "123456789012"]
        products = [
            {
                "rank": index,
                "asin": asin,
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index, asin in enumerate(asins, start=1)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", *asins],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 1,
                "total_results": len(products),
                "rank_start": 1,
                "rank_end": len(products),
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=700,
            search_top_k=20,
        )

        self.assertEqual(product_ids(visible), asins)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
