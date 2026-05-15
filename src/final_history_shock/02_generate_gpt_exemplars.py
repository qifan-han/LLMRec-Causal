"""02 — Generate GPT recommendation exemplars for prompt calibration.

180 GPT calls: 6 categories x 10 personas x 3 regimes.
Used for few-shot prompt calibration of local LLM, not fine-tuning.

Output:
  data/final_history_shock/gpt_exemplars/gpt_recommendation_exemplars.jsonl
  data/final_history_shock/gpt_exemplars/selected_few_shot_examples.json

Usage:
  python 02_generate_gpt_exemplars.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import gpt_json_call, append_jsonl, load_jsonl, DATA_DIR

EXEMPLAR_DIR = DATA_DIR / "gpt_exemplars"
EXEMPLAR_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = EXEMPLAR_DIR / "gpt_recommendation_exemplars.jsonl"

CATEGORIES = [
    "headphones", "phone_chargers", "wireless_routers",
    "coffee_makers", "office_chairs", "running_shoes",
]

MINI_CATALOGS = {
    "headphones": [
        {"id": "hp_01", "brand": "Sony", "model": "WH-1000XM5", "price": 348, "type": "over-ear wireless ANC", "best_for": "commuting, travel"},
        {"id": "hp_02", "brand": "JBL", "model": "Tune 510BT", "price": 35, "type": "on-ear wireless", "best_for": "casual listening, budget"},
        {"id": "hp_03", "brand": "Beyerdynamic", "model": "DT 770 Pro", "price": 159, "type": "over-ear wired studio", "best_for": "studio monitoring, mixing"},
        {"id": "hp_04", "brand": "Apple", "model": "AirPods Max", "price": 549, "type": "over-ear wireless ANC", "best_for": "Apple ecosystem, premium"},
        {"id": "hp_05", "brand": "Audio-Technica", "model": "ATH-M50x", "price": 149, "type": "over-ear wired", "best_for": "studio, balanced sound"},
        {"id": "hp_06", "brand": "Sennheiser", "model": "HD 560S", "price": 199, "type": "open-back wired", "best_for": "audiophile, home listening"},
        {"id": "hp_07", "brand": "Anker", "model": "Soundcore Q45", "price": 99, "type": "over-ear wireless ANC", "best_for": "budget ANC, travel"},
        {"id": "hp_08", "brand": "Bose", "model": "QuietComfort 45", "price": 279, "type": "over-ear wireless ANC", "best_for": "comfort, noise cancelling"},
    ],
    "phone_chargers": [
        {"id": "pc_01", "brand": "Anker", "model": "Nano II 65W", "price": 36, "type": "USB-C GaN", "best_for": "laptop + phone charging"},
        {"id": "pc_02", "brand": "Apple", "model": "20W USB-C", "price": 19, "type": "USB-C single port", "best_for": "iPhone fast charging"},
        {"id": "pc_03", "brand": "Belkin", "model": "BoostCharge 3-in-1", "price": 65, "type": "wireless charging station", "best_for": "multi-device households"},
        {"id": "pc_04", "brand": "Samsung", "model": "25W Super Fast", "price": 15, "type": "USB-C single port", "best_for": "Samsung Galaxy fast charge"},
        {"id": "pc_05", "brand": "Ugreen", "model": "Nexode 100W", "price": 50, "type": "USB-C GaN multi-port", "best_for": "power users, laptop + devices"},
        {"id": "pc_06", "brand": "Anker", "model": "PowerPort III Nano", "price": 12, "type": "USB-C compact", "best_for": "travel, minimal size"},
        {"id": "pc_07", "brand": "RAVPower", "model": "PD Pioneer 45W", "price": 28, "type": "USB-C dual port", "best_for": "tablet + phone simultaneous"},
        {"id": "pc_08", "brand": "Mophie", "model": "Speedcharge 30W", "price": 35, "type": "USB-C compact", "best_for": "everyday fast charging"},
    ],
    "wireless_routers": [
        {"id": "wr_01", "brand": "TP-Link", "model": "Archer AX73", "price": 170, "type": "WiFi 6 dual-band", "best_for": "large homes"},
        {"id": "wr_02", "brand": "Netgear", "model": "Nighthawk RAX50", "price": 200, "type": "WiFi 6 dual-band", "best_for": "gaming, streaming"},
        {"id": "wr_03", "brand": "Asus", "model": "RT-AX86U", "price": 250, "type": "WiFi 6 gaming router", "best_for": "gamers, power users"},
        {"id": "wr_04", "brand": "Google", "model": "Nest Wifi Pro", "price": 200, "type": "WiFi 6E mesh", "best_for": "mesh coverage, simplicity"},
        {"id": "wr_05", "brand": "Linksys", "model": "Hydra Pro 6E", "price": 300, "type": "WiFi 6E tri-band", "best_for": "future-proofing, many devices"},
        {"id": "wr_06", "brand": "TP-Link", "model": "Archer AX21", "price": 70, "type": "WiFi 6 budget", "best_for": "apartments, budget"},
        {"id": "wr_07", "brand": "Eero", "model": "Eero 6+", "price": 140, "type": "WiFi 6 mesh", "best_for": "easy setup, Alexa integration"},
        {"id": "wr_08", "brand": "Asus", "model": "ZenWiFi AX", "price": 350, "type": "WiFi 6 mesh", "best_for": "whole-home coverage, premium"},
    ],
    "coffee_makers": [
        {"id": "cm_01", "brand": "Breville", "model": "Barista Express", "price": 700, "type": "espresso machine", "best_for": "home barista, espresso"},
        {"id": "cm_02", "brand": "Mr. Coffee", "model": "12-Cup Programmable", "price": 30, "type": "drip coffee maker", "best_for": "budget, simple drip"},
        {"id": "cm_03", "brand": "Nespresso", "model": "Vertuo Next", "price": 160, "type": "capsule machine", "best_for": "convenience, consistent espresso"},
        {"id": "cm_04", "brand": "Chemex", "model": "Classic 6-Cup", "price": 45, "type": "pour-over", "best_for": "clean flavor, manual brewing"},
        {"id": "cm_05", "brand": "Keurig", "model": "K-Supreme Plus", "price": 170, "type": "single-serve pod", "best_for": "quick single cups, variety"},
        {"id": "cm_06", "brand": "Bonavita", "model": "Connoisseur 8-Cup", "price": 100, "type": "precision drip", "best_for": "SCA-certified drip quality"},
        {"id": "cm_07", "brand": "AeroPress", "model": "Original", "price": 35, "type": "manual press", "best_for": "travel, camping, concentrated coffee"},
        {"id": "cm_08", "brand": "De'Longhi", "model": "Magnifica S", "price": 450, "type": "super-automatic espresso", "best_for": "one-touch espresso and latte"},
    ],
    "office_chairs": [
        {"id": "oc_01", "brand": "Herman Miller", "model": "Aeron", "price": 1395, "type": "ergonomic mesh", "best_for": "long hours, ergonomic support"},
        {"id": "oc_02", "brand": "Secretlab", "model": "Titan Evo 2022", "price": 499, "type": "gaming/office hybrid", "best_for": "gaming + work, adjustable"},
        {"id": "oc_03", "brand": "IKEA", "model": "Markus", "price": 229, "type": "high-back mesh", "best_for": "budget ergonomic"},
        {"id": "oc_04", "brand": "Steelcase", "model": "Leap V2", "price": 1189, "type": "ergonomic task chair", "best_for": "adjustability, back support"},
        {"id": "oc_05", "brand": "Autonomous", "model": "ErgoChair Pro", "price": 449, "type": "ergonomic mesh", "best_for": "mid-range ergonomic"},
        {"id": "oc_06", "brand": "HON", "model": "Ignition 2.0", "price": 350, "type": "mesh task chair", "best_for": "office standard, durability"},
        {"id": "oc_07", "brand": "Flash Furniture", "model": "Mid-Back Mesh", "price": 95, "type": "basic mesh", "best_for": "extreme budget, light use"},
        {"id": "oc_08", "brand": "Branch", "model": "Ergonomic Chair", "price": 349, "type": "direct-to-consumer ergo", "best_for": "home office, good value"},
    ],
    "running_shoes": [
        {"id": "rs_01", "brand": "Nike", "model": "Pegasus 41", "price": 130, "type": "daily trainer", "best_for": "everyday running, cushioning"},
        {"id": "rs_02", "brand": "Brooks", "model": "Ghost 15", "price": 140, "type": "neutral daily trainer", "best_for": "beginners, comfortable runs"},
        {"id": "rs_03", "brand": "Hoka", "model": "Clifton 9", "price": 145, "type": "max cushion daily", "best_for": "recovery runs, joint protection"},
        {"id": "rs_04", "brand": "Asics", "model": "Gel-Kayano 30", "price": 160, "type": "stability trainer", "best_for": "overpronators, long runs"},
        {"id": "rs_05", "brand": "New Balance", "model": "Fresh Foam 1080v13", "price": 160, "type": "premium cushion", "best_for": "marathon training, plush feel"},
        {"id": "rs_06", "brand": "Saucony", "model": "Kinvara 14", "price": 120, "type": "lightweight trainer", "best_for": "tempo runs, speed work"},
        {"id": "rs_07", "brand": "On", "model": "Cloudmonster", "price": 170, "type": "max cushion responsive", "best_for": "long runs, bounce-back feel"},
        {"id": "rs_08", "brand": "Adidas", "model": "Ultraboost Light", "price": 190, "type": "premium lifestyle runner", "best_for": "casual + running, style"},
    ],
}

PERSONAS_PER_CAT = [
    {"name": "Budget Student", "budget": "low", "knowledge": "low", "context": "buying first pair, tight budget"},
    {"name": "Commuter Professional", "budget": "medium", "knowledge": "medium", "context": "daily commute, wants comfort"},
    {"name": "Tech Enthusiast", "budget": "high", "knowledge": "high", "context": "wants latest features, reads reviews"},
    {"name": "Parent Buying Gift", "budget": "medium", "knowledge": "low", "context": "buying for teenager, unsure what's good"},
    {"name": "Remote Worker", "budget": "medium-high", "knowledge": "medium", "context": "long hours at desk, values comfort"},
    {"name": "Casual User", "budget": "low-medium", "knowledge": "low", "context": "just needs something that works"},
    {"name": "Fitness Enthusiast", "budget": "medium", "knowledge": "medium", "context": "active lifestyle, needs durability"},
    {"name": "Small Business Owner", "budget": "medium", "knowledge": "low", "context": "equipping office, bulk decision"},
    {"name": "Retiree", "budget": "medium-high", "knowledge": "low", "context": "replacing old device, values simplicity"},
    {"name": "Content Creator", "budget": "high", "knowledge": "high", "context": "professional use, needs quality"},
]

QUAL_HISTORY_EXAMPLE = (
    "Historical background (internal only, do not cite numbers): "
    "This product has been frequently chosen by budget-conscious buyers in this category. "
    "Post-purchase feedback is generally positive, with occasional notes about build quality. "
    "Among similar buyers, this product is often considered a reliable choice."
)

REGIMES = {
    "generic": {
        "instruction": (
            "You are a helpful product recommender. Recommend the single best product for this consumer "
            "from the catalog. Be realistic, concise, and helpful. Focus on fit with the consumer's needs."
        ),
    },
    "consumer_centric": {
        "instruction": (
            "You are a balanced, consumer-first product recommender. Prioritize fit, budget, tradeoffs, "
            "and post-purchase satisfaction. Be transparent about limitations. Do not oversell."
        ),
    },
    "history_aware": {
        "instruction": (
            "You are a product recommender with access to internal historical purchase summaries. "
            "Treat them as background evidence only. Do not reveal, quote, or approximate any "
            "conversion rates, satisfaction rates, percentages, rankings, sample sizes, scores, "
            "or raw historical numbers. You may only refer qualitatively to historical patterns using "
            'phrases like "popular among similar buyers," "historically reliable," '
            '"often chosen for this use case," or "mixed feedback among heavy users." '
            "Write a natural, consumer-facing recommendation."
        ),
    },
}

EXEMPLAR_OUTPUT_SCHEMA = """\
Return ONLY valid JSON matching this schema:
{
  "selected_product_id": "...",
  "recommendation_text": "A 2-4 sentence consumer-facing recommendation",
  "why_it_fits": "1 sentence on why this product fits the consumer",
  "tradeoff_note": "1 sentence on the main tradeoff or limitation",
  "history_used_qualitatively": true/false,
  "forbidden_numeric_history_leakage": false
}"""


def build_exemplar_prompt(category: str, persona: dict, regime: str,
                          catalog: list[dict], history: str | None = None) -> str:
    parts = [REGIMES[regime]["instruction"]]

    parts.append(f"\nCategory: {category}")
    parts.append(f"\nConsumer: {persona['name']} — {persona['context']}")
    parts.append(f"Budget: {persona['budget']}, Technical knowledge: {persona['knowledge']}")

    cat_text = "\n".join(
        f"- {p['id']}: {p['brand']} {p['model']} (${p['price']}) — {p['type']}, best for: {p['best_for']}"
        for p in catalog
    )
    parts.append(f"\nProduct catalog:\n{cat_text}")

    if regime == "history_aware" and history:
        parts.append(f"\n{history}")

    parts.append(f"\n{EXEMPLAR_OUTPUT_SCHEMA}")
    return "\n".join(parts)


def main():
    existing = load_jsonl(CACHE_PATH)
    done_ids = {r["example_id"] for r in existing}
    print(f"Cache has {len(done_ids)} existing exemplars")

    total = len(CATEGORIES) * len(PERSONAS_PER_CAT) * len(REGIMES)
    remaining = total - len(done_ids)
    print(f"Total needed: {total}, remaining: {remaining}")

    if remaining == 0:
        print("All exemplars already generated.")
        select_few_shot()
        return

    count = 0
    for cat in CATEGORIES:
        catalog = MINI_CATALOGS[cat]
        for pi, persona in enumerate(PERSONAS_PER_CAT):
            for regime in REGIMES:
                eid = f"{cat}_{pi:02d}_{regime}"
                if eid in done_ids:
                    continue

                history = QUAL_HISTORY_EXAMPLE if regime == "history_aware" else None
                prompt = build_exemplar_prompt(cat, persona, regime, catalog, history)

                parsed, raw = gpt_json_call(prompt)
                count += 1

                row = {
                    "example_id": eid,
                    "category": cat,
                    "persona_index": pi,
                    "persona_name": persona["name"],
                    "regime": regime,
                    "prompt": prompt,
                    "response_text": raw["output_text"],
                    "parsed": parsed,
                    "model": raw["model"],
                    "usage": raw["usage"],
                    "timestamp": raw["timestamp"],
                }
                append_jsonl(CACHE_PATH, row)

                if count % 10 == 0 or count == remaining:
                    print(f"  [{count}/{remaining}] {eid}")

    print(f"\nGenerated {count} exemplars. Total cached: {len(done_ids) + count}")
    select_few_shot()


def select_few_shot():
    """Select high-quality exemplars for few-shot prompts."""
    exemplars = load_jsonl(CACHE_PATH)
    if not exemplars:
        print("No exemplars to select from")
        return

    selected = {"generic": [], "consumer_centric": [], "history_aware": []}

    for regime in selected:
        regime_exs = [e for e in exemplars if e["regime"] == regime
                      and not e.get("parsed", {}).get("_parse_failed")]

        final_cats = [e for e in regime_exs if e["category"] in ("headphones", "phone_chargers")]
        pool = final_cats if len(final_cats) >= 4 else regime_exs

        for ex in pool[:2]:
            selected[regime].append({
                "example_id": ex["example_id"],
                "category": ex["category"],
                "regime": regime,
                "recommendation_text": ex["parsed"].get("recommendation_text", ""),
                "tradeoff_note": ex["parsed"].get("tradeoff_note", ""),
            })

    out_path = EXEMPLAR_DIR / "selected_few_shot_examples.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Selected few-shot examples → {out_path}")
    for regime, exs in selected.items():
        print(f"  {regime}: {len(exs)} examples")


if __name__ == "__main__":
    main()
