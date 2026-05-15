"""
Generate product catalogs for the Unbundling LLM Recommenders simulation.

Creates realistic product catalogs for 3 categories:
- Phone chargers (low complexity, 8 products)
- Headphones (medium complexity, 10 products)
- Laptops (high complexity, 10 products)

Each product has: brand, price, quality, attributes, review summary,
use-case fit scores, weakness, and experimental role indicators.
"""

import json
import pathlib

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalogs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_phone_charger_catalog():
    return {
        "category": "phone_charger",
        "category_display_name": "Phone Chargers",
        "complexity": "low",
        "focal_brand": "Anker",
        "sponsored_product": "charger_voltix_65w",
        "use_cases": ["travel", "office_desk", "bedside", "fast_phone_only", "multi_device"],
        "products": [
            {
                "product_id": "charger_anker_nano_67w",
                "brand_name": "Anker",
                "brand_status": "incumbent",
                "price": 35.99,
                "quality_score": 88,
                "attributes": {
                    "wattage": 67,
                    "ports": "2 USB-C, 1 USB-A",
                    "form_factor": "compact_foldable",
                    "weight_grams": 113,
                    "gan_technology": True,
                    "pps_support": True,
                    "cable_included": False,
                },
                "review_summary": "Compact GaN charger widely praised for fast charging and pocket-friendly size. Some users note it gets warm under full load and the USB-A port is slower than expected.",
                "use_case_fit": {
                    "travel": 0.90,
                    "office_desk": 0.80,
                    "bedside": 0.55,
                    "fast_phone_only": 0.75,
                    "multi_device": 0.85,
                },
                "weakness": "Gets warm under sustained full-power output; no cable included.",
                "margin_tier": "medium",
                "focal_brand": True,
            },
            {
                "product_id": "charger_anker_prime_100w",
                "brand_name": "Anker",
                "brand_status": "incumbent",
                "price": 59.99,
                "quality_score": 91,
                "attributes": {
                    "wattage": 100,
                    "ports": "2 USB-C, 1 USB-A",
                    "form_factor": "compact_brick",
                    "weight_grams": 177,
                    "gan_technology": True,
                    "pps_support": True,
                    "cable_included": False,
                },
                "review_summary": "High-powered GaN charger that can charge a laptop and phone simultaneously. Build quality is excellent but it is noticeably heavier and pricier than 65W alternatives.",
                "use_case_fit": {
                    "travel": 0.60,
                    "office_desk": 0.90,
                    "bedside": 0.30,
                    "fast_phone_only": 0.40,
                    "multi_device": 0.90,
                },
                "weakness": "Premium price; heavier than most pocket chargers; overkill for phone-only charging.",
                "margin_tier": "high",
                "focal_brand": True,
            },
            {
                "product_id": "charger_apple_35w_duo",
                "brand_name": "Apple",
                "brand_status": "incumbent",
                "price": 59.00,
                "quality_score": 78,
                "attributes": {
                    "wattage": 35,
                    "ports": "2 USB-C",
                    "form_factor": "compact_foldable",
                    "weight_grams": 105,
                    "gan_technology": False,
                    "pps_support": False,
                    "cable_included": False,
                },
                "review_summary": "Sleek dual-port charger from Apple with signature minimalist design. Charges Apple devices well but the 35W total output is low for the price, and it lacks PPS for fast-charging many Android phones.",
                "use_case_fit": {
                    "travel": 0.85,
                    "office_desk": 0.60,
                    "bedside": 0.70,
                    "fast_phone_only": 0.50,
                    "multi_device": 0.55,
                },
                "weakness": "Only 35W total; expensive for the wattage; no PPS support for non-Apple devices.",
                "margin_tier": "high",
                "focal_brand": False,
            },
            {
                "product_id": "charger_samsung_45w",
                "brand_name": "Samsung",
                "brand_status": "incumbent",
                "price": 29.99,
                "quality_score": 82,
                "attributes": {
                    "wattage": 45,
                    "ports": "1 USB-C",
                    "form_factor": "compact_brick",
                    "weight_grams": 95,
                    "gan_technology": False,
                    "pps_support": True,
                    "cable_included": True,
                },
                "review_summary": "Official Samsung charger with PPS support for full-speed Galaxy charging. Simple and reliable but has only a single port and the design feels dated compared to GaN competitors.",
                "use_case_fit": {
                    "travel": 0.65,
                    "office_desk": 0.55,
                    "bedside": 0.70,
                    "fast_phone_only": 0.90,
                    "multi_device": 0.20,
                },
                "weakness": "Single port only; no GaN — less efficient and slightly bulkier than rivals at this wattage.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "charger_baseus_65w",
                "brand_name": "Baseus",
                "brand_status": "entrant",
                "price": 25.99,
                "quality_score": 79,
                "attributes": {
                    "wattage": 65,
                    "ports": "2 USB-C, 1 USB-A",
                    "form_factor": "compact_foldable",
                    "weight_grams": 120,
                    "gan_technology": True,
                    "pps_support": True,
                    "cable_included": False,
                },
                "review_summary": "Aggressive value play with 65W GaN output at a budget price. Most users are satisfied but some report inconsistent fast-charging handshake with certain laptops and a faint coil whine.",
                "use_case_fit": {
                    "travel": 0.85,
                    "office_desk": 0.75,
                    "bedside": 0.55,
                    "fast_phone_only": 0.70,
                    "multi_device": 0.80,
                },
                "weakness": "Occasional charging negotiation issues with some laptops; faint coil whine reported by some users.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "charger_voltix_65w",
                "brand_name": "Voltix",
                "brand_status": "entrant",
                "price": 22.99,
                "quality_score": 72,
                "attributes": {
                    "wattage": 65,
                    "ports": "1 USB-C, 1 USB-A",
                    "form_factor": "standard_brick",
                    "weight_grams": 140,
                    "gan_technology": True,
                    "pps_support": True,
                    "cable_included": True,
                },
                "review_summary": "Budget GaN charger that delivers good wattage at the lowest price in its class. Cable included is a plus, but the build quality feels cheap and it runs hotter than name-brand alternatives.",
                "use_case_fit": {
                    "travel": 0.60,
                    "office_desk": 0.65,
                    "bedside": 0.50,
                    "fast_phone_only": 0.75,
                    "multi_device": 0.50,
                },
                "weakness": "Plasticky build; runs hot; lesser-known brand with limited warranty support.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "charger_ugreen_100w",
                "brand_name": "UGREEN",
                "brand_status": "entrant",
                "price": 44.99,
                "quality_score": 85,
                "attributes": {
                    "wattage": 100,
                    "ports": "3 USB-C, 1 USB-A",
                    "form_factor": "desktop_brick",
                    "weight_grams": 240,
                    "gan_technology": True,
                    "pps_support": True,
                    "cable_included": False,
                },
                "review_summary": "4-port 100W desktop charger that undercuts Anker on price while matching specs. Excellent for powering multiple devices but too large for travel and the brand is less recognized.",
                "use_case_fit": {
                    "travel": 0.25,
                    "office_desk": 0.95,
                    "bedside": 0.30,
                    "fast_phone_only": 0.45,
                    "multi_device": 0.95,
                },
                "weakness": "Too large and heavy for travel; lesser-known brand; no cable included.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "charger_belkin_25w",
                "brand_name": "Belkin",
                "brand_status": "incumbent",
                "price": 19.99,
                "quality_score": 75,
                "attributes": {
                    "wattage": 25,
                    "ports": "1 USB-C",
                    "form_factor": "ultra_compact",
                    "weight_grams": 58,
                    "gan_technology": False,
                    "pps_support": True,
                    "cable_included": False,
                },
                "review_summary": "The smallest and lightest charger in this roundup, ideal for bedside or tossing in a bag. Limited to 25W and a single port, so it cannot charge laptops or multiple devices.",
                "use_case_fit": {
                    "travel": 0.75,
                    "office_desk": 0.35,
                    "bedside": 0.95,
                    "fast_phone_only": 0.85,
                    "multi_device": 0.10,
                },
                "weakness": "Only 25W — too slow for laptops; single port; no GaN technology.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
        ],
    }


def build_headphones_catalog():
    return {
        "category": "headphones",
        "category_display_name": "Over-Ear Headphones",
        "complexity": "medium",
        "focal_brand": "Sony",
        "sponsored_product": "headphones_edifier_stax",
        "use_cases": [
            "commuting",
            "office_work",
            "gym_running",
            "audiophile_home",
            "gaming",
            "budget_casual",
        ],
        "products": [
            {
                "product_id": "headphones_sony_wh1000xm5",
                "brand_name": "Sony",
                "brand_status": "incumbent",
                "price": 348.00,
                "quality_score": 90,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "excellent",
                    "battery_life_hours": 30,
                    "weight_grams": 250,
                    "driver_size_mm": 30,
                    "codec_support": ["LDAC", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": False,
                },
                "review_summary": "Industry-leading noise cancellation with a comfortable lightweight build. Sound quality is excellent for a wireless headphone. The non-foldable design is less travel-friendly than its predecessor, and call microphone quality is average.",
                "use_case_fit": {
                    "commuting": 0.90,
                    "office_work": 0.90,
                    "gym_running": 0.25,
                    "audiophile_home": 0.70,
                    "gaming": 0.40,
                    "budget_casual": 0.20,
                },
                "weakness": "Non-foldable design; call microphone is average; premium price.",
                "margin_tier": "high",
                "focal_brand": True,
            },
            {
                "product_id": "headphones_sony_ult_wear",
                "brand_name": "Sony",
                "brand_status": "incumbent",
                "price": 198.00,
                "quality_score": 80,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "good",
                    "battery_life_hours": 50,
                    "weight_grams": 255,
                    "driver_size_mm": 40,
                    "codec_support": ["LDAC", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Bass-heavy Sony headphone with an impressive 50-hour battery life at a mid-range price. Good noise cancellation but not class-leading. The heavy bass emphasis may not suit all listeners and the build feels less premium than the XM5.",
                "use_case_fit": {
                    "commuting": 0.80,
                    "office_work": 0.70,
                    "gym_running": 0.35,
                    "audiophile_home": 0.45,
                    "gaming": 0.50,
                    "budget_casual": 0.65,
                },
                "weakness": "Bass-heavy tuning not for everyone; build quality a step below XM5; ANC not class-leading.",
                "margin_tier": "medium",
                "focal_brand": True,
            },
            {
                "product_id": "headphones_bose_qc_ultra",
                "brand_name": "Bose",
                "brand_status": "incumbent",
                "price": 379.00,
                "quality_score": 89,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "excellent",
                    "battery_life_hours": 24,
                    "weight_grams": 250,
                    "driver_size_mm": 35,
                    "codec_support": ["aptX Adaptive", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Arguably the best noise cancellation available, with immersive spatial audio and luxurious comfort. Battery life is shorter than Sony's and the price is the highest in this roundup. Sound is slightly warm and smooth rather than analytical.",
                "use_case_fit": {
                    "commuting": 0.95,
                    "office_work": 0.90,
                    "gym_running": 0.20,
                    "audiophile_home": 0.65,
                    "gaming": 0.35,
                    "budget_casual": 0.15,
                },
                "weakness": "Highest price in roundup; 24-hour battery shorter than competitors; warm sound may lack detail for critical listening.",
                "margin_tier": "high",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_apple_airpods_max",
                "brand_name": "Apple",
                "brand_status": "incumbent",
                "price": 449.00,
                "quality_score": 86,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "excellent",
                    "battery_life_hours": 20,
                    "weight_grams": 384,
                    "driver_size_mm": 40,
                    "codec_support": ["AAC"],
                    "multipoint": False,
                    "foldable": False,
                },
                "review_summary": "Premium build quality with aluminum ear cups and exceptional integration with Apple devices. Noise cancellation and spatial audio are top-tier. However, they are heavy, expensive, lack high-res codec support, and work poorly outside the Apple ecosystem.",
                "use_case_fit": {
                    "commuting": 0.70,
                    "office_work": 0.75,
                    "gym_running": 0.05,
                    "audiophile_home": 0.60,
                    "gaming": 0.30,
                    "budget_casual": 0.05,
                },
                "weakness": "Heaviest option at 384g; AAC only — no LDAC or aptX; poor experience on Android; most expensive.",
                "margin_tier": "high",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_sennheiser_momentum4",
                "brand_name": "Sennheiser",
                "brand_status": "incumbent",
                "price": 299.95,
                "quality_score": 87,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "very_good",
                    "battery_life_hours": 60,
                    "weight_grams": 293,
                    "driver_size_mm": 42,
                    "codec_support": ["aptX", "aptX Adaptive", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Audiophile-leaning wireless headphone with the best sound quality in this roundup and a remarkable 60-hour battery life. Noise cancellation is strong but slightly below Sony and Bose. The touch controls have a learning curve.",
                "use_case_fit": {
                    "commuting": 0.80,
                    "office_work": 0.85,
                    "gym_running": 0.15,
                    "audiophile_home": 0.90,
                    "gaming": 0.45,
                    "budget_casual": 0.25,
                },
                "weakness": "Touch controls can be finicky; ANC slightly behind Sony/Bose; not the most comfortable for very long sessions.",
                "margin_tier": "high",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_jbl_tune770nc",
                "brand_name": "JBL",
                "brand_status": "incumbent",
                "price": 79.95,
                "quality_score": 72,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "basic",
                    "battery_life_hours": 44,
                    "weight_grams": 225,
                    "driver_size_mm": 40,
                    "codec_support": ["AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Best value ANC headphone on the market with solid battery life and the punchy JBL sound signature. Noise cancellation is present but noticeably weaker than premium options. Build quality is plastic-heavy.",
                "use_case_fit": {
                    "commuting": 0.65,
                    "office_work": 0.60,
                    "gym_running": 0.45,
                    "audiophile_home": 0.30,
                    "gaming": 0.50,
                    "budget_casual": 0.90,
                },
                "weakness": "ANC is basic; all-plastic build; no high-res codec; sound lacks detail at higher volumes.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_edifier_stax",
                "brand_name": "Edifier",
                "brand_status": "entrant",
                "price": 129.99,
                "quality_score": 78,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "good",
                    "battery_life_hours": 38,
                    "weight_grams": 267,
                    "driver_size_mm": 40,
                    "codec_support": ["LDAC", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "A mid-range surprise from a brand better known for speakers. LDAC support and good noise cancellation at this price point are unusual. Sound is balanced but the brand lacks the cachet and after-sales support of larger competitors.",
                "use_case_fit": {
                    "commuting": 0.75,
                    "office_work": 0.70,
                    "gym_running": 0.30,
                    "audiophile_home": 0.55,
                    "gaming": 0.45,
                    "budget_casual": 0.75,
                },
                "weakness": "Lesser-known brand in headphones; after-sales support uncertain; comfort padding could be thicker.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_soundcore_space_q45",
                "brand_name": "Soundcore",
                "brand_status": "entrant",
                "price": 99.99,
                "quality_score": 76,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "good",
                    "battery_life_hours": 50,
                    "weight_grams": 295,
                    "driver_size_mm": 40,
                    "codec_support": ["LDAC", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Punches above its price with LDAC support and 50-hour battery life. ANC is surprisingly effective for a sub-$100 headphone. However, the headband pressure is tight for larger heads and the companion app is buggy.",
                "use_case_fit": {
                    "commuting": 0.75,
                    "office_work": 0.65,
                    "gym_running": 0.35,
                    "audiophile_home": 0.40,
                    "gaming": 0.50,
                    "budget_casual": 0.85,
                },
                "weakness": "Tight clamping force; buggy companion app; plasticky build; Soundcore brand less trusted than Sony/Bose.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_hifiman_deva_se",
                "brand_name": "HiFiMAN",
                "brand_status": "entrant",
                "price": 109.00,
                "quality_score": 81,
                "attributes": {
                    "type": "over-ear wired/wireless",
                    "noise_cancellation": "none",
                    "battery_life_hours": 80,
                    "weight_grams": 360,
                    "driver_size_mm": 78,
                    "codec_support": ["LDAC", "AAC", "SBC"],
                    "multipoint": False,
                    "foldable": False,
                },
                "review_summary": "Open-back planar magnetic headphone with wireless capability — a rare combination. Sound quality is exceptional for the price with a wide soundstage. Zero noise isolation makes it unsuitable for commuting or noisy environments, and it is heavy.",
                "use_case_fit": {
                    "commuting": 0.05,
                    "office_work": 0.30,
                    "gym_running": 0.0,
                    "audiophile_home": 0.95,
                    "gaming": 0.70,
                    "budget_casual": 0.20,
                },
                "weakness": "Open-back design leaks sound and offers zero isolation; heavy at 360g; not portable.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "headphones_marshall_monitor3",
                "brand_name": "Marshall",
                "brand_status": "incumbent",
                "price": 249.99,
                "quality_score": 77,
                "attributes": {
                    "type": "over-ear wireless",
                    "noise_cancellation": "good",
                    "battery_life_hours": 70,
                    "weight_grams": 320,
                    "driver_size_mm": 40,
                    "codec_support": ["LC3plus", "AAC", "SBC"],
                    "multipoint": True,
                    "foldable": True,
                },
                "review_summary": "Iconic rock-and-roll aesthetic with a 70-hour battery and good ANC. Sound is warm and engaging for rock and pop. The vintage-inspired design is polarizing, and the sound signature is too bass-heavy for genres that demand clarity.",
                "use_case_fit": {
                    "commuting": 0.70,
                    "office_work": 0.60,
                    "gym_running": 0.20,
                    "audiophile_home": 0.50,
                    "gaming": 0.40,
                    "budget_casual": 0.55,
                },
                "weakness": "Polarizing design; bass-heavy tuning unsuitable for classical or acoustic genres; heavier than average.",
                "margin_tier": "high",
                "focal_brand": False,
            },
        ],
    }


def build_laptop_catalog():
    return {
        "category": "laptop",
        "category_display_name": "Laptops for Professionals",
        "complexity": "high",
        "focal_brand": "Apple",
        "sponsored_product": "laptop_framework_16",
        "use_cases": [
            "data_analysis",
            "software_development",
            "creative_media",
            "business_travel",
            "student_general",
            "budget_office",
        ],
        "products": [
            {
                "product_id": "laptop_macbook_pro_m4",
                "brand_name": "Apple",
                "brand_status": "incumbent",
                "price": 1999.00,
                "quality_score": 93,
                "attributes": {
                    "processor": "Apple M4 Pro",
                    "ram_gb": 24,
                    "storage_gb": 512,
                    "display_inches": 14.2,
                    "display_type": "mini-LED XDR",
                    "battery_hours": 17,
                    "weight_kg": 1.55,
                    "gpu": "integrated 20-core",
                    "os": "macOS",
                    "ports": "3 Thunderbolt 4, HDMI, SD, MagSafe",
                },
                "review_summary": "Best-in-class performance-per-watt with exceptional battery life and a stunning display. The M4 Pro handles data analysis, creative work, and development with ease. macOS limits some enterprise software compatibility, RAM is not user-upgradeable, and the price is high.",
                "use_case_fit": {
                    "data_analysis": 0.85,
                    "software_development": 0.80,
                    "creative_media": 0.95,
                    "business_travel": 0.75,
                    "student_general": 0.45,
                    "budget_office": 0.10,
                },
                "weakness": "macOS incompatible with some enterprise/Windows-only tools; 512GB base storage fills fast; non-upgradeable RAM; expensive.",
                "margin_tier": "high",
                "focal_brand": True,
            },
            {
                "product_id": "laptop_macbook_air_m4",
                "brand_name": "Apple",
                "brand_status": "incumbent",
                "price": 1099.00,
                "quality_score": 87,
                "attributes": {
                    "processor": "Apple M4",
                    "ram_gb": 16,
                    "storage_gb": 256,
                    "display_inches": 13.6,
                    "display_type": "Liquid Retina",
                    "battery_hours": 18,
                    "weight_kg": 1.24,
                    "gpu": "integrated 10-core",
                    "os": "macOS",
                    "ports": "2 Thunderbolt 4, MagSafe",
                },
                "review_summary": "Ultralight and fanless with all-day battery life. Handles everyday work and light creative tasks admirably. The 256GB base storage is cramped, the display lacks HDR, and the M4 chip struggles with sustained heavy workloads compared to the M4 Pro.",
                "use_case_fit": {
                    "data_analysis": 0.55,
                    "software_development": 0.60,
                    "creative_media": 0.50,
                    "business_travel": 0.85,
                    "student_general": 0.80,
                    "budget_office": 0.45,
                },
                "weakness": "Only 256GB base storage; limited port selection; not powerful enough for heavy data or video work.",
                "margin_tier": "high",
                "focal_brand": True,
            },
            {
                "product_id": "laptop_thinkpad_x1_carbon",
                "brand_name": "Lenovo",
                "brand_status": "incumbent",
                "price": 1549.00,
                "quality_score": 88,
                "attributes": {
                    "processor": "Intel Core Ultra 7 155H",
                    "ram_gb": 32,
                    "storage_gb": 512,
                    "display_inches": 14.0,
                    "display_type": "OLED 2.8K",
                    "battery_hours": 12,
                    "weight_kg": 1.09,
                    "gpu": "Intel Arc integrated",
                    "os": "Windows 11 Pro",
                    "ports": "2 Thunderbolt 4, 2 USB-A, HDMI",
                },
                "review_summary": "The gold standard for business ultrabooks: superb keyboard, gorgeous OLED display, and the lightest laptop in this roundup at 1.09 kg. Windows compatibility is universal. Battery life is shorter than Apple competitors and the Intel GPU is weak for creative workloads.",
                "use_case_fit": {
                    "data_analysis": 0.80,
                    "software_development": 0.85,
                    "creative_media": 0.55,
                    "business_travel": 0.90,
                    "student_general": 0.70,
                    "budget_office": 0.40,
                },
                "weakness": "Battery life shorter than MacBooks; integrated Intel GPU is weak for GPU-heavy tasks; OLED can have burn-in over years.",
                "margin_tier": "high",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_dell_xps_14",
                "brand_name": "Dell",
                "brand_status": "incumbent",
                "price": 1399.00,
                "quality_score": 82,
                "attributes": {
                    "processor": "Intel Core Ultra 7 155H",
                    "ram_gb": 16,
                    "storage_gb": 512,
                    "display_inches": 14.5,
                    "display_type": "OLED 3.2K touch",
                    "battery_hours": 10,
                    "weight_kg": 1.69,
                    "gpu": "Intel Arc integrated",
                    "os": "Windows 11 Home",
                    "ports": "2 Thunderbolt 4, 1 USB-C",
                },
                "review_summary": "Stunning OLED touchscreen in a sleek aluminum chassis. The keyboard layout is polarizing — function keys are replaced by a capacitive touch bar. Battery life is the worst in this roundup for a non-gaming laptop, and the lack of USB-A requires dongles.",
                "use_case_fit": {
                    "data_analysis": 0.65,
                    "software_development": 0.60,
                    "creative_media": 0.70,
                    "business_travel": 0.60,
                    "student_general": 0.65,
                    "budget_office": 0.30,
                },
                "weakness": "Poor battery life (~10 hrs); controversial touch function row; no USB-A; heavier than competing ultrabooks.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_hp_elitebook_845",
                "brand_name": "HP",
                "brand_status": "incumbent",
                "price": 1249.00,
                "quality_score": 83,
                "attributes": {
                    "processor": "AMD Ryzen 7 PRO 7840U",
                    "ram_gb": 32,
                    "storage_gb": 512,
                    "display_inches": 14.0,
                    "display_type": "IPS 1920x1200",
                    "battery_hours": 14,
                    "weight_kg": 1.40,
                    "gpu": "AMD Radeon 780M integrated",
                    "os": "Windows 11 Pro",
                    "ports": "2 USB-C, 2 USB-A, HDMI",
                },
                "review_summary": "Enterprise workhorse with AMD Ryzen Pro security features, 32GB RAM, and a full port selection. The Radeon integrated GPU is better than Intel's for light creative work. The IPS display is adequate but dull compared to OLED options, and the design is conservative.",
                "use_case_fit": {
                    "data_analysis": 0.80,
                    "software_development": 0.80,
                    "creative_media": 0.50,
                    "business_travel": 0.80,
                    "student_general": 0.65,
                    "budget_office": 0.55,
                },
                "weakness": "Display is mediocre compared to OLED competitors; design is bland; heavier than X1 Carbon.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_framework_16",
                "brand_name": "Framework",
                "brand_status": "entrant",
                "price": 1399.00,
                "quality_score": 80,
                "attributes": {
                    "processor": "AMD Ryzen 7 7840HS",
                    "ram_gb": 32,
                    "storage_gb": 1000,
                    "display_inches": 16.0,
                    "display_type": "IPS 2560x1600",
                    "battery_hours": 10,
                    "weight_kg": 2.10,
                    "gpu": "AMD Radeon RX 7700S (optional module)",
                    "os": "Windows 11 / Linux",
                    "ports": "6 modular expansion card bays",
                },
                "review_summary": "The most repairable and upgradeable laptop ever made. Every component — RAM, storage, battery, GPU, ports, keyboard — is user-replaceable. Appeals strongly to tech enthusiasts and sustainability-minded buyers. Build quality is good but not as polished as ThinkPad or MacBook, and the modular design adds weight.",
                "use_case_fit": {
                    "data_analysis": 0.75,
                    "software_development": 0.85,
                    "creative_media": 0.65,
                    "business_travel": 0.45,
                    "student_general": 0.60,
                    "budget_office": 0.35,
                },
                "weakness": "Heavy at 2.1 kg; modular expansion bays add bulk; battery life only average; brand is young with unproven long-term support.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_asus_zenbook_14",
                "brand_name": "ASUS",
                "brand_status": "incumbent",
                "price": 849.00,
                "quality_score": 79,
                "attributes": {
                    "processor": "Intel Core Ultra 5 125H",
                    "ram_gb": 16,
                    "storage_gb": 512,
                    "display_inches": 14.0,
                    "display_type": "OLED 1920x1200",
                    "battery_hours": 13,
                    "weight_kg": 1.28,
                    "gpu": "Intel Arc integrated",
                    "os": "Windows 11 Home",
                    "ports": "1 Thunderbolt 4, 1 USB-C, 1 USB-A, HDMI",
                },
                "review_summary": "Remarkable value: OLED display, lightweight design, and decent performance under $900. The best option for budget-conscious buyers who want a quality screen. The Intel Core Ultra 5 is less powerful than competitors' Core Ultra 7 or Ryzen 7, and the speakers are tinny.",
                "use_case_fit": {
                    "data_analysis": 0.55,
                    "software_development": 0.60,
                    "creative_media": 0.55,
                    "business_travel": 0.75,
                    "student_general": 0.90,
                    "budget_office": 0.80,
                },
                "weakness": "Less powerful Core Ultra 5 processor; weak speakers; only 16GB RAM with no upgrade path.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_acer_swift_go_14",
                "brand_name": "Acer",
                "brand_status": "incumbent",
                "price": 699.00,
                "quality_score": 73,
                "attributes": {
                    "processor": "AMD Ryzen 5 7530U",
                    "ram_gb": 16,
                    "storage_gb": 512,
                    "display_inches": 14.0,
                    "display_type": "IPS 1920x1200",
                    "battery_hours": 11,
                    "weight_kg": 1.32,
                    "gpu": "AMD Radeon integrated",
                    "os": "Windows 11 Home",
                    "ports": "1 USB-C, 2 USB-A, HDMI",
                },
                "review_summary": "The cheapest laptop in this roundup that still delivers a usable professional experience. Good for documents, spreadsheets, and web browsing. The IPS display is dim, the chassis creaks under pressure, and performance drops noticeably under sustained heavy workloads.",
                "use_case_fit": {
                    "data_analysis": 0.35,
                    "software_development": 0.40,
                    "creative_media": 0.25,
                    "business_travel": 0.60,
                    "student_general": 0.75,
                    "budget_office": 0.90,
                },
                "weakness": "Dim display; flexes under pressure; performance drops under sustained load; no Thunderbolt.",
                "margin_tier": "low",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_lg_gram_14",
                "brand_name": "LG",
                "brand_status": "entrant",
                "price": 1299.00,
                "quality_score": 81,
                "attributes": {
                    "processor": "Intel Core Ultra 7 155H",
                    "ram_gb": 16,
                    "storage_gb": 512,
                    "display_inches": 14.0,
                    "display_type": "OLED 2.8K",
                    "battery_hours": 15,
                    "weight_kg": 0.99,
                    "gpu": "Intel Arc integrated",
                    "os": "Windows 11 Home",
                    "ports": "2 Thunderbolt 4, 1 USB-A, HDMI, microSD",
                },
                "review_summary": "Incredibly light at under 1 kg with a beautiful OLED display and solid battery life. A strong pick for frequent flyers who want a premium screen. The chassis is thin and flexes slightly, and the keyboard is shallow compared to ThinkPad. Only 16GB RAM limits multitasking.",
                "use_case_fit": {
                    "data_analysis": 0.60,
                    "software_development": 0.65,
                    "creative_media": 0.55,
                    "business_travel": 0.95,
                    "student_general": 0.70,
                    "budget_office": 0.35,
                },
                "weakness": "Chassis flexes under pressure; shallow keyboard travel; only 16GB RAM; LG less proven in laptops than Lenovo/Apple.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
            {
                "product_id": "laptop_system76_lemur",
                "brand_name": "System76",
                "brand_status": "entrant",
                "price": 1149.00,
                "quality_score": 76,
                "attributes": {
                    "processor": "Intel Core Ultra 7 155H",
                    "ram_gb": 32,
                    "storage_gb": 500,
                    "display_inches": 14.0,
                    "display_type": "IPS 1920x1200",
                    "battery_hours": 10,
                    "weight_kg": 1.62,
                    "gpu": "Intel Arc integrated",
                    "os": "Pop!_OS Linux (Ubuntu-based)",
                    "ports": "1 Thunderbolt 4, 1 USB-C, 2 USB-A, HDMI",
                },
                "review_summary": "Purpose-built Linux laptop with excellent open-source software support and 32GB RAM. Ideal for developers and data scientists who prefer a native Linux experience. The IPS display is average, build quality is a step below ThinkPad, and the brand is niche with limited service centers.",
                "use_case_fit": {
                    "data_analysis": 0.80,
                    "software_development": 0.90,
                    "creative_media": 0.35,
                    "business_travel": 0.50,
                    "student_general": 0.40,
                    "budget_office": 0.30,
                },
                "weakness": "Linux-only can limit software compatibility; average display; niche brand with limited physical support; plain design.",
                "margin_tier": "medium",
                "focal_brand": False,
            },
        ],
    }


def validate_catalog(catalog: dict) -> list[str]:
    """Check catalog quality and return a list of issues."""
    issues = []
    category = catalog["category"]
    products = catalog["products"]
    use_cases = catalog["use_cases"]

    # --- Counts ---
    n = len(products)
    if n < 8:
        issues.append(f"[{category}] Only {n} products (need >= 8)")

    n_incumbent = sum(1 for p in products if p["brand_status"] == "incumbent")
    n_entrant = sum(1 for p in products if p["brand_status"] == "entrant")
    if n_entrant < 2:
        issues.append(f"[{category}] Only {n_entrant} entrants (need >= 2)")
    if n_incumbent < 3:
        issues.append(f"[{category}] Only {n_incumbent} incumbents (need >= 3)")

    # --- Price variation ---
    prices = [p["price"] for p in products]
    price_range_ratio = max(prices) / min(prices)
    if price_range_ratio < 2.0:
        issues.append(
            f"[{category}] Price range ratio is only {price_range_ratio:.1f}x "
            f"(min={min(prices)}, max={max(prices)}). Need >= 2x for meaningful variation."
        )

    # --- Quality variation ---
    quality_scores = [p["quality_score"] for p in products]
    q_range = max(quality_scores) - min(quality_scores)
    if q_range < 15:
        issues.append(
            f"[{category}] Quality range is only {q_range} points "
            f"(min={min(quality_scores)}, max={max(quality_scores)}). Need >= 15."
        )

    # --- Dominance check: no product should dominate on ALL use cases ---
    for p in products:
        fits = [p["use_case_fit"].get(uc, 0) for uc in use_cases]
        if all(f >= 0.80 for f in fits):
            issues.append(
                f"[{category}] Product {p['product_id']} dominates ALL use cases "
                f"(all fit >= 0.80). Every product needs a weakness use case."
            )

    # --- Each use case should have a different top product ---
    top_per_uc = {}
    for uc in use_cases:
        best = max(products, key=lambda p: p["use_case_fit"].get(uc, 0))
        top_per_uc[uc] = best["product_id"]
    n_distinct_tops = len(set(top_per_uc.values()))
    if n_distinct_tops < min(3, len(use_cases)):
        issues.append(
            f"[{category}] Only {n_distinct_tops} distinct top products across "
            f"{len(use_cases)} use cases. Need >= 3 for retrieval variation."
        )

    # --- Focal brand and sponsored product exist ---
    focal_brand = catalog.get("focal_brand")
    focal_products = [p for p in products if p.get("focal_brand")]
    if not focal_products:
        issues.append(f"[{category}] No product marked as focal_brand=True")

    sponsored_id = catalog.get("sponsored_product")
    if sponsored_id:
        sponsored = [p for p in products if p["product_id"] == sponsored_id]
        if not sponsored:
            issues.append(
                f"[{category}] sponsored_product '{sponsored_id}' not found in catalog"
            )

    # --- Use-case fit completeness ---
    for p in products:
        missing = [uc for uc in use_cases if uc not in p["use_case_fit"]]
        if missing:
            issues.append(
                f"[{category}] Product {p['product_id']} missing use_case_fit for: {missing}"
            )

    # --- Unique product IDs ---
    ids = [p["product_id"] for p in products]
    if len(ids) != len(set(ids)):
        issues.append(f"[{category}] Duplicate product IDs found")

    # --- Review summary and weakness not empty ---
    for p in products:
        if len(p.get("review_summary", "")) < 50:
            issues.append(f"[{category}] {p['product_id']}: review_summary too short")
        if len(p.get("weakness", "")) < 20:
            issues.append(f"[{category}] {p['product_id']}: weakness too short")

    return issues


def print_catalog_summary(catalog: dict):
    """Print a human-readable summary for quick evaluation."""
    cat = catalog["category"]
    products = catalog["products"]
    use_cases = catalog["use_cases"]

    print(f"\n{'='*70}")
    print(f"  {catalog['category_display_name']}  ({cat})")
    print(f"  Complexity: {catalog['complexity']}  |  Products: {len(products)}")
    print(f"  Focal brand: {catalog['focal_brand']}  |  Sponsored: {catalog['sponsored_product']}")
    print(f"{'='*70}")

    # Price & quality table
    print(f"\n  {'Product':<35} {'Brand':<12} {'Status':<10} {'Price':>8} {'Quality':>7}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*8} {'-'*7}")
    for p in sorted(products, key=lambda x: -x["quality_score"]):
        print(
            f"  {p['product_id']:<35} {p['brand_name']:<12} "
            f"{p['brand_status']:<10} ${p['price']:>7.2f} {p['quality_score']:>5}"
        )

    # Use-case top picks
    print(f"\n  Top product per use case:")
    for uc in use_cases:
        best = max(products, key=lambda p: p["use_case_fit"].get(uc, 0))
        score = best["use_case_fit"].get(uc, 0)
        print(f"    {uc:<20} → {best['product_id']:<35} (fit={score:.2f})")

    # Entrant vs incumbent counts
    n_inc = sum(1 for p in products if p["brand_status"] == "incumbent")
    n_ent = sum(1 for p in products if p["brand_status"] == "entrant")
    print(f"\n  Brand mix: {n_inc} incumbents, {n_ent} entrants")

    prices = [p["price"] for p in products]
    print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f} (ratio: {max(prices)/min(prices):.1f}x)")

    qs = [p["quality_score"] for p in products]
    print(f"  Quality range: {min(qs)} - {max(qs)} (spread: {max(qs)-min(qs)} pts)")


def main():
    catalogs = {
        "phone_charger": build_phone_charger_catalog(),
        "headphones": build_headphones_catalog(),
        "laptop": build_laptop_catalog(),
    }

    all_issues = []

    for name, catalog in catalogs.items():
        # Validate
        issues = validate_catalog(catalog)
        all_issues.extend(issues)

        # Print summary
        print_catalog_summary(catalog)

        # Save
        out_path = OUTPUT_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"\n  Saved to: {out_path}")

    # Report issues
    print(f"\n{'='*70}")
    print("  VALIDATION REPORT")
    print(f"{'='*70}")
    if all_issues:
        for issue in all_issues:
            print(f"  ISSUE: {issue}")
    else:
        print("  All catalogs passed validation.")

    return len(all_issues)


if __name__ == "__main__":
    exit(main())
