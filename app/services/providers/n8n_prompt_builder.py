"""
n8n-Style Prompt Builder
Builds YAML-formatted prompts matching the n8n workflow structure
"""
from typing import Dict, Any, List


class N8nPromptBuilder:
    """Build prompts in n8n YAML format for image and video generation"""

    # UGC camera keywords (from n8n workflow)
    IMAGE_CAMERA_KEYWORDS = "unremarkable amateur iPhone photo, reddit image, snapchat photo, Casual iPhone selfie, slightly uneven framing, Authentic share, slightly blurry, Amateur quality phone photo"

    VIDEO_CAMERA_KEYWORDS = "amateur iphone selfie video, unremarkable amateur iPhone video, snapchat video, Casual iPhone selfie video, slightly uneven framing, Authentic share, slightly shaky camera, Amateur quality phone video"

    @staticmethod
    def build_person_prompt(
        age: str,
        gender: str,
        ethnicity: str,
        expression: str,
        clothing: str,
        background: str
    ) -> str:
        """
        Build Stage 1 person generation prompt in n8n YAML format

        Args:
            age: Age range (e.g., "20s", "30s")
            gender: Gender (e.g., "male", "female")
            ethnicity: Ethnicity (e.g., "caucasian", "hispanic")
            expression: Facial expression (e.g., "happy", "excited")
            clothing: Clothing style (e.g., "casual", "business")
            background: Background setting (e.g., "home", "outdoor")

        Returns:
            YAML-formatted prompt string
        """
        # Map inputs to natural descriptions
        age_map = {
            "20s": "person in their 20s",
            "30s": "person in their 30s",
            "teens": "teenager",
            "40s": "person in their 40s"
        }

        clothing_map = {
            "casual": "casual everyday clothing",
            "business": "business casual attire",
            "athletic": "athletic wear",
            "streetwear": "trendy streetwear"
        }

        background_map = {
            "home": "cozy home interior",
            "outdoor": "casual outdoor setting",
            "office": "modern office environment",
            "cafe": "coffee shop interior"
        }

        # Build character description
        age_desc = age_map.get(age, age)
        character_desc = f"{age_desc}, {ethnicity} {gender}, {expression} expression, wearing {clothing_map.get(clothing, clothing)}"

        # Build YAML prompt (matching n8n structure)
        prompt_lines = [
            f"action: person with hand raised naturally in front of body, palm open and visible",
            f"character: {character_desc}",
            f"setting: {background_map.get(background, background)}, casual real-world environment",
            f"camera: {N8nPromptBuilder.IMAGE_CAMERA_KEYWORDS}",
            f"style: candid UGC look, no filters, visible imperfections, messy hair, uneven skin texture, blemishes, natural lighting with slight imperfections",
            f"composition: Medium shot portrait, person centered with space around them",
            f"framing: good framing with person fully visible in frame"
        ]

        return "\n".join(prompt_lines)

    @staticmethod
    def build_composite_prompt(
        person_description: str,
        product_description: str,
        brand_name: str = "Unknown"
    ) -> str:
        """
        Build Stage 2 composite prompt in n8n YAML format

        Args:
            person_description: Description of the person from Stage 1
            product_description: Visual description of the product
            brand_name: Brand name of the product

        Returns:
            YAML-formatted prompt string
        """
        # Build YAML prompt (matching n8n structure for compositing)
        prompt_lines = [
            f"action: same person now casually presenting the product to camera, held at comfortable natural distance",
            f"character: keep the same person from the original image - same face, expression, and overall appearance",
            f"product: {product_description}, show product with all visible text clear and accurate, brand name {brand_name}",
            f"hand_position: relaxed grip with fingers naturally curved around product, as if showing it to a friend",
            f"framing: both person and product completely within frame, nothing cropped, well-positioned composition",
            f"scale: product appears normal-sized relative to hand, not oversized or too prominent",
            f"lighting: preserve the EXACT same lighting, exposure, and color temperature from the original image - do not brighten or change lighting",
            f"camera: {N8nPromptBuilder.IMAGE_CAMERA_KEYWORDS}",
            f"style: candid UGC look, natural integration, slightly imperfect lighting is authentic",
            f"integration: seamless as if person always held the product, natural shadows where hand touches product",
            f"text_accuracy: preserve all visible product text exactly as shown - logos, slogans, packaging claims"
        ]

        return "\n".join(prompt_lines)

    @staticmethod
    def build_video_prompt(
        dialogue: str,
        action: str,
        emotion: str,
        character_description: str = "person from the image",
        product_type: str = "product"
    ) -> str:
        """
        Build Stage 3 video animation prompt in n8n YAML format

        Args:
            dialogue: What the person says in the video (under 150 chars)
            action: What the person is doing (should be minimal/static)
            emotion: Emotional tone (e.g., "happy", "excited", "casual")
            character_description: Description of the person
            product_type: Type of product being shown

        Returns:
            YAML-formatted prompt string
        """
        # Ensure action is static (matching n8n best practices)
        if not action:
            action = "character sits holding the product casually while speaking"

        # Build YAML prompt (matching n8n video structure)
        prompt_lines = [
            f"dialogue: {dialogue}",
            f"action: {action}",
            f"camera: {N8nPromptBuilder.VIDEO_CAMERA_KEYWORDS}, natural daylight",
            f"emotion: {emotion}, casual and authentic",
            f"character: {character_description}",
            f"motion: MINIMAL - character completely still except for talking, mouth movement only, hands frozen in position holding product",
            f"product_movement: product stays STATIONARY in hand, absolutely no movement",
            f"body: body position FROZEN, only facial expressions change through face and voice",
            f"pacing: SLOW and STEADY to avoid artifacts"
        ]

        return "\n".join(prompt_lines)

    @staticmethod
    def generate_casual_dialogue(product_type: str, brand_name: str = "") -> str:
        """
        Generate casual UGC-style dialogue based on product type

        Args:
            product_type: Type of product (drink, bag, tech, skincare, etc.)
            brand_name: Optional brand name to mention

        Returns:
            Casual dialogue string under 150 characters
        """
        # Product-specific dialogue templates (matching n8n style)
        templates = {
            "drink": [
                f"okay so... {brand_name + ' ' if brand_name else ''}this is actually really good... super refreshing",
                f"so I tried this and... honestly tastes amazing... would definitely recommend",
                f"this has been my go-to lately... the flavor is just perfect"
            ],
            "beverage": [
                f"okay so... {brand_name + ' ' if brand_name else ''}this is actually really good... super refreshing",
                f"so I tried this and... honestly tastes amazing... would definitely recommend"
            ],
            "bag": [
                f"so I got this bag and... it fits everything I need... love the design",
                f"okay so this bag... super practical and looks really good... been using it every day"
            ],
            "accessory": [
                f"so I got this and... honestly love it... goes with everything",
                f"okay so... {brand_name + ' ' if brand_name else ''}this is actually perfect... really good quality"
            ],
            "tech": [
                f"so I tried this out and... works really well... super easy to use",
                f"okay so... {brand_name + ' ' if brand_name else ''}this is actually pretty cool... the features are great"
            ],
            "skincare": [
                f"so I've been using this and... my skin feels so much better... highly recommend",
                f"okay so... {brand_name + ' ' if brand_name else ''}this has been amazing... texture is so good"
            ],
            "beauty": [
                f"so I tried this and... honestly really impressed... the results are great",
                f"okay so... this is actually really good... been using it for weeks now"
            ],
            "food": [
                f"so I tried this and... honestly so good... you need to try it",
                f"okay so... {brand_name + ' ' if brand_name else ''}this is actually delicious... highly recommend"
            ]
        }

        # Default generic dialogue
        default = [
            f"so I got this and... honestly really love it... would definitely recommend",
            f"okay so... {brand_name + ' ' if brand_name else ''}this is actually really good... been using it a lot"
        ]

        # Get appropriate template
        dialogue_options = templates.get(product_type.lower(), default)

        # Return first option (can be randomized later if needed)
        return dialogue_options[0]
