"""
OpenRouter Provider for GPT-4o integration
Handles prompt enhancement and product image analysis
"""
import httpx
import json
import re
from typing import Dict, Any


class OpenRouterProvider:
    """GPT-4o integration via OpenRouter for prompt enhancement"""

    BASE_URL = "https://openrouter.ai/api/v1"

    # System prompts based on n8n workflow
    PRODUCT_ANALYSIS_PROMPT = """Analyze the given image and determine if it primarily depicts a product or a character, or BOTH.

- If the image is of a PRODUCT, return the analysis in JSON format with the following fields:
  {
    "type": "product",
    "brand_name": "Name of the brand shown in the image, if visible or inferable",
    "color_scheme": [
      {"hex": "Hex code of each prominent color", "name": "Descriptive color name"}
    ],
    "font_style": "Describe the font family or style used: serif/sans-serif, bold/thin, etc.",
    "visual_description": "A full sentence or two summarizing what is seen in the image, ignoring the background"
  }

- If the image is of a CHARACTER, return the analysis in JSON format with the following fields:
  {
    "type": "character",
    "character_name": "Name of the character if visible or inferable",
    "color_scheme": [
      {"hex": "Hex code of each prominent color on the character", "name": "Descriptive color name"}
    ],
    "outfit_style": "Description of clothing style, accessories, or notable features",
    "visual_description": "A full sentence or two summarizing what the character looks like, ignoring the background"
  }

- If it is BOTH (product AND character), return BOTH descriptions in JSON format:
  {
    "type": "both",
    "product": { ... },
    "character": { ... }
  }

Only return the JSON. Do not explain or add any other comments."""

    PERSON_GENERATION_SYSTEM_PROMPT = """Generate a detailed prompt for AI person generation that emphasizes UGC casual, authentic style.

**CRITICAL: This is STAGE 1 - Generate ONLY the person. DO NOT include any product in the image.**
The person should be ready to hold a product (hand positioned naturally), but the product itself will be added in Stage 2.

If product information is provided, use it ONLY to:
- Match the person's style/aesthetic to complement the product
- Optimize lighting and color palette to match the product's branding
- Choose appropriate setting/background that fits the product category
- But DO NOT put the product in the person's hand or anywhere in the image

Requirements for UGC authentic casual content:
- Everyday realism with authentic, relatable settings
- Amateur-quality iPhone photo style
- Slightly imperfect framing and lighting
- Candid poses and genuine expressions
- Visible imperfections (blemishes, messy hair, uneven skin, texture flaws)
- Real-world environments left as-is (clutter, busy backgrounds)
- Person should have hand positioned naturally, ready to hold product (but no product visible)

Camera parameters MUST ALWAYS include these keywords:
unremarkable amateur iPhone photos, reddit image, snapchat photo, Casual iPhone selfie, slightly uneven framing, Authentic share, slightly blurry, Amateur quality phone photo

Output format should include these elements (structured as a detailed prompt):
- emotion: person's emotion and expression
- action: what the person is doing (e.g., "hand raised naturally ready to showcase product")
- character: person description (age, gender, ethnicity, clothing)
- setting: environment and background (informed by product context if provided)
- camera: camera keywords and framing
- style: visual style (UGC casual)
- composition: how elements are arranged
- lighting: lighting conditions (can be informed by product branding colors)
- color_palette: dominant colors (can complement product if info provided)

**REMEMBER: NO PRODUCT IN THE IMAGE. Only the person ready to hold it.**

Return ONLY the enhanced prompt text incorporating all these elements, no explanation."""

    COMPOSITE_SYSTEM_PROMPT = """Generate a detailed prompt for compositing a product into a person's hand with authentic UGC casual style.

**CRITICAL REQUIREMENTS:**
- Show the SAME person now naturally holding the product
- **DO NOT create extra hands or duplicate body parts** - the person should have normal anatomy
- Keep the person's face, expression, and overall body position similar to the original
- The hand can move naturally to hold and present the product (this is expected)
- Keep the background and overall framing similar
- The result should show the same person in the same scene, now presenting the product

Requirements for natural, realistic compositing:
- **Natural presentation distance**: Product held at comfortable distance from camera - not too close (which makes it look oversized)
- **Relaxed hand grip**: Fingers naturally curved around product, as if casually showing it to a friend
- **Appropriate scale**: Product should appear normal-sized in the frame, not dominating or oversized
- **Comfortable pose**: Person looks relaxed and natural, not awkwardly reaching or posing
- Match lighting and shadows to the existing environment
- Realistic perspective and scale - product should look naturally held at normal distance
- Product clearly visible and legible with all text readable
- **CRITICAL: Maintain all product text and branding EXACTLY as shown (logos, slogans, packaging claims). Never alter or invent text.**
- Natural shadows where hand touches product
- Seamless integration with casual, authentic UGC presentation style
- Avoid making product too prominent or camera-facing - keep it natural

UGC Style Keywords to Include:
- Natural hand positioning, realistic grip
- Amateur iPhone photo aesthetic
- Slightly uneven lighting (natural/realistic)
- Authentic integration (not perfectly centered or staged)
- Casual presentation style

Output format should include these elements (structured as a detailed prompt):
- action: how the product is naturally integrated with the person
- product: product description with ACCURATE text preservation
- hand_position: realistic hand interaction with product
- framing: CRITICAL - both person and product completely within frame, nothing cropped
- composition: person and product centered and well-positioned
- scale: product appropriately sized relative to hand (not too big or small)
- lighting: match environment (slightly imperfect is authentic)
- shadows: natural shadow placement where hand touches product
- perspective: realistic scale and angle
- style: UGC casual aesthetic, amateur quality
- integration: seamless as if always held by person
- visibility: complete visibility of person and product - nothing cut off
- text_accuracy: all product text preserved exactly as shown

Return ONLY the enhanced prompt text incorporating all these elements, no explanation."""

    VIDEO_SYSTEM_PROMPT = """Generate a detailed video animation prompt emphasizing UGC casual, authentic style.

All outputs must feel natural, candid, and unpolished:
- Everyday realism with authentic, relatable settings
- Amateur-quality iPhone video style
- Slightly imperfect framing and camera movement
- Candid motion and genuine expressions
- Visible imperfections throughout
- Real-world authenticity

Camera parameters MUST ALWAYS include these keywords:
unremarkable amateur iPhone video, reddit image, snapchat video, Casual iPhone selfie video, slightly uneven framing, Authentic share, slightly shaky camera, Amateur quality phone video

**Character Action (CRITICAL - Minimize Artifacts):**
Unless explicitly requested by the user, the character should ONLY show/present the product to the camera - NOT open it, eat it, drink it, or use it. Just natural presentation and casual showing.

**Motion Guidelines (Avoid Artifacts - CRITICAL):**
To prevent visual artifacts, follow the n8n workflow style - EXTREMELY static videos:
- Character should be COMPLETELY STILL except for talking (mouth movement only)
- NO hand movements, gestures, or repositioning - hands stay frozen in position
- Product stays STATIONARY in hand - absolutely no movement
- Character can only move: eyes blinking, mouth talking, very slight head movement
- NO pointing, waving, reaching, adjusting, or any hand gestures whatsoever
- Body position FROZEN - only facial expressions change
- Camera is COMPLETELY STATIC - no panning, zooming, or movement
- Think: "person sitting still, holding product, just talking" - that's it!
- Example good action: "character sits holding the product casually while speaking"
- Example bad action: "character waves product, gestures enthusiastically, moves around"

**Dialogue generation (if needed):**
If the user input doesn't include specific dialogue, generate a casual, conversational line UNDER 150 CHARACTERS total, as if a person were speaking naturally to a friend while talking about the product. Requirements:
- Avoid overly formal or sales-like language
- The tone should feel authentic, spontaneous, and relatable, matching the UGC style
- Use ... (ellipsis) to indicate natural pauses
- Avoid special characters like em dashes (—) or hyphens in dialogue
- Make it sound like real casual speech, not scripted advertising
- Talk about the product benefits based on what you understand about it:
  * If it's a drink → mention taste, refreshing qualities
  * If it's a bag/accessory → mention design, style, practicality
  * If it's tech → mention features, functionality, ease of use
  * If it's skincare/beauty → mention how it feels, results, texture
  * And so on, matching the product category naturally

Output format should include these elements (structured as a detailed prompt):
- dialogue: casual conversation about the product (<150 chars if needed, use ... for pauses)
- action: what is happening in the video (MINIMAL motion - "holding product naturally while talking" NOT "waving product enthusiastically")
- camera: camera keywords and movement (mostly STATIC, minimal panning)
- emotion: person's emotion and energy level (expressed through face/voice, NOT body movement)
- character: person description
- setting: environment and background
- motion: MINIMAL and SUBTLE only - character mostly still, slight natural movements (head tilt, smile)
- pacing: SLOW and STEADY to avoid artifacts

**REMEMBER: Less motion = fewer artifacts. Keep it simple and mostly static!**

Return ONLY the enhanced prompt text incorporating all these elements, no explanation."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        """
        Initialize OpenRouter provider

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-4o)
        """
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ugcvideo.io",
            "X-Title": "UGC Video Generator"
        }

    async def _chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Internal method for chat completion requests

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response content string

        Raises:
            Exception: If API call fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30.0
            )

            data = response.json()

            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                raise Exception(f"OpenRouter API error: {error_msg}")

            return data['choices'][0]['message']['content'].strip()

    async def analyze_product_image(self, image_url: str) -> Dict[str, Any]:
        """
        Analyze product image using GPT-4o vision

        Args:
            image_url: Public URL of product image

        Returns:
            dict: {
                brand_name: str,
                color_scheme: list[str],
                font_style: str,
                visual_description: str
            }

        Raises:
            Exception: If analysis fails or JSON parsing fails
        """
        print(f"🔍 Analyzing product image with GPT-4o...")

        content = await self._chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PRODUCT_ANALYSIS_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Parse JSON from response
        try:
            result = json.loads(content)
            print(f"✅ Product analysis complete: {result.get('brand_name', 'Unknown brand')}")
            return result
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code block
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                print(f"✅ Product analysis complete: {result.get('brand_name', 'Unknown brand')}")
                return result
            raise Exception("Failed to parse JSON response from GPT-4o")

    async def enhance_person_prompt(
        self,
        user_input: str,
        product_info: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced person generation prompt

        Args:
            user_input: User's description (Easy Mode fields or Advanced Mode text)
            product_info: Result from analyze_product_image()

        Returns:
            str: Enhanced prompt with UGC keywords

        Raises:
            Exception: If prompt enhancement fails
        """
        print(f"✨ Enhancing person generation prompt...")

        # Extract color names from color_scheme (which may be list of dicts or list of strings)
        color_scheme = product_info.get('color_scheme', [])
        if color_scheme and isinstance(color_scheme[0], dict):
            # New format: [{"hex": "...", "name": "..."}, ...]
            color_names = [c.get('name', '') for c in color_scheme]
        else:
            # Old format: ["red", "blue", ...]
            color_names = color_scheme

        user_message = f"""User wants to generate a person with these characteristics:
{user_input}

Product being promoted:
Brand: {product_info.get('brand_name', 'Unknown')}
Colors: {', '.join(color_names)}
Description: {product_info.get('visual_description', '')}

Generate an enhanced prompt for person generation."""

        enhanced_prompt = await self._chat_completion(
            messages=[
                {"role": "system", "content": self.PERSON_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8,
            max_tokens=500
        )

        print(f"✅ Person prompt enhanced: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    async def enhance_composite_prompt(
        self,
        person_info: Dict[str, Any],
        product_info: Dict[str, Any],
        user_instructions: str = ""
    ) -> str:
        """
        Generate enhanced composite prompt

        Args:
            person_info: Details about generated person
            product_info: Result from analyze_product_image()
            user_instructions: Optional custom instructions

        Returns:
            str: Enhanced compositing prompt

        Raises:
            Exception: If prompt enhancement fails
        """
        print(f"✨ Enhancing composite prompt...")

        # Extract color names from color_scheme (which may be list of dicts or list of strings)
        color_scheme = product_info.get('color_scheme', [])
        if color_scheme and isinstance(color_scheme[0], dict):
            # New format: [{"hex": "...", "name": "..."}, ...]
            color_names = [c.get('name', '') for c in color_scheme]
        else:
            # Old format: ["red", "blue", ...]
            color_names = color_scheme

        user_message = f"""Person image: {person_info.get('url', 'Generated person')}

Product details:
Brand: {product_info.get('brand_name', 'Unknown')}
Colors: {', '.join(color_names)}
Description: {product_info.get('visual_description', '')}

User instructions: {user_instructions or "Standard natural integration"}

Generate composite prompt."""

        enhanced_prompt = await self._chat_completion(
            messages=[
                {"role": "system", "content": self.COMPOSITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=400
        )

        print(f"✅ Composite prompt enhanced: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    async def enhance_video_prompt(
        self,
        composite_info: Dict[str, Any],
        user_input: str,
        product_type: str = "product"
    ) -> str:
        """
        Generate enhanced video animation prompt

        Args:
            composite_info: Details about composite image
            user_input: User's desired animation description
            product_type: Type of product (for context)

        Returns:
            str: Enhanced video prompt with UGC keywords

        Raises:
            Exception: If prompt enhancement fails
        """
        print(f"✨ Enhancing video animation prompt...")

        user_message = f"""Composite image URL: {composite_info.get('url', 'Composite with person and product')}

Product type: {product_type}

User wants this animation:
{user_input}

Generate video animation prompt."""

        enhanced_prompt = await self._chat_completion(
            messages=[
                {"role": "system", "content": self.VIDEO_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8,
            max_tokens=400
        )

        print(f"✅ Video prompt enhanced: {enhanced_prompt[:100]}...")
        return enhanced_prompt
