"""
Test script to verify n8n-style prompt generation
"""
from app.services.providers.n8n_prompt_builder import N8nPromptBuilder


def test_person_prompt():
    print("=" * 80)
    print("STAGE 1: Person Generation Prompt (n8n YAML format)")
    print("=" * 80)

    prompt = N8nPromptBuilder.build_person_prompt(
        age="20s",
        gender="female",
        ethnicity="hispanic",
        expression="happy",
        clothing="casual",
        background="home"
    )

    print(prompt)
    print("\n")


def test_composite_prompt():
    print("=" * 80)
    print("STAGE 2: Composite Prompt (n8n YAML format)")
    print("=" * 80)

    prompt = N8nPromptBuilder.build_composite_prompt(
        person_description="person from the original image",
        product_description="orange energy drink can with bold logo",
        brand_name="EnergyBoost"
    )

    print(prompt)
    print("\n")


def test_video_prompt():
    print("=" * 80)
    print("STAGE 3: Video Prompt (n8n YAML format)")
    print("=" * 80)

    # Test with auto-generated dialogue
    dialogue = N8nPromptBuilder.generate_casual_dialogue(
        product_type="drink",
        brand_name="EnergyBoost"
    )

    prompt = N8nPromptBuilder.build_video_prompt(
        dialogue=dialogue,
        action="character sits holding the product casually while speaking",
        emotion="happy and casual",
        character_description="person from the composite image",
        product_type="drink"
    )

    print(prompt)
    print("\n")


if __name__ == "__main__":
    print("\n🧪 Testing n8n-style prompt generation...\n")

    test_person_prompt()
    test_composite_prompt()
    test_video_prompt()

    print("=" * 80)
    print("✅ All prompts generated successfully!")
    print("=" * 80)
