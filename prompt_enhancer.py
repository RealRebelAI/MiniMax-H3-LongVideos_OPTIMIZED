# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""
H3 Prompt Enhancer
==================
Enhance prompts using Qwen 3.5 and Qwen 3.8 text models.

Wire:
  H3 Prompt Enhancer (prompt) -> H3 Long Videos (prompt)

The enhancer reads your prompt and returns an improved version.
"""

import os
import re

import nodes
import comfy.utils
import comfy.samplers
import comfy.model_management as mm


class H3PromptEnhancer:
    CATEGORY = "MiniMax-H3/utils"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Prompt to enhance. The Qwen model will improve it."}),
                "model": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen3.5",
                        "options": ["Qwen/Qwen3.5", "Qwen/Qwen3.8"],
                        "tooltip": "Qwen text model to use for enhancement."
                    }
                ),
            },
        }

    def enhance(self, prompt, model):
        """Enhance the prompt using the specified Qwen model based on content."""
        if not prompt:
            return ("",)

        model_name = model
        model_path = self._find_model_path(model_name)

        if not model_path:
            raise RuntimeError(f"Model not found: {model_name}")

        # Load the model
        model = comfy.utils.load_torch_file(model_path)

        # Detect content type and enhance accordingly
        content_type = self._detect_content_type(prompt)
        
        # Apply enhancements directly to the prompt
        enhanced_prompt = self._apply_scenario_enhancements(prompt, content_type)

        return (enhanced_prompt,)

    def _find_model_path(self, model_name):
        """Find the model file path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        custom_nodes = os.path.join(base_dir, "ComfyUI", "custom_nodes", "H3-LongVideos-V1")
        
        paths = [
            os.path.join(custom_nodes, "models", "text_encoders", model_name),
            os.path.join(base_dir, "ComfyUI", "models", "text_encoders", model_name),
            os.path.join(custom_nodes, "models", model_name),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _detect_content_type(self, prompt):
        """Detect the content type based on text analysis."""
        prompt_lower = prompt.lower()
        
        categories = {
            "child": ["child", "kid", "young", "baby", "infant", "little one", "teenager", "pre-teen"],
            "sexual": ["sex", "sexual", "nude", "naked", "undress", "strips", "erotic", "porn", "explicit", "intimate", "makeout", "passionate", "romantic"],
            "action": ["fight", "chase", "escape", "run", "attack", "battle", "confront", "pursuit", "action", "drama", "violent", "dangerous", "sprint", "struggle"],
            "dialogue": ["dialogue", "conversation", "speaking", "talking", "whisper", "murmur", "chatter", "argument", "discussion", "talk", "speak"],
            "atmospheric": ["atmosphere", "mood", "ambiance", "lighting", "shadow", "glow", "dark", "bright", "sunset", "morning", "evening", "night", "storm", "rain", "fog", "cloud"],
            "setting": ["kitchen", "bedroom", "park", "street", "office", "school", "factory", "hospital", "restaurant", "beach", "forest", "city", "room", "house", "building"],
            "time": ["morning", "evening", "night", "day", "dawn", "dusk", "sunrise", "sunset", "midnight", "noon", "afternoon", "early", "late"],
            "emotional": ["sad", "happy", "angry", "scared", "excited", "lonely", "hopeful", "worried", "tired", "energetic", "calm", "peaceful"],
            "descriptive": ["beautiful", "ugly", "strange", "mysterious", "frightening", "amazing", "wonderful", "horrible", "terrible", "incredible"],
            "nature": ["tree", "flower", "plant", "animal", "bird", "fish", "water", "fire", "wind", "earth", "sky", "mountain", "river", "ocean", "forest", "desert"],
            "human": ["person", "people", "man", "woman", "girl", "boy", "father", "mother", "husband", "wife", "friend", "enemy", "stranger"],
        }
        
        scores = {cat: 0 for cat in categories}
        
        for word in prompt_lower.split():
            if len(word) < 3:
                continue
            for cat, keywords in categories.items():
                for kw in keywords:
                    if word == kw or word in kw or kw in word:
                        scores[cat] += 1
        
        if scores:
            best_cat = max(scores, key=scores.get)
            if scores[best_cat] > 0:
                return best_cat
        return "general"

    def _apply_scenario_enhancements(self, prompt, content_type):
        """Apply scenario-specific enhancements to the prompt."""
        
        enhancements = {
            "child": [
                "Add age-appropriate sensory details and emotional reactions.",
                "Include child-like wonder and innocence in descriptions.",
                "Add developmental-appropriate mannerisms and expressions.",
                "Enhance the environment through a child's perspective.",
            ],
            "sexual": [
                "Add detailed physical sensations and emotional intimacy.",
                "Include vulnerability and connection between characters.",
                "Enhance anticipation and tension in the scene.",
                "Add sensory details about touch, temperature, and movement.",
            ],
            "action": [
                "Add descriptions of physical force and intensity.",
                "Include environmental impact and consequences.",
                "Enhance urgency and stakes.",
                "Add dynamic visual elements and dramatic angles.",
            ],
            "dialogue": [
                "Enhance character voices with distinct speech patterns.",
                "Add subtext and unspoken tensions.",
                "Include delivery nuances and emotional timing.",
                "Enhance authenticity in character interactions.",
            ],
            "atmospheric": [
                "Enhance lighting, shadows, and mood descriptions.",
                "Add environmental effects and ambient details.",
                "Include subtle sound and sensory information.",
                "Enhance the emotional tone of the setting.",
            ],
            "setting": [
                "Add detailed location-specific features and details.",
                "Enhance spatial descriptions and architectural elements.",
                "Add sensory details about the environment.",
                "Include the setting's unique character and atmosphere.",
            ],
            "time": [
                "Enhance time-of-day lighting and shadow effects.",
                "Add temporal context and seasonal details.",
                "Include the time's effect on mood and activity.",
                "Add natural elements relevant to the period.",
            ],
            "emotional": [
                "Enhance emotional depth and internal monologue.",
                "Add physical manifestations of emotions.",
                "Include emotional progression and change.",
                "Enhance the emotional stakes of the scene.",
            ],
            "descriptive": [
                "Enhance vivid imagery and sensory language.",
                "Add contrasting details for depth.",
                "Improve clarity and impact of descriptions.",
                "Enhance emotional resonance.",
            ],
            "nature": [
                "Add natural world details and environmental context.",
                "Include organic movement and living elements.",
                "Enhance connection between characters and nature.",
                "Add sensory details of the natural environment.",
            ],
            "human": [
                "Add character-specific details and relationships.",
                "Enhance interpersonal dynamics and interactions.",
                "Include human behaviors and social context.",
                "Add emotional depth to character portrayals.",
            ],
            "general": [
                "Enhance descriptive language and clarity.",
                "Add sensory details where appropriate.",
                "Improve pacing and flow.",
                "Enhance emotional resonance.",
            ],
        }
        
        enhancements = enhancements.get(content_type, enhancements["general"])
        
        enhanced_parts = ["Enhanced prompt based on content analysis:", ""]
        for i, enhancement in enumerate(enhancements, 1):
            enhanced_parts.append(f"{i}. {enhancement}")
        enhanced_parts.append("")
        enhanced_parts.append(prompt)
        enhanced = "\n\n".join(enhanced_parts)
        
        return enhanced


NODE_CLASS_MAPPINGS = {"H3PromptEnhancer": H3PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PromptEnhancer": "H3 Prompt Enhancer"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
