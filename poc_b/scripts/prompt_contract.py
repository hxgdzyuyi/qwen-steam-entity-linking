"""Dependency-light prompt contract for PoC B training and inference."""

from __future__ import annotations


DEFAULT_PROMPT_STYLE = "steam_game"
DEFAULT_PROMPT_TEMPLATE = "Steam 游戏：{surface_form}"

# Keep every template short and place the entity at the end. PoC B pools the
# final non-padding token, so suffix-style questions would move the pooling
# position away from the entity and make the shared task text more dominant.
PROMPT_STYLES: tuple[tuple[str, str], ...] = (
    (DEFAULT_PROMPT_STYLE, DEFAULT_PROMPT_TEMPLATE),
    ("raw", "{surface_form}"),
    ("game_entity", "游戏实体：{surface_form}"),
    ("game_name", "游戏名称：{surface_form}"),
    ("identify_game", "需要识别的游戏：{surface_form}"),
    ("english_entity", "Steam game entity: {surface_form}"),
)


def prompt_style_names() -> tuple[str, ...]:
    return tuple(name for name, _ in PROMPT_STYLES)


def render_prompt(surface_form: str, style: str) -> str:
    templates = dict(PROMPT_STYLES)
    if style not in templates:
        raise ValueError(f"unknown prompt style: {style}")
    rendered = templates[style].format(surface_form=surface_form)
    if not rendered.endswith(surface_form):
        raise ValueError(f"prompt style {style} must end with the surface form")
    return rendered
