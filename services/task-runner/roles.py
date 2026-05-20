"""
Role definitions for the Kimi Agent Harness trajectory system.
Each message in the trajectory is tagged with one of these roles.
"""

from enum import Enum


class Role(str, Enum):
    SYSTEM    = "system"     # Initial system prompt
    USER      = "user"       # Human task instruction
    ASSISTANT = "assistant"  # LLM generated response
    TOOL      = "tool"       # Tool call result (search, vision, file ops)
