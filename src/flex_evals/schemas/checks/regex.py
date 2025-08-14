"""RegexCheck schema class for type-safe regex check definitions."""

from typing import ClassVar
from pydantic import BaseModel, Field

from ...constants import CheckType
from ..check import Check, SchemaCheck, OptionalJSONPath


class RegexFlags(BaseModel):
    """Regex matching options."""

    case_insensitive: bool = Field(False, description="If true, ignores case when matching")
    multiline: bool = Field(False, description="If true, ^ and $ match line boundaries")
    dot_all: bool = Field(False, description="If true, . matches newline characters")


class RegexCheck(SchemaCheck):
    """Checks if a text value matches a specified regular expression pattern."""

    VERSION: ClassVar[str] = "1.0.0"
    CHECK_TYPE: ClassVar[CheckType] = CheckType.REGEX

    text: str = OptionalJSONPath(
        "Text to test against the pattern or JSONPath expression pointing to the text",
    )
    pattern: str = OptionalJSONPath(
        "Regular expression pattern to match against the text, or JSONPath expression pointing to pattern",  # noqa: E501
    )
    negate: bool = Field(False, description="If true, passes when pattern doesn't match")
    flags: RegexFlags | None = Field(None, description="Regex matching options")

    def to_check(self) -> Check:
        """Convert to generic Check object for execution."""
        arguments = {
            "text": self.text,
            "pattern": self.pattern,
            "negate": self.negate,
        }

        if self.flags is not None:
            arguments["flags"] = self.flags.model_dump()

        return Check(
            type=self.check_type,
            arguments=arguments,
            version=self.VERSION,
        )
