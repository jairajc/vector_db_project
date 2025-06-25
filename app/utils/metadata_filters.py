"""Advanced metadata filtering system for vector search"""

import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json


class FilterOperator(Enum):
    """Supported filter operators"""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    CONTAINS = "contains"  # String contains (case-insensitive)
    STARTS_WITH = "starts_with"  # String starts with
    ENDS_WITH = "ends_with"  # String ends with
    REGEX = "regex"  # Regular expression match
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist
    DATE_RANGE = "date_range"  # Date within range
    ARRAY_CONTAINS = "array_contains"  # Array contains value
    ARRAY_LENGTH = "array_length"  # Array has specific length


@dataclass
class DateRange:
    """Date range for filtering"""

    start: Optional[datetime] = None
    end: Optional[datetime] = None


@dataclass
class MetadataFilter:
    """Enhanced metadata filter with support for complex operations"""

    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False

    def __post_init__(self):
        if isinstance(self.operator, str):
            self.operator = FilterOperator(self.operator)


class MetadataFilterEngine:
    """
    Advanced metadata filtering engine with:
    - Complex query evaluation
    - Type coercion
    - Performance optimization
    - Error handling
    """

    def __init__(self):
        self._operator_handlers = {
            FilterOperator.EQ: self._handle_eq,
            FilterOperator.NE: self._handle_ne,
            FilterOperator.GT: self._handle_gt,
            FilterOperator.GTE: self._handle_gte,
            FilterOperator.LT: self._handle_lt,
            FilterOperator.LTE: self._handle_lte,
            FilterOperator.IN: self._handle_in,
            FilterOperator.NOT_IN: self._handle_not_in,
            FilterOperator.CONTAINS: self._handle_contains,
            FilterOperator.STARTS_WITH: self._handle_starts_with,
            FilterOperator.ENDS_WITH: self._handle_ends_with,
            FilterOperator.REGEX: self._handle_regex,
            FilterOperator.EXISTS: self._handle_exists,
            FilterOperator.NOT_EXISTS: self._handle_not_exists,
            FilterOperator.DATE_RANGE: self._handle_date_range,
            FilterOperator.ARRAY_CONTAINS: self._handle_array_contains,
            FilterOperator.ARRAY_LENGTH: self._handle_array_length,
        }

    def apply_filters(
        self, metadata: Dict[str, Any], filters: List[MetadataFilter], mode: str = "and"
    ) -> bool:
        """
        Apply metadata filters to a metadata dictionary

        Args:
            metadata: Metadata dictionary to filter
            filters: List of filters to apply
            mode: Filter combination mode ("and" or "or")

        Returns:
            True if metadata passes filters, False otherwise
        """
        if not filters:
            return True

        results = []
        for filter_obj in filters:
            try:
                result = self._evaluate_single_filter(metadata, filter_obj)
                results.append(result)
            except Exception as e:
                # Log error and treat as filter failure
                results.append(False)

        if mode.lower() == "or":
            return any(results)
        else:  # default to "and"
            return all(results)

    def _evaluate_single_filter(
        self, metadata: Dict[str, Any], filter_obj: MetadataFilter
    ) -> bool:
        """Evaluate a single metadata filter"""

        # Get the field value from metadata
        field_value = self._get_field_value(metadata, filter_obj.field)

        # Get the handler for this operator
        handler = self._operator_handlers.get(filter_obj.operator)
        if not handler:
            raise ValueError(f"Unsupported operator: {filter_obj.operator}")

        # Apply the filter
        return handler(field_value, filter_obj.value, filter_obj.case_sensitive)

    def _get_field_value(self, metadata: Dict[str, Any], field_path: str) -> Any:
        """
        Get field value from metadata, supporting nested paths

        Examples:
            "source" -> metadata["source"]
            "custom_fields.author" -> metadata["custom_fields"]["author"]
            "tags[0]" -> metadata["tags"][0]
        """
        try:
            # Handle nested field paths
            if "." in field_path:
                keys = field_path.split(".")
                value = metadata
                for key in keys:
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        return None
                return value

            # Handle array indexing
            if "[" in field_path and "]" in field_path:
                field_name = field_path.split("[")[0]
                index_str = field_path.split("[")[1].split("]")[0]

                try:
                    index = int(index_str)
                    array_value = metadata.get(field_name)
                    if isinstance(array_value, list) and 0 <= index < len(array_value):
                        return array_value[index]
                except (ValueError, IndexError):
                    pass

                return None

            # Simple field access
            return metadata.get(field_path)

        except Exception:
            return None

    # Filter Handlers

    def _handle_eq(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle equality comparison"""
        if field_value is None:
            return filter_value is None

        if isinstance(field_value, str) and isinstance(filter_value, str):
            if not case_sensitive:
                return field_value.lower() == filter_value.lower()

        return field_value == filter_value

    def _handle_ne(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle not equal comparison"""
        return not self._handle_eq(field_value, filter_value, case_sensitive)

    def _handle_gt(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle greater than comparison"""
        if field_value is None:
            return False

        try:
            # Handle datetime comparisons
            if isinstance(filter_value, str):
                field_value = self._parse_datetime_if_needed(field_value)
                filter_value = self._parse_datetime_if_needed(filter_value)

            return field_value > filter_value
        except (TypeError, ValueError):
            return False

    def _handle_gte(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle greater than or equal comparison"""
        return self._handle_gt(
            field_value, filter_value, case_sensitive
        ) or self._handle_eq(field_value, filter_value, case_sensitive)

    def _handle_lt(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle less than comparison"""
        if field_value is None:
            return False

        try:
            # Handle datetime comparisons
            if isinstance(filter_value, str):
                field_value = self._parse_datetime_if_needed(field_value)
                filter_value = self._parse_datetime_if_needed(filter_value)

            return field_value < filter_value
        except (TypeError, ValueError):
            return False

    def _handle_lte(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle less than or equal comparison"""
        return self._handle_lt(
            field_value, filter_value, case_sensitive
        ) or self._handle_eq(field_value, filter_value, case_sensitive)

    def _handle_in(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle 'in' list comparison"""
        if not isinstance(filter_value, (list, tuple, set)):
            return False

        if field_value is None:
            return None in filter_value

        if isinstance(field_value, str) and not case_sensitive:
            field_value = field_value.lower()
            filter_value = [
                v.lower() if isinstance(v, str) else v for v in filter_value
            ]

        return field_value in filter_value

    def _handle_not_in(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle 'not in' list comparison"""
        return not self._handle_in(field_value, filter_value, case_sensitive)

    def _handle_contains(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle string contains comparison"""
        if field_value is None or filter_value is None:
            return False

        field_str = str(field_value)
        filter_str = str(filter_value)

        if not case_sensitive:
            field_str = field_str.lower()
            filter_str = filter_str.lower()

        return filter_str in field_str

    def _handle_starts_with(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle string starts with comparison"""
        if field_value is None or filter_value is None:
            return False

        field_str = str(field_value)
        filter_str = str(filter_value)

        if not case_sensitive:
            field_str = field_str.lower()
            filter_str = filter_str.lower()

        return field_str.startswith(filter_str)

    def _handle_ends_with(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle string ends with comparison"""
        if field_value is None or filter_value is None:
            return False

        field_str = str(field_value)
        filter_str = str(filter_value)

        if not case_sensitive:
            field_str = field_str.lower()
            filter_str = filter_str.lower()

        return field_str.endswith(filter_str)

    def _handle_regex(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle regular expression matching"""
        if field_value is None or filter_value is None:
            return False

        try:
            field_str = str(field_value)
            pattern = str(filter_value)

            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(pattern, field_str, flags))
        except re.error:
            return False

    def _handle_exists(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle field existence check"""
        # For exists check, we consider None as non-existent
        return field_value is not None

    def _handle_not_exists(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle field non-existence check"""
        return not self._handle_exists(field_value, filter_value, case_sensitive)

    def _handle_date_range(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle date range comparison"""
        if field_value is None:
            return False

        try:
            # Parse field value as datetime
            field_dt = self._parse_datetime_if_needed(field_value)
            if field_dt is None:
                return False

            # Parse filter value as date range
            if isinstance(filter_value, dict):
                start_date = filter_value.get("start")
                end_date = filter_value.get("end")
            elif isinstance(filter_value, DateRange):
                start_date = filter_value.start
                end_date = filter_value.end
            else:
                return False

            # Check range
            if start_date and field_dt < self._parse_datetime_if_needed(start_date):
                return False

            if end_date and field_dt > self._parse_datetime_if_needed(end_date):
                return False

            return True

        except (ValueError, TypeError):
            return False

    def _handle_array_contains(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle array contains value check"""
        if not isinstance(field_value, (list, tuple)):
            return False

        for item in field_value:
            if self._handle_eq(item, filter_value, case_sensitive):
                return True

        return False

    def _handle_array_length(
        self, field_value: Any, filter_value: Any, case_sensitive: bool
    ) -> bool:
        """Handle array length check"""
        if not isinstance(field_value, (list, tuple)):
            return False

        try:
            expected_length = int(filter_value)
            return len(field_value) == expected_length
        except (ValueError, TypeError):
            return False

    # Utility Methods

    def _parse_datetime_if_needed(self, value: Any) -> Optional[datetime]:
        """Parse value as datetime if it's a string"""
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                # Try ISO format first
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    # Try common datetime formats
                    formats = [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%SZ",
                    ]
                    for fmt in formats:
                        try:
                            return datetime.strptime(value, fmt)
                        except ValueError:
                            continue
                except ValueError:
                    pass

        return None

    def create_date_range_filter(
        self,
        field: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> MetadataFilter:
        """Create a date range filter"""
        date_range = {"start": start_date, "end": end_date}
        return MetadataFilter(
            field=field, operator=FilterOperator.DATE_RANGE, value=date_range
        )

    def create_text_search_filter(
        self, field: str, search_term: str, case_sensitive: bool = False
    ) -> MetadataFilter:
        """Create a text search filter (contains)"""
        return MetadataFilter(
            field=field,
            operator=FilterOperator.CONTAINS,
            value=search_term,
            case_sensitive=case_sensitive,
        )

    def validate_filter(self, filter_obj: MetadataFilter) -> List[str]:
        """Validate a metadata filter and return any errors"""
        errors = []

        if not filter_obj.field:
            errors.append("Field name is required")

        if filter_obj.operator not in self._operator_handlers:
            errors.append(f"Unsupported operator: {filter_obj.operator}")

        # Operator-specific validation
        if filter_obj.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(filter_obj.value, (list, tuple)):
                errors.append(
                    f"Operator {filter_obj.operator.value} requires a list value"
                )

        if filter_obj.operator == FilterOperator.DATE_RANGE:
            if not isinstance(filter_obj.value, dict):
                errors.append(
                    "Date range operator requires a dict with 'start' and/or 'end' keys"
                )

        if filter_obj.operator == FilterOperator.REGEX:
            try:
                re.compile(str(filter_obj.value))
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")

        return errors


# Global filter engine instance
filter_engine = MetadataFilterEngine()
