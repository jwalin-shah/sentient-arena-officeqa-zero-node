from __future__ import annotations


def parse_schema(name: str = "officeqa_parse_v1") -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "metric",
                    "entity_category",
                    "time_period",
                    "operation",
                    "expected_answer_type",
                ],
                "properties": {
                    "metric": {"type": "string"},
                    "entity_category": {"type": "string"},
                    "time_period": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["lookup", "difference", "average", "ratio"],
                    },
                    "expected_answer_type": {"type": "string"},
                },
            },
        },
    }


def solve_schema(name: str = "officeqa_solve_v1") -> dict:
    evidence_row = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_file",
            "table_or_section",
            "row_label",
            "column_label",
            "raw_value",
            "unit",
            "matched_snippet",
        ],
        "properties": {
            "source_file": {"type": "string"},
            "table_or_section": {"type": "string"},
            "row_label": {"type": "string"},
            "column_label": {"type": "string"},
            "raw_value": {"type": ["string", "number"]},
            "unit": {"type": "string"},
            "matched_snippet": {"type": "string"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "final_answer",
                    "reasoning_summary",
                    "confidence",
                    "evidence_rows",
                ],
                "properties": {
                    "final_answer": {"type": ["string", "number"]},
                    "reasoning_summary": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_rows": {"type": "array", "items": evidence_row},
                },
            },
        },
    }


def grounding_judge_schema(name: str = "officeqa_grounding_judge_v1") -> dict:
    row_check = {
        "type": "object",
        "additionalProperties": False,
        "required": ["row_index", "grounded", "reason_codes", "confidence"],
        "properties": {
            "row_index": {"type": "integer"},
            "grounded": {"type": "boolean"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["overall_grounded", "summary", "row_checks"],
                "properties": {
                    "overall_grounded": {"type": "boolean"},
                    "summary": {"type": "string"},
                    "row_checks": {"type": "array", "items": row_check},
                },
            },
        },
    }
