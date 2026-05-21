"""
ai_query.py
-----------
AIQueryEngine
    Translates a plain-English question into a pandas code snippet via the
    Gemini API, executes that code against the current DataFrame, and returns
    a structured result dict.

Public API
----------
    AIQueryEngine.query(question, df, history)  → dict
        Keys: answer, thought, code, result_df, error
"""

import json
import re

import numpy as np
import pandas as pd
import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL


class AIQueryEngine:
    """Wraps the Gemini chat API for text-to-pandas queries."""

    # ── Public entry-point ─────────────────────────────────────────────────────

    @staticmethod
    def query(question: str, df: pd.DataFrame, history: list) -> dict:
        if not GEMINI_API_KEY:
            return AIQueryEngine._error_result(
                "No Gemini API key set. Add GEMINI_API_KEY to your environment variables."
            )

        system_prompt  = AIQueryEngine._build_system_prompt(df)
        gemini_history = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
            for m in history
        ]

        raw = ""
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(temperature=0.1),
            )
            chat     = model.start_chat(history=gemini_history)
            response = chat.send_message(question)
            raw      = response.text.strip()

            parsed = AIQueryEngine._parse_response(raw)
            if parsed is None:
                return AIQueryEngine._error_result("Model did not return valid JSON.", code=raw)

            code = parsed.get("code", "").strip()
            code = code.replace("\\n", "\n").replace("\\'", "'")

            result_df = AIQueryEngine._execute_code(code, df)

            # Always derive the answer from the actual result_df — never trust
            # the model's pre-computed answer string for numeric/named results.
            auto = AIQueryEngine._auto_answer(result_df)
            if auto:
                answer_text = auto
            else:
                answer_text = parsed.get("answer", "")
                answer_text = AIQueryEngine._fix_answer_number(answer_text, result_df)

            return {
                "answer":    answer_text,
                "thought":   parsed.get("thought", ""),
                "code":      code,
                "result_df": result_df,
                "error":     None,
            }

        except json.JSONDecodeError as exc:
            return AIQueryEngine._error_result(f"JSON parse error: {exc}", code=raw)
        except Exception as exc:
            return AIQueryEngine._error_result(str(exc), code=raw)

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(df: pd.DataFrame) -> str:
        return f"""You are a Python/pandas expert assistant.
The user has a DataFrame called `df` with the following structure:
- Columns : {list(df.columns)}
- Dtypes  : {df.dtypes.to_dict()}
- Sample  : {df.head(3).to_dict()}

Rules:
1. Write pandas code that answers the question using the variable `df`.
2. Store the final result in a variable called `result_df` (always a DataFrame or None).
3. If the result is a single number or value, wrap it: result_df = pd.DataFrame([{{"result": <value>}}])
4. For ranking/top-N questions, sort descending and keep all relevant rows in result_df.
5. ALWAYS store the complete answer data in result_df — do not summarise in code, let result_df hold the full answer.
6. Return ONLY a valid JSON object — no markdown fences, no extra text — with these exact keys:
   - "thought" : your brief reasoning (1-2 sentences)
   - "code"    : the executable Python/pandas code. Use only single quotes for Python strings inside the code field. Never use double quotes inside Python string literals.
   - "answer"  : plain-English answer. CRITICAL: you MUST read the actual computed values from result_df to write this answer. Read result_df.iloc[0] for the top result. Never guess, hallucinate, or hardcode numbers — always derive from what your code computes.

CRITICAL JSON RULES:
- The entire response must be valid JSON.
- Never use unescaped double quotes inside string values.
- Never include raw newline characters inside string values — use \\n if needed.
- Use single quotes for all Python string literals in the code field.

Example output format:
{{"thought":"I will group by month and compute cancellation rate, then find the max.","code":"grouped = df.groupby('month_name').apply(lambda x: (x['status'] == 'Annulé').mean() * 100).reset_index()\\ngrouped.columns = ['month_name', 'cancellation_rate']\\nresult_df = grouped.sort_values('cancellation_rate', ascending=False)","answer":"The month with the highest cancellation rate is July, with a rate of 16.49%."}}"""

    @staticmethod
    def _parse_response(raw: str) -> dict | None:
        clean = re.sub(r"```(?:json|python)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return None

        json_str = match.group()

        # First attempt
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Second attempt: escape bare newlines / tabs
        json_str2 = re.sub(r"(?<!\\)\n", r"\\n", json_str)
        json_str2 = re.sub(r"(?<!\\)\t", r"\\t", json_str2)
        try:
            return json.loads(json_str2)
        except json.JSONDecodeError:
            pass

        # Fallback: regex extraction
        thought_m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
        answer_m  = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',  json_str)
        code_m    = re.search(r'"code"\s*:\s*"(.*?)"(?=\s*,\s*"(?:answer|thought)")', json_str, re.DOTALL)
        if not code_m:
            code_m = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
        return {
            "thought": thought_m.group(1) if thought_m else "",
            "code":    code_m.group(1).replace("\\n", "\n") if code_m else "",
            "answer":  answer_m.group(1) if answer_m else "",
        }

    @staticmethod
    def _execute_code(code: str, df: pd.DataFrame):
        if not code:
            return None
        local_vars = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, {}, local_vars)  # nosec
        result = local_vars.get("result_df", None)

        if isinstance(result, pd.Series):
            return result.to_frame()
        if isinstance(result, (int, float, str, bool, np.integer, np.floating)):
            return pd.DataFrame([{"result": result}])
        if isinstance(result, (list, tuple, dict, np.ndarray)):
            try:
                return pd.DataFrame(result)
            except Exception:
                return pd.DataFrame([{"result": str(result)}])
        if result is not None and not isinstance(result, pd.DataFrame):
            try:
                return pd.DataFrame([{"result": str(result)}])
            except Exception:
                return None
        return result

    @staticmethod
    def _auto_answer(result_df) -> str | None:
        """
        Generate a confident answer directly from result_df so the model's
        pre-written answer string is never trusted for numeric/named values.
        """
        if result_df is None or result_df.empty:
            return None

        # Single scalar result — e.g. result_df = pd.DataFrame([{"result": 42}])
        if result_df.shape == (1, 1) and result_df.columns[0] == "result":
            val = result_df.iloc[0, 0]
            if isinstance(val, (int, np.integer)):
                return f"The answer is **{int(val):,}**."
            if isinstance(val, (float, np.floating)):
                return f"The answer is **{float(val):,.2f}**."
            return f"The answer is **{val}**."

        # Multi-column result — read first row for the top answer
        if result_df.shape[0] >= 1:
            row   = result_df.iloc[0]
            parts = []
            for col, val in row.items():
                if isinstance(val, (int, np.integer)):
                    parts.append(f"**{col}**: {int(val):,}")
                elif isinstance(val, (float, np.floating)):
                    parts.append(f"**{col}**: {float(val):,.2f}")
                else:
                    parts.append(f"**{col}**: {val}")
            suffix = f" ({len(result_df):,} rows total)" if len(result_df) > 1 else ""
            return "Top result — " + " · ".join(parts) + suffix + "."

        return None

    @staticmethod
    def _fix_answer_number(answer: str, result_df) -> str:
        """
        Last-resort fix: replace any mismatched scalar in the model's answer
        string with the actual computed value from result_df.
        Only runs when _auto_answer returns None (e.g. empty result).
        """
        if result_df is None or result_df.empty:
            return answer
        try:
            if result_df.shape == (1, 1):
                actual = result_df.iloc[0, 0]
                if not isinstance(actual, (int, float, np.integer, np.floating)):
                    return answer
                actual_n = int(actual) if isinstance(actual, (int, np.integer)) else float(actual)
                nums = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
                if nums and float(nums[0]) != actual_n:
                    actual_str = (
                        str(int(actual_n))
                        if isinstance(actual_n, float) and actual_n == int(actual_n)
                        else str(actual_n)
                    )
                    answer = re.sub(r"\b" + re.escape(nums[0]) + r"\b", actual_str, answer, count=1)
        except Exception:
            pass
        return answer

    @staticmethod
    def _error_result(message: str, code: str = "") -> dict:
        return {
            "answer": "", "thought": "", "code": code,
            "result_df": None, "error": message,
        }