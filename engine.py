import openpyxl
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter
import pandas as pd
import json
from openai import OpenAI
import io
import difflib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Styles ────────────────────────────────────────────────────────────────────
YELLOW_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
BLUE_FILL   = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
WHITE_FONT  = Font(color="FFFFFF", bold=True)


# ── Column Matcher ────────────────────────────────────────────────────────────

def find_best_column_match(aip_col: str, mpd_columns: list) -> str | None:
    aip_col_clean = aip_col.strip().lower()

    if aip_col in mpd_columns:
        return aip_col

    for mpd_col in mpd_columns:
        if mpd_col.strip().lower() == aip_col_clean:
            return mpd_col

    for mpd_col in mpd_columns:
        mpd_col_clean = mpd_col.strip().lower()
        if aip_col_clean in mpd_col_clean or mpd_col_clean in aip_col_clean:
            return mpd_col

    matches = difflib.get_close_matches(
        aip_col_clean,
        [c.strip().lower() for c in mpd_columns],
        n=1,
        cutoff=0.6
    )
    if matches:
        for mpd_col in mpd_columns:
            if mpd_col.strip().lower() == matches[0]:
                return mpd_col

    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_header_row(ws, max_search=20):
    best_row_num = 1
    best_count   = 0

    for row in ws.iter_rows(min_row=1, max_row=max_search):
        count = sum(1 for cell in row if cell.value not in (None, ""))
        if count > best_count:
            best_count   = count
            best_row_num = row[0].row

    headers = {}
    for cell in ws[best_row_num]:
        if cell.value not in (None, ""):
            headers[str(cell.value).strip()] = cell.column

    return best_row_num, headers


def build_task_index(ws, header_row_num, task_col_idx):
    index = {}
    for col in ws.iter_cols(
        min_col=task_col_idx,
        max_col=task_col_idx,
        min_row=header_row_num + 1
    ):
        for cell in col:
            if cell.value not in (None, ""):
                index[str(cell.value).strip()] = cell.row
    return index


def read_mpd_dataframe(file_bytes):
    raw        = pd.read_excel(file_bytes, header=None, nrows=25)
    header_idx = int(raw.notna().sum(axis=1).idxmax())

    file_bytes.seek(0)
    df         = pd.read_excel(file_bytes, header=header_idx)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ── SOC Batch Processor ───────────────────────────────────────────────────────

def process_batch(client, batch_rows, batch_num, aip_cols_json, system_prompt, max_retries=3):
    """
    Process a single batch of SOC rows with retry logic.
    """
    numbered = "\n".join(f"{j+1}. {row}" for j, row in enumerate(batch_rows))

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model           = "gpt-4o-mini",
                messages        = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Process these SOC rows:\n\n{numbered}"},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0,
                timeout         = 60,
            )
            parsed = json.loads(resp.choices[0].message.content)
            tasks  = parsed.get("tasks", parsed if isinstance(parsed, list) else [])
            return batch_num, tasks

        except Exception as exc:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} attempt {attempt+1} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"[LLM] Batch {batch_num} failed after {max_retries} attempts — skipping.")
    return batch_num, []


# ── SOC Parser (LLM) ──────────────────────────────────────────────────────────

def parse_soc_with_llm(soc_file_bytes, aip_columns: list,
                        api_key: str, progress_callback=None) -> list:
    client = OpenAI(api_key=api_key)

    # Read SOC
    soc_df         = pd.read_excel(soc_file_bytes)
    soc_df.columns = [str(c).strip() for c in soc_df.columns]

    # One text string per row
    rows_text = []
    for _, row in soc_df.iterrows():
        parts = [
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val) and str(val).strip()
        ]
        rows_text.append(" | ".join(parts))

    aip_cols_json = json.dumps(aip_columns)
    batch_size    = 150
    batches       = [
        rows_text[i : i + batch_size]
        for i in range(0, len(rows_text), batch_size)
    ]
    total_batches = len(batches)

    system_prompt = f"""You are an expert in aircraft maintenance documentation.

Your job: analyze rows from a Summary of Changes (SOC) document and extract structured data.

The AIP uses these EXACT column names (use exact spelling when returning results):
{aip_cols_json}

For each SOC row:
1. Extract the Task ID — found in a field labeled "Task", "Task No", "Task Number", or similar.
2. Identify which AIP columns changed — based on the change description text.
   Map the description to the closest matching AIP column name from the list above.
   For example if the SOC mentions "100% Interval" or "interval revised", map it to the AIP
   column that best represents interval, even if the name is slightly different.
3. Extract the full original change description text from the SOC row as "soc_text".
   This should be the raw description of what changed, exactly as written in the SOC.
   Include all relevant change details — do not summarize or shorten it.

Rules:
- Return ONLY a JSON object in this exact format:
  {{
    "tasks": [
      {{
        "task_id": "...",
        "changed_columns": ["Col1", "Col2"],
        "soc_text": "full original change description from the SOC row"
      }},
      ...
    ]
  }}
- Use EXACT column names from the AIP list above for changed_columns.
- If a Task ID is missing, skip that row.
- If a change cannot be mapped to any AIP column, still include the task with an empty changed_columns list but keep the soc_text.
"""

    # ── Parallel processing ───────────────────────────────────────────────────
    results_map   = {}
    completed     = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                process_batch,
                client,
                batch,
                batch_num,
                aip_cols_json,
                system_prompt
            ): batch_num
            for batch_num, batch in enumerate(batches, start=1)
        }

        for future in as_completed(futures):
            batch_num, tasks = future.result()
            results_map[batch_num] = tasks
            completed += 1

            if progress_callback:
                progress_callback(completed, total_batches)

    # ── Reassemble in order ───────────────────────────────────────────────────
    all_results = []
    for batch_num in sorted(results_map.keys()):
        all_results.extend(results_map[batch_num])

    return all_results


# ── AIP Updater ───────────────────────────────────────────────────────────────

def update_aip(aip_file_bytes, mpd_df: pd.DataFrame,
               task_changes: list, task_id_col_name: str):
    wb = openpyxl.load_workbook(aip_file_bytes)
    ws = wb.active

    header_row_num, aip_headers = find_header_row(ws)

    task_id_col_matched = find_best_column_match(task_id_col_name, list(aip_headers.keys()))
    if task_id_col_matched is None:
        raise ValueError(
            f"Column '{task_id_col_name}' not found in AIP.\n"
            f"Found columns: {list(aip_headers.keys())}"
        )

    task_col_idx   = aip_headers[task_id_col_matched]
    aip_task_index = build_task_index(ws, header_row_num, task_col_idx)

    mpd_task_col = find_best_column_match(task_id_col_name, list(mpd_df.columns))
    if mpd_task_col is None:
        raise ValueError(
            f"Task ID column not found in MPD.\n"
            f"MPD columns: {list(mpd_df.columns)}"
        )

    mpd_df               = mpd_df.copy()
    mpd_df[mpd_task_col] = mpd_df[mpd_task_col].astype(str).str.strip()
    mpd_lookup           = mpd_df.set_index(mpd_task_col).to_dict(orient="index")
    mpd_columns          = list(mpd_df.columns)

    change_log = []
    flagged    = []

    for entry in task_changes:
        task_id         = str(entry.get("task_id", "")).strip()
        changed_columns = entry.get("changed_columns", [])
        soc_text        = str(entry.get("soc_text", "")).strip()

        if not task_id:
            continue

        if task_id not in aip_task_index:
            flagged.append({
                "Task ID":         task_id,
                "Issue":           "Not found in AIP",
                "SOC Description": soc_text,
            })
            continue

        if task_id not in mpd_lookup:
            flagged.append({
                "Task ID":         task_id,
                "Issue":           "Not found in MPD",
                "SOC Description": soc_text,
            })
            continue

        aip_row      = aip_task_index[task_id]
        mpd_row_data = mpd_lookup[task_id]

        for col_name in changed_columns:
            aip_col_matched = find_best_column_match(col_name, list(aip_headers.keys()))
            if aip_col_matched is None:
                flagged.append({
                    "Task ID":         task_id,
                    "Issue":           f"Column '{col_name}' not found in AIP",
                    "SOC Description": soc_text,
                })
                continue

            mpd_col_matched = find_best_column_match(col_name, mpd_columns)
            if mpd_col_matched is None:
                mpd_col_matched = find_best_column_match(aip_col_matched, mpd_columns)
            if mpd_col_matched is None:
                flagged.append({
                    "Task ID":         task_id,
                    "Issue":           f"Column '{col_name}' not found in MPD",
                    "SOC Description": soc_text,
                })
                continue

            aip_col_idx = aip_headers[aip_col_matched]
            cell        = ws.cell(row=aip_row, column=aip_col_idx)

            old_value = cell.value
            new_value = mpd_row_data.get(mpd_col_matched)

            if isinstance(new_value, float) and pd.isna(new_value):
                new_value = None

            old_str = str(old_value).strip() if old_value is not None else ""
            new_str = str(new_value).strip() if new_value is not None else ""
            if old_str == new_str:
                continue

            cell.value = new_value
            cell.fill  = YELLOW_FILL

            change_log.append({
                "Task ID":         task_id,
                "AIP Column":      aip_col_matched,
                "MPD Column":      mpd_col_matched,
                "Old Value":       old_str,
                "New Value":       new_str,
                "SOC Description": soc_text,
            })

    # ── Change Log sheet ──────────────────────────────────────────────────────
    if "Change Log" in wb.sheetnames:
        del wb["Change Log"]

    log_ws      = wb.create_sheet("Change Log")
    log_headers = [
        "Task ID",
        "AIP Column",
        "MPD Column",
        "Old Value",
        "New Value",
        "SOC Description",
    ]

    for c_idx, h in enumerate(log_headers, 1):
        cell      = log_ws.cell(row=1, column=c_idx, value=h)
        cell.fill = BLUE_FILL
        cell.font = WHITE_FONT

    for r_idx, entry in enumerate(change_log, 2):
        log_ws.cell(row=r_idx, column=1, value=entry["Task ID"])
        log_ws.cell(row=r_idx, column=2, value=entry["AIP Column"])
        log_ws.cell(row=r_idx, column=3, value=entry["MPD Column"])
        log_ws.cell(row=r_idx, column=4, value=entry["Old Value"])
        log_ws.cell(row=r_idx, column=5, value=entry["New Value"])
        log_ws.cell(row=r_idx, column=6, value=entry["SOC Description"])

    for col in log_ws.columns:
        width = max((len(str(cell.value or "")) for cell in col), default=10) + 4
        log_ws.column_dimensions[
            get_column_letter(col[0].column)
        ].width = min(width, 60)

    return wb, change_log, flagged
