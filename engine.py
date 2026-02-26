import openpyxl
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter
import pandas as pd
import json
from openai import OpenAI
import io

# ── Styles ────────────────────────────────────────────────────────────────────
YELLOW_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
BLUE_FILL   = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
WHITE_FONT  = Font(color="FFFFFF", bold=True)


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


# ── SOC Parser (LLM) ──────────────────────────────────────────────────────────

def parse_soc_with_llm(soc_file_bytes, aip_columns: list,
                        api_key: str, progress_callback=None) -> list:
    client = OpenAI(api_key=api_key)

    soc_df         = pd.read_excel(soc_file_bytes)
    soc_df.columns = [str(c).strip() for c in soc_df.columns]

    rows_text = []
    for _, row in soc_df.iterrows():
        parts = [
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val) and str(val).strip()
        ]
        rows_text.append(" | ".join(parts))

    aip_cols_json = json.dumps(aip_columns)
    all_results   = []
    batch_size    = 50
    total_batches = (len(rows_text) + batch_size - 1) // batch_size

    system_prompt = f"""You are an expert in aircraft maintenance documentation.

Your job: analyze rows from a Summary of Changes (SOC) document and extract structured data.

The AIP uses these EXACT column names:
{aip_cols_json}

For each SOC row:
1. Extract the Task ID.
2. Identify which AIP columns changed.
3. Provide a concise "SOC Remark" summarizing the specific change described for audit purposes.

Rules:
- Return ONLY a JSON object: {{"tasks": [{{"task_id": "...", "changed_columns": ["Col1"], "soc_remark": "..."}}, ...]}}
- Use EXACT column names from the AIP list.
"""

    for batch_num, i in enumerate(range(0, len(rows_text), batch_size), start=1):
        batch    = rows_text[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {row}" for j, row in enumerate(batch))

        try:
            resp = client.chat.completions.create(
                model           = "gpt-4o-mini",
                messages        = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Process these SOC rows:\n\n{numbered}"},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            tasks  = parsed.get("tasks", [])
            all_results.extend(tasks)

        except Exception as exc:
            print(f"[LLM] Batch {batch_num} failed: {exc}")

        if progress_callback:
            progress_callback(batch_num, total_batches)

    return all_results


# ── AIP Updater ───────────────────────────────────────────────────────────────

def update_aip(aip_file_bytes, mpd_df: pd.DataFrame,
                task_changes: list, task_id_col_name: str):
    wb = openpyxl.load_workbook(aip_file_bytes)
    ws = wb.active

    header_row_num, aip_headers = find_header_row(ws)

    if task_id_col_name not in aip_headers:
        raise ValueError(f"Column '{task_id_col_name}' not found in AIP.")

    # ── Setup Audit Column at the end ─────────────────────────────────────────
    audit_col_name = "SOC Remarks"
    if audit_col_name in aip_headers:
        audit_col_idx = aip_headers[audit_col_name]
    else:
        audit_col_idx = max(aip_headers.values()) + 1
        header_cell = ws.cell(row=header_row_num, column=audit_col_idx, value=audit_col_name)
        header_cell.font = Font(bold=True)
        aip_headers[audit_col_name] = audit_col_idx

    task_col_idx   = aip_headers[task_id_col_name]
    aip_task_index = build_task_index(ws, header_row_num, task_col_idx)

    # ── MPD task ID column detection ──────────────────────────────────────────
    mpd_task_col = None
    for col in mpd_df.columns:
        if col.strip().lower() == task_id_col_name.strip().lower():
            mpd_task_col = col
            break
    if mpd_task_col is None:
        for col in mpd_df.columns:
            if "task" in col.lower():
                mpd_task_col = col
                break
    
    mpd_df = mpd_df.copy()
    mpd_df[mpd_task_col] = mpd_df[mpd_task_col].astype(str).str.strip()
    mpd_lookup = mpd_df.set_index(mpd_task_col).to_dict(orient="index")

    change_log = []
    flagged    = []

    # ── Apply changes ─────────────────────────────────────────────────────────
    for entry in task_changes:
        task_id         = str(entry.get("task_id", "")).strip()
        changed_columns = entry.get("changed_columns", [])
        soc_remark      = entry.get("soc_remark", "")

        if not task_id or task_id not in aip_task_index:
            if task_id: flagged.append({"Task ID": task_id, "Issue": "Not found in AIP"})
            continue

        aip_row = aip_task_index[task_id]
        
        # ── Write the Remark to the end column ────────────────────────────────
        if soc_remark:
            remark_cell = ws.cell(row=aip_row, column=audit_col_idx)
            remark_cell.value = soc_remark
            remark_cell.fill = YELLOW_FILL

        if task_id not in mpd_lookup:
            flagged.append({"Task ID": task_id, "Issue": "Not found in MPD"})
            continue

        mpd_row_data = mpd_lookup[task_id]

        for col_name in changed_columns:
            if col_name not in aip_headers or col_name not in mpd_row_data:
                continue

            aip_col_idx = aip_headers[col_name]
            cell        = ws.cell(row=aip_row, column=aip_col_idx)

            old_value = cell.value
            new_value = mpd_row_data[col_name]

            if isinstance(new_value, float) and pd.isna(new_value):
                new_value = None

            old_str = str(old_value).strip() if old_value is not None else ""
            new_str = str(new_value).strip() if new_value is not None else ""
            
            if old_str != new_str:
                cell.value = new_value
                cell.fill  = YELLOW_FILL

                change_log.append({
                    "Task ID":   task_id,
                    "Column":    col_name,
                    "Old Value": old_str,
                    "New Value": new_str,
                })

    # ── Change Log sheet ──────────────────────────────────────────────────────
    if "Change Log" in wb.sheetnames:
        del wb["Change Log"]

    log_ws      = wb.create_sheet("Change Log")
    log_headers = ["Task ID", "Column", "Old Value", "New Value"]

    for c_idx, h in enumerate(log_headers, 1):
        cell      = log_ws.cell(row=1, column=c_idx, value=h)
        cell.fill = BLUE_FILL
        cell.font = WHITE_FONT

    for r_idx, entry in enumerate(change_log, 2):
        log_ws.cell(row=r_idx, column=1, value=entry["Task ID"])
        log_ws.cell(row=r_idx, column=2, value=entry["Column"])
        log_ws.cell(row=r_idx, column=3, value=entry["Old Value"])
        log_ws.cell(row=r_idx, column=4, value=entry["New Value"])

    return wb, change_log, flagged
