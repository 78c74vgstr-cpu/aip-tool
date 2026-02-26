import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import pandas as pd
import json
from openai import OpenAI
import io
import time
import concurrent.futures
from threading import Lock

# ── Styles ────────────────────────────────────────────────────────────────────
YELLOW_FILL      = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # updated cell
BLUE_FILL        = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")  # log header
RED_FILL         = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")  # manual review needed
ORANGE_FILL      = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")  # description partial change
GREEN_FILL       = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")  # new task row
GREY_FILL        = PatternFill(start_color="BFBFBF", end_color="BFBFBF", fill_type="solid")  # deleted task row
SOC_NOTE_FILL    = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # SOC note column
WHITE_FONT       = Font(color="FFFFFF", bold=True)
WRAP_ALIGNMENT   = Alignment(wrap_text=True, vertical="top")

# Columns whose changes are "text diffs" — flag orange for manual review of exact wording
DESCRIPTION_LIKE_COLUMNS = {"description", "task description", "notes", "remarks", "comment", "procedure"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_header_row(ws, max_search=20):
    best_row_num, best_count = 1, 0
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
    for col in ws.iter_cols(min_col=task_col_idx, max_col=task_col_idx, min_row=header_row_num + 1):
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


def _is_description_column(col_name: str) -> bool:
    return col_name.strip().lower() in DESCRIPTION_LIKE_COLUMNS


# ── SOC Parser (LLM) ──────────────────────────────────────────────────────────

_BATCH_SIZE    = 20    # smaller = faster per call, less likely to time out
_CALL_TIMEOUT  = 90   # seconds per API call before we give up and retry
_MAX_RETRIES   = 3    # retry a failed/timed-out batch this many times
_MAX_WORKERS   = 5    # concurrent API calls (stay under OpenAI rate limits)


def _call_llm_with_retry(client, system_prompt: str, numbered: str,
                          batch_num: int, timeout: int, max_retries: int) -> list:
    """Call OpenAI with a hard timeout and exponential-backoff retry. Returns list of tasks."""
    for attempt in range(1, max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    client.chat.completions.create,
                    model           = "gpt-4o-mini",
                    messages        = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": f"Process these SOC rows:\n\n{numbered}"},
                    ],
                    response_format = {"type": "json_object"},
                    temperature     = 0,
                )
                resp = future.result(timeout=timeout)

            parsed = json.loads(resp.choices[0].message.content)
            tasks  = parsed.get("tasks", parsed if isinstance(parsed, list) else [])
            return tasks

        except concurrent.futures.TimeoutError:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} timed out (attempt {attempt}/{max_retries}), retrying in {wait}s…")
            time.sleep(wait)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} error (attempt {attempt}/{max_retries}): {exc}, retrying in {wait}s…")
            time.sleep(wait)

    print(f"[LLM] Batch {batch_num} permanently failed after {max_retries} attempts — skipping.")
    return []


def parse_soc_with_llm(
    soc_file_bytes,
    aip_columns: list,
    api_key: str,
    progress_callback=None,
    checkpoint_state: dict | None = None,   # pass st.session_state dict to enable resume
) -> list:
    """
    Parse SOC rows with GPT concurrently, with per-call timeouts, retries,
    and optional Streamlit session_state checkpointing so you can resume
    if the app restarts mid-run.

    checkpoint_state: if provided (e.g. st.session_state), completed batch
    results are saved under key "soc_parse_checkpoint" and resumed on restart.

    Returns list of dicts per task:
      {
        "task_id": "12345",
        "changed_columns": ["Interval", "Zone"],
        "unmappable_changes": ["Applicability revised"],
        "soc_note": "raw SOC text",
        "is_new_task": false,
        "is_deleted_task": false
      }
    """
    client = OpenAI(api_key=api_key)

    soc_df         = pd.read_excel(soc_file_bytes)
    soc_df.columns = [str(c).strip() for c in soc_df.columns]

    rows_text = []
    for _, row in soc_df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items()
                 if pd.notna(val) and str(val).strip()]
        rows_text.append(" | ".join(parts))

    aip_cols_json = json.dumps(aip_columns)

    system_prompt = f"""You are an expert in aircraft maintenance documentation.

Your job: analyze rows from a Summary of Changes (SOC) document and extract structured data.

The AIP uses these EXACT column names (use exact spelling when returning results):
{aip_cols_json}

For each SOC row return a JSON entry with these fields:
- "task_id"           : Task ID found in "Task", "Task No", "Task Number", or similar field
- "changed_columns"   : List of AIP column names (exact spelling) that changed — ONLY names from the AIP list above
- "unmappable_changes": Changes that CANNOT be mapped to any AIP column (e.g. "Applicability revised")
- "soc_note"          : Complete raw text of this SOC row exactly as provided
- "is_new_task"       : true if this task is entirely NEW
- "is_deleted_task"   : true if this task is DELETED / REMOVED

Rules:
- Return ONLY: {{"tasks": [...]}}
- Skip rows with no Task ID.
- Never invent column names — only use names from the AIP list for changed_columns.
"""

    # ── Build batches ─────────────────────────────────────────────────────────
    batches = []
    for i in range(0, len(rows_text), _BATCH_SIZE):
        chunk    = rows_text[i : i + _BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {r}" for j, r in enumerate(chunk))
        batches.append((i // _BATCH_SIZE + 1, numbered))   # (batch_num, text)

    total_batches = len(batches)

    # ── Resume from checkpoint if available ──────────────────────────────────
    completed: dict[int, list] = {}   # batch_num → tasks
    if checkpoint_state is not None:
        completed = checkpoint_state.get("soc_parse_checkpoint", {})
        if completed:
            print(f"[Checkpoint] Resuming — {len(completed)}/{total_batches} batches already done.")

    remaining_batches = [(num, text) for num, text in batches if num not in completed]

    # ── Concurrent processing ─────────────────────────────────────────────────
    lock = Lock()

    def process_batch(args):
        batch_num, numbered = args
        tasks = _call_llm_with_retry(
            client, system_prompt, numbered,
            batch_num, _CALL_TIMEOUT, _MAX_RETRIES
        )
        with lock:
            completed[batch_num] = tasks
            if checkpoint_state is not None:
                checkpoint_state["soc_parse_checkpoint"] = completed.copy()
            if progress_callback:
                progress_callback(len(completed), total_batches)
        return batch_num, tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        list(executor.map(process_batch, remaining_batches))

    # ── Reassemble in original order ──────────────────────────────────────────
    all_results = []
    for batch_num in sorted(completed.keys()):
        all_results.extend(completed[batch_num])

    # Clear checkpoint on successful completion
    if checkpoint_state is not None:
        checkpoint_state.pop("soc_parse_checkpoint", None)

    return all_results


# ── AIP Updater ───────────────────────────────────────────────────────────────

def update_aip(aip_file_bytes, mpd_df: pd.DataFrame,
               task_changes: list, task_id_col_name: str):
    """
    Surgically update AIP with enhanced audit trail:

    Behavior per task:
    - Updated cells           → YELLOW highlight
    - Description-like cols   → ORANGE highlight (partial text change, needs manual diff)
    - Unmappable changes      → RED highlight in SOC Note cell
    - New tasks (whole row)   → GREEN highlight
    - Deleted tasks (row)     → GREY highlight
    - No changes              → SOC Note says "No changes in this revision"
    - SOC Note column added   → last column, shows raw SOC text + unmappable change list

    Returns: (workbook, change_log_list, flagged_list)
    """
    wb = openpyxl.load_workbook(aip_file_bytes)
    ws = wb.active

    header_row_num, aip_headers = find_header_row(ws)

    if task_id_col_name not in aip_headers:
        raise ValueError(
            f"Column '{task_id_col_name}' not found in AIP.\n"
            f"Found columns: {list(aip_headers.keys())}"
        )

    task_col_idx   = aip_headers[task_id_col_name]
    aip_task_index = build_task_index(ws, header_row_num, task_col_idx)

    # ── MPD task ID column ────────────────────────────────────────────────────
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
    if mpd_task_col is None:
        raise ValueError(
            f"Task ID column not found in MPD.\nMPD columns: {list(mpd_df.columns)}"
        )

    mpd_df               = mpd_df.copy()
    mpd_df[mpd_task_col] = mpd_df[mpd_task_col].astype(str).str.strip()
    mpd_lookup           = mpd_df.set_index(mpd_task_col).to_dict(orient="index")

    # ── Add SOC Note column header ────────────────────────────────────────────
    max_col       = ws.max_column
    soc_note_col  = max_col + 1
    header_cell   = ws.cell(row=header_row_num, column=soc_note_col, value="SOC Note")
    header_cell.fill = BLUE_FILL
    header_cell.font = WHITE_FONT
    ws.column_dimensions[get_column_letter(soc_note_col)].width = 50

    # ── Build set of task IDs that appear in SOC ──────────────────────────────
    soc_task_ids = {str(entry.get("task_id", "")).strip() for entry in task_changes if entry.get("task_id")}

    # Mark tasks that had zero changes (exist in AIP but not in SOC) with a note
    for task_id, aip_row in aip_task_index.items():
        if task_id not in soc_task_ids:
            cell = ws.cell(row=aip_row, column=soc_note_col)
            cell.value     = "No changes in this revision"
            cell.fill      = SOC_NOTE_FILL
            cell.alignment = WRAP_ALIGNMENT

    # ── Apply changes ─────────────────────────────────────────────────────────
    change_log = []
    flagged    = []

    for entry in task_changes:
        task_id           = str(entry.get("task_id", "")).strip()
        changed_columns   = entry.get("changed_columns", [])
        unmappable        = entry.get("unmappable_changes", [])
        soc_note_text     = entry.get("soc_note", "")
        is_new_task       = entry.get("is_new_task", False)
        is_deleted_task   = entry.get("is_deleted_task", False)

        if not task_id:
            continue

        # ── NEW TASK: highlight full row green ────────────────────────────────
        if is_new_task:
            if task_id in aip_task_index:
                aip_row = aip_task_index[task_id]
                for c in range(1, ws.max_column + 1):
                    ws.cell(row=aip_row, column=c).fill = GREEN_FILL
                note = f"NEW TASK\n{soc_note_text}"
                note_cell = ws.cell(row=aip_row, column=soc_note_col)
                note_cell.value     = note
                note_cell.fill      = GREEN_FILL
                note_cell.alignment = WRAP_ALIGNMENT
            else:
                flagged.append({"Task ID": task_id, "Issue": "New task not found in AIP — add manually"})
            continue

        # ── DELETED TASK: highlight full row grey ─────────────────────────────
        if is_deleted_task:
            if task_id in aip_task_index:
                aip_row = aip_task_index[task_id]
                for c in range(1, ws.max_column + 1):
                    ws.cell(row=aip_row, column=c).fill = GREY_FILL
                note_cell = ws.cell(row=aip_row, column=soc_note_col)
                note_cell.value     = f"DELETED TASK — review for removal\n{soc_note_text}"
                note_cell.fill      = GREY_FILL
                note_cell.alignment = WRAP_ALIGNMENT
            else:
                flagged.append({"Task ID": task_id, "Issue": "Deleted task not found in AIP"})
            continue

        # ── REGULAR TASK ──────────────────────────────────────────────────────
        if task_id not in aip_task_index:
            flagged.append({"Task ID": task_id, "Issue": "Not found in AIP"})
            continue

        if task_id not in mpd_lookup:
            flagged.append({"Task ID": task_id, "Issue": "Not found in MPD"})
            continue

        aip_row      = aip_task_index[task_id]
        mpd_row_data = mpd_lookup[task_id]

        # Apply mapped column updates
        for col_name in changed_columns:
            if col_name not in aip_headers:
                flagged.append({"Task ID": task_id, "Issue": f"Column '{col_name}' not in AIP"})
                continue
            if col_name not in mpd_row_data:
                flagged.append({"Task ID": task_id, "Issue": f"Column '{col_name}' not in MPD"})
                continue

            aip_col_idx = aip_headers[col_name]
            cell        = ws.cell(row=aip_row, column=aip_col_idx)
            old_value   = cell.value
            new_value   = mpd_row_data[col_name]

            if isinstance(new_value, float) and pd.isna(new_value):
                new_value = None

            old_str = str(old_value).strip() if old_value is not None else ""
            new_str = str(new_value).strip() if new_value is not None else ""
            if old_str == new_str:
                continue

            cell.value = new_value

            # Description-like columns get orange (manual diff review); others get yellow
            if _is_description_column(col_name):
                cell.fill = ORANGE_FILL
            else:
                cell.fill = YELLOW_FILL

            change_log.append({
                "Task ID":   task_id,
                "Column":    col_name,
                "Old Value": old_str,
                "New Value": new_str,
            })

        # ── SOC Note cell ─────────────────────────────────────────────────────
        note_cell = ws.cell(row=aip_row, column=soc_note_col)
        note_parts = []
        if soc_note_text:
            note_parts.append(f"SOC: {soc_note_text}")
        if unmappable:
            note_parts.append("⚠ Manual review required:")
            note_parts.extend(f"  • {u}" for u in unmappable)

        if note_parts:
            note_cell.value     = "\n".join(note_parts)
            note_cell.alignment = WRAP_ALIGNMENT
            # If there are unmappable changes, highlight the note cell RED
            if unmappable:
                note_cell.fill = RED_FILL
            else:
                note_cell.fill = SOC_NOTE_FILL
        else:
            note_cell.value = "No changes in this revision"
            note_cell.fill  = SOC_NOTE_FILL
            note_cell.alignment = WRAP_ALIGNMENT

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

    for col in log_ws.columns:
        width = max((len(str(cell.value or "")) for cell in col), default=10) + 4
        log_ws.column_dimensions[get_column_letter(col[0].column)].width = min(width, 60)

    # ── Colour Legend sheet ───────────────────────────────────────────────────
    if "Legend" in wb.sheetnames:
        del wb["Legend"]

    legend_ws = wb.create_sheet("Legend")
    legend_ws.column_dimensions["A"].width = 25
    legend_ws.column_dimensions["B"].width = 55

    legend_title = legend_ws.cell(row=1, column=1, value="Colour Legend")
    legend_title.font = Font(bold=True, size=13)

    legend_data = [
        (YELLOW_FILL,   "Yellow",      "Cell value updated from MPD (interval, zone, etc.)"),
        (ORANGE_FILL,   "Orange",      "Description/text column changed — review wording manually"),
        (RED_FILL,      "Red",         "Change exists in SOC but no matching AIP column — manual action required"),
        (GREEN_FILL,    "Green",       "New task added — review applicability before keeping"),
        (GREY_FILL,     "Grey",        "Task deleted in SOC — review and remove if applicable"),
        (SOC_NOTE_FILL, "Light Green", "SOC Note — no special action needed"),
    ]

    for i, (fill, label, description) in enumerate(legend_data, start=3):
        color_cell       = legend_ws.cell(row=i, column=1, value=label)
        color_cell.fill  = fill
        desc_cell        = legend_ws.cell(row=i, column=2, value=description)
        desc_cell.fill   = fill

    return wb, change_log, flagged
