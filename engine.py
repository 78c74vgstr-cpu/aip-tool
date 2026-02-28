import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import pandas as pd
import json
import time
import concurrent.futures
from threading import Lock
from openai import OpenAI

# ── Styles ────────────────────────────────────────────────────────────────────
YELLOW_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # updated cell
GREEN_FILL  = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # new task row
BLUE_FILL   = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")  # headers
RED_FILL    = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # manual review note
WHITE_FONT  = Font(color="FFFFFF", bold=True)
WRAP_ALIGN  = Alignment(wrap_text=True, vertical="top")

# ── AMP column → MPD column name mapping ─────────────────────────────────────
# AMP and MPD use different names for the same columns.
AMP_TO_MPD = {
    "TASK NUMBER":             "TASK NUMBER",
    "DESCRIPTION":             "DESCRIPTION",
    "ACCESS":                  "ACCESS",
    "PREPARATION":             "PREPARATION",
    "ZONE":                    "ZONE",
    "TASK CODE":               "TASK CODE",
    "100% THRESHOLD":          "100%\nTHRESHOLD",
    "INTERVAL":                "SAMPLE\nINTERVAL",
    "100% INTERVAL":           "100%\nINTERVAL",
    "SOURCE TASK\nREFERENCE":  "SOURCE TASK\nREFERENCE",
    "REFERENCE":               "REFERENCE",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_header_row(ws, max_search=20):
    best_row, best_count = 1, 0
    for row in ws.iter_rows(min_row=1, max_row=max_search):
        count = sum(1 for cell in row if cell.value not in (None, ""))
        if count > best_count:
            best_count = count
            best_row   = row[0].row
    headers = {}
    for cell in ws[best_row]:
        if cell.value not in (None, ""):
            headers[str(cell.value).strip()] = cell.column
    return best_row, headers


def build_task_index(ws, header_row, task_col_idx):
    index = {}
    for col in ws.iter_cols(min_col=task_col_idx, max_col=task_col_idx, min_row=header_row + 1):
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


def _format_unmappable(unmappable: list) -> str:
    """
    Format unmappable changes into a readable SOC Note.
    unmappable is a list of either:
      - dicts: {"field": ..., "old": ..., "new": ..., "is_mh": ...}
      - strings: legacy plain text (fallback)
    """
    lines = []
    for item in unmappable:
        if isinstance(item, dict):
            field  = item.get("field", "")
            old    = item.get("old", "").strip()
            new    = item.get("new", "").strip()
            is_mh  = item.get("is_mh", False)

            if is_mh:
                lines.append(f"• {field}: {old} → {new}  (update in planning system)")
            else:
                lines.append(f"• {field}:")
                lines.append(f"  WAS: {old}")
                lines.append(f"  NOW: {new}")
        else:
            lines.append(f"• {item}")
    return "\n".join(lines)


def normalize(val):
    import re
    return re.sub(r'\s+', ' ', str(val or "")).strip()


# ── SOC Parser (LLM) ──────────────────────────────────────────────────────────

_BATCH_SIZE  = 20
_TIMEOUT     = 90
_MAX_RETRIES = 3
_MAX_WORKERS = 5


def _call_llm(client, system_prompt, numbered, batch_num):
    for attempt in range(1, _MAX_RETRIES + 1):
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
                resp = future.result(timeout=_TIMEOUT)
            parsed = json.loads(resp.choices[0].message.content)
            return parsed.get("tasks", parsed if isinstance(parsed, list) else [])
        except concurrent.futures.TimeoutError:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} timeout (attempt {attempt}), retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[LLM] Batch {batch_num} error: {exc}, retrying in {wait}s")
            time.sleep(wait)
    print(f"[LLM] Batch {batch_num} permanently failed — skipping.")
    return []


def parse_soc_with_llm(soc_file_bytes, aip_columns: list, api_key: str,
                        progress_callback=None, checkpoint_state=None) -> list:
    """
    Parse SOC with GPT. Returns list of:
      {
        "task_id":         "200147-02-1",
        "mvt":             "R" | "N" | "D",
        "changed_columns": ["INTERVAL", "100% THRESHOLD"],  # exact AIP column names
        "unmappable":      ["APPLICABILITY", "TASK NOTE"],  # fields with no AIP column
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

Analyze rows from a Summary of Changes (SOC) and extract structured data.

The AIP uses these EXACT column names:
{aip_cols_json}

For each SOC row return:
- "task_id"        : Task ID from "TASK NUMBER" field
- "mvt"            : Movement code exactly as given — "R" (revised), "N" (new), or "D" (deleted)
- "changed_columns": List of AIP column names (exact spelling) that changed.
                     Map change descriptions to the closest matching AIP column name.
- "unmappable"     : List of objects for changes that have NO matching AIP column.
                     Each object must have:
                       "field"   : the field name (e.g. "APPLICABILITY", "TASK NOTE", "TASK MH")
                       "old"     : the old value exactly as written in the SOC
                       "new"     : the new value exactly as written in the SOC
                       "is_mh"   : true if this is a man-hours field (TASK MH, ACCESS MH, PREP MH, MEN),
                                   false otherwise
                     Examples of unmappable fields: APPLICABILITY, TASK NOTE, TASK MH, ACCESS MH,
                     IN PREPARATION MH, IN REFERENCE(S), TPS MARKER, SKILL

Rules:
- Return ONLY: {{"tasks": [...]}}
- Skip rows with no Task ID.
- Only use exact column names from the AIP list for changed_columns.
- Never put unmappable fields into changed_columns — they go into unmappable instead.
- For unmappable, extract the exact old and new values from the SOC description text.
"""

    batches = [(i // _BATCH_SIZE + 1,
                "\n".join(f"{j+1}. {r}" for j, r in enumerate(rows_text[i:i+_BATCH_SIZE])))
               for i in range(0, len(rows_text), _BATCH_SIZE)]

    total_batches = len(batches)
    completed     = {}

    if checkpoint_state is not None:
        completed = checkpoint_state.get("soc_checkpoint", {})
        if completed:
            print(f"[Checkpoint] Resuming — {len(completed)}/{total_batches} batches done.")

    remaining = [(num, text) for num, text in batches if num not in completed]
    lock      = Lock()

    def process(args):
        num, text = args
        tasks = _call_llm(client, system_prompt, text, num)
        with lock:
            completed[num] = tasks
            if checkpoint_state is not None:
                checkpoint_state["soc_checkpoint"] = completed.copy()
        return tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [executor.submit(process, args) for args in remaining]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            if progress_callback:
                progress_callback(len(completed), total_batches)

    all_results = []
    for num in sorted(completed.keys()):
        all_results.extend(completed[num])

    if checkpoint_state is not None:
        checkpoint_state.pop("soc_checkpoint", None)

    return all_results


# ── AIP Updater ───────────────────────────────────────────────────────────────

def update_aip(aip_file_bytes, mpd_df: pd.DataFrame,
               task_changes: list, task_id_col_name: str):
    """
    Update AIP from parsed SOC changes.

    - REV column added: R for revised, N for new, blank for unchanged
    - R tasks: changed cells updated (YELLOW), unmappable changes noted in SOC Note (RED)
    - N tasks: full row appended from MPD data (GREEN highlight), REV = N
               operator reviews applicability and deletes row if not applicable
    - D tasks: silently skipped
    - Tasks not in AMP: silently skipped (not applicable to us)

    Returns: (workbook, change_log, flagged)
    """
    wb = openpyxl.load_workbook(aip_file_bytes)
    ws = wb.active

    header_row, aip_headers = find_header_row(ws)

    if task_id_col_name not in aip_headers:
        raise ValueError(
            f"Column '{task_id_col_name}' not found in AIP.\n"
            f"Found: {list(aip_headers.keys())}"
        )

    task_col_idx   = aip_headers[task_id_col_name]
    aip_task_index = build_task_index(ws, header_row, task_col_idx)

    # ── MPD lookup ────────────────────────────────────────────────────────────
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
        raise ValueError(f"Task ID column not found in MPD.\nMPD columns: {list(mpd_df.columns)}")

    mpd_df               = mpd_df.copy()
    mpd_df[mpd_task_col] = mpd_df[mpd_task_col].astype(str).str.strip()
    mpd_lookup           = mpd_df.set_index(mpd_task_col).to_dict(orient="index")

    # ── Add REV and SOC Note columns ──────────────────────────────────────────
    max_col      = ws.max_column
    rev_col      = max_col + 1
    soc_note_col = max_col + 2

    for col_idx, label in [(rev_col, "REV"), (soc_note_col, "SOC Note")]:
        h      = ws.cell(row=header_row, column=col_idx, value=label)
        h.fill = BLUE_FILL
        h.font = WHITE_FONT

    ws.column_dimensions[get_column_letter(rev_col)].width      = 6
    ws.column_dimensions[get_column_letter(soc_note_col)].width = 55

    # ── Process each SOC entry ────────────────────────────────────────────────
    change_log = []
    flagged    = []

    for entry in task_changes:
        task_id         = str(entry.get("task_id", "")).strip()
        mvt             = str(entry.get("mvt", "R")).strip().upper()
        changed_columns = entry.get("changed_columns", [])
        unmappable      = entry.get("unmappable", [])

        if not task_id:
            continue

        # ── DELETED: skip silently ────────────────────────────────────────────
        if mvt == "D":
            continue

        # ── NEW TASK: insert full row from MPD ────────────────────────────────
        if mvt == "N":
            if task_id in aip_task_index:
                # Already in AMP somehow — just tag it
                ws.cell(row=aip_task_index[task_id], column=rev_col,
                        value="N").alignment = Alignment(horizontal="center")
                continue

            if task_id not in mpd_lookup:
                flagged.append({"Task ID": task_id, "MVT": "N",
                                "Issue": "New task not found in MPD"})
                continue

            mpd_row_data = mpd_lookup[task_id]
            new_row      = ws.max_row + 1

            # Write each AMP column from MPD
            for amp_col, mpd_col in AMP_TO_MPD.items():
                if amp_col not in aip_headers:
                    continue
                val = mpd_row_data.get(mpd_col, None)
                if isinstance(val, float) and pd.isna(val):
                    val = None
                cell      = ws.cell(row=new_row, column=aip_headers[amp_col], value=val)
                cell.fill = GREEN_FILL

            # Fill any remaining columns green for visual consistency
            for c in range(1, soc_note_col + 1):
                cell = ws.cell(row=new_row, column=c)
                rgb  = cell.fill.fgColor.rgb if cell.fill and cell.fill.fgColor else "00000000"
                if rgb in ("00000000", "FFFFFFFF"):
                    cell.fill = GREEN_FILL

            # REV = N
            rc           = ws.cell(row=new_row, column=rev_col, value="N")
            rc.fill      = GREEN_FILL
            rc.alignment = Alignment(horizontal="center")

            # SOC Note
            nc           = ws.cell(row=new_row, column=soc_note_col)
            nc.alignment = WRAP_ALIGN
            if unmappable:
                nc.value = "NEW — review applicability\n" + _format_unmappable(unmappable)
                nc.fill  = RED_FILL
            else:
                nc.value = "NEW — review applicability"
                nc.fill  = GREEN_FILL

            aip_task_index[task_id] = new_row
            change_log.append({
                "Task ID":   task_id,
                "Column":    "— NEW ROW —",
                "Old Value": "",
                "New Value": "Inserted from MPD",
            })
            continue

        # ── REVISED TASK ──────────────────────────────────────────────────────
        if task_id not in aip_task_index:
            # Not in our AMP — not applicable to us, skip silently
            continue

        aip_row = aip_task_index[task_id]

        # Tag REV = R
        rc           = ws.cell(row=aip_row, column=rev_col, value="R")
        rc.alignment = Alignment(horizontal="center")

        # Update changed columns from MPD
        if task_id in mpd_lookup:
            mpd_row_data = mpd_lookup[task_id]
            for amp_col in changed_columns:
                if amp_col not in aip_headers:
                    continue
                mpd_col = AMP_TO_MPD.get(amp_col)
                if not mpd_col or mpd_col not in mpd_row_data:
                    continue
                new_val = mpd_row_data[mpd_col]
                if isinstance(new_val, float) and pd.isna(new_val):
                    new_val = None
                cell = ws.cell(row=aip_row, column=aip_headers[amp_col])
                if normalize(cell.value) == normalize(new_val):
                    continue  # whitespace-only diff — skip
                cell.value = new_val
                cell.fill  = YELLOW_FILL
                change_log.append({
                    "Task ID":   task_id,
                    "Column":    amp_col,
                    "Old Value": normalize(cell.value),
                    "New Value": normalize(new_val),
                })

        # Unmappable changes → SOC Note (red)
        if unmappable:
            nc           = ws.cell(row=aip_row, column=soc_note_col)
            nc.value     = _format_unmappable(unmappable)
            nc.fill      = RED_FILL
            nc.alignment = WRAP_ALIGN

    # ── Change Log sheet ──────────────────────────────────────────────────────
    if "Change Log" in wb.sheetnames:
        del wb["Change Log"]
    log_ws = wb.create_sheet("Change Log")

    for c_idx, h in enumerate(["Task ID", "Column", "Old Value", "New Value"], 1):
        cell      = log_ws.cell(row=1, column=c_idx, value=h)
        cell.fill = BLUE_FILL
        cell.font = WHITE_FONT

    for r_idx, e in enumerate(change_log, 2):
        log_ws.cell(row=r_idx, column=1, value=e["Task ID"])
        log_ws.cell(row=r_idx, column=2, value=e["Column"])
        log_ws.cell(row=r_idx, column=3, value=e["Old Value"])
        log_ws.cell(row=r_idx, column=4, value=e["New Value"])

    for col in log_ws.columns:
        width = max((len(str(c.value or "")) for c in col), default=10) + 4
        log_ws.column_dimensions[get_column_letter(col[0].column)].width = min(width, 60)

    # ── Legend sheet ──────────────────────────────────────────────────────────
    if "Legend" in wb.sheetnames:
        del wb["Legend"]
    leg_ws = wb.create_sheet("Legend")
    leg_ws.column_dimensions["A"].width = 22
    leg_ws.column_dimensions["B"].width = 65

    leg_ws.cell(row=1, column=1, value="Colour Legend").font = Font(bold=True, size=13)
    for i, (fill, label, desc) in enumerate([
        (YELLOW_FILL, "Yellow cell",  "Cell updated automatically from MPD"),
        (GREEN_FILL,  "Green row",    "New task — review applicability, delete row if not applicable to your fleet"),
        (RED_FILL,    "Red SOC Note", "Manual action needed — shows exact old/new values for Applicability, Task Note, etc."),
    ], start=3):
        leg_ws.cell(row=i, column=1, value=label).fill = fill
        leg_ws.cell(row=i, column=2, value=desc).fill  = fill

    return wb, change_log, flagged
